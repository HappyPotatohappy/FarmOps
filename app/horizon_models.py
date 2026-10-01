"""Independent, lazy-loaded immutable multi-horizon forecasting bundles.

Training never runs during serving. An experimental finite forecast is distinct
from a passed validation gate. Inference failures are not replaced by baselines.
"""
from __future__ import annotations
import copy
from datetime import timedelta
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from threading import RLock
import numpy as np
import pandas as pd
from .horizon_data import HORIZONS, encode_context, tree_features

REGISTRY_ID = 'BeeOPS_Horizon_Weight'


def train_lstm(x, y, epochs=20, seed=42):
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    import tensorflow as tf
    try:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except RuntimeError:
        pass
    tf.keras.utils.set_random_seed(seed)
    model = tf.keras.Sequential([tf.keras.layers.Input(shape=x.shape[1:]),
        tf.keras.layers.LSTM(16), tf.keras.layers.Dense(32, activation='tanh'),
        tf.keras.layers.Dense(y.shape[1])])
    def observed_target_mae(actual, predicted):
        observed=tf.math.is_finite(actual)
        safe_actual=tf.where(observed,actual,predicted)
        errors=tf.where(observed,tf.abs(safe_actual-predicted),tf.zeros_like(predicted))
        return tf.math.divide_no_nan(tf.reduce_sum(errors,axis=-1),
                                    tf.reduce_sum(tf.cast(observed,predicted.dtype),axis=-1))
    model.compile(optimizer=tf.keras.optimizers.Adam(.001), loss=observed_target_mae)
    # Explicit batches avoid a private tf.data threadpool in constrained runtimes.
    rng, losses = np.random.default_rng(seed), []
    for _ in range(epochs):
        order = rng.permutation(len(x)); model.reset_metrics()
        loss = None
        for start in range(0, len(x), 32):
            indices = order[start:start+32]
            loss = model.train_on_batch(x[indices], y[indices])
        losses.append(float(np.asarray(loss)))
    return model, {'epochs_completed': epochs, 'training_mae_scaled': losses,
                   'architecture': 'LSTM(16)-Dense(32,tanh)-Dense(168), direct relative weight deltas', 'seed': seed}


def train_lightgbm(samples, seed=42):
    import lightgbm as lgb
    hours = np.unique(np.r_[1, np.arange(4, 169, 4), 24, 72, 168])
    x = np.concatenate([tree_features(s.context, s.origin, hours[hours<=len(s.target)]) for s in samples])
    y = np.concatenate([s.target[hours[hours<=len(s.target)]-1] - s.context[-1,0] for s in samples])
    model = lgb.LGBMRegressor(n_estimators=160, learning_rate=.035, num_leaves=15,
        min_child_samples=40, reg_lambda=3., verbosity=-1, random_state=seed, n_jobs=1)
    model.fit(x, y)
    return model.booster_, {'training_rows': len(y), 'training_horizons': hours.tolist(),
        'method': 'Direct horizon-conditioned delta regression; no recursive target lags', 'seed': seed}


class ChronosAdapter:
    def __init__(self, local_path):
        self.path = Path(local_path)
        if not self.path.is_dir():
            raise FileNotFoundError(f'Local Chronos-2 checkpoint unavailable: {self.path}')

    def predict(self, contexts, horizon=168):
        # TensorFlow/LightGBM and Torch native runtimes conflict on some macOS
        # builds. A separate interpreter also contains hard native failures:
        # they become explicit inference errors instead of killing the API.
        with tempfile.TemporaryDirectory(prefix='beeops-chronos-') as folder:
            input_path, output_path = Path(folder)/'input.npy', Path(folder)/'output.npy'
            np.save(input_path,np.stack(contexts),allow_pickle=False)
            result = subprocess.run([sys.executable,'-m','app.horizon_models','--chronos-worker',
                str(self.path.resolve()),str(input_path),str(output_path),str(horizon)],
                cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,timeout=300,
                env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'})
            if result.returncode != 0:
                raise RuntimeError(f'Chronos subprocess failed ({result.returncode}): {result.stderr[-2000:]}')
            return np.load(output_path,allow_pickle=False)


def select_weights(predictions, validation_targets, sample_weights=None, minimum_lstm=.10):
    """Inputs are development validation only. Calibration/test are not accepted.

    Fixed 0.05 simplex grid, a common weight across horizons when data are small.
    The floor is an engineering requirement, not an empirical biological rule.
    """
    names = sorted(predictions)
    if 'lstm' not in names:
        raise ValueError('An actual lstm prediction is required')
    y = np.asarray(validation_targets)
    if y.size == 0 or not np.isfinite(y).all():
        raise ValueError('Finite development validation targets are required')
    if any(np.shape(p) != y.shape or not np.isfinite(p).all() for p in predictions.values()):
        raise ValueError('Candidate predictions must share finite validation origins and targets')
    best, best_score = None, float('inf')
    for units in itertools.product(range(21), repeat=len(names)-1):
        last = 20 - sum(units)
        if last < 0:
            continue
        weights = dict(zip(names, [u/20 for u in (*units, last)]))
        if weights['lstm'] + 1e-12 < minimum_lstm:
            continue
        error = np.abs(combine_predictions(predictions, weights) - y)
        per_origin = error.mean(axis=tuple(range(1,error.ndim))) if error.ndim > 1 else error
        score = float(np.average(per_origin, weights=sample_weights))
        if score < best_score - 1e-12:
            best, best_score = weights, score
    if best is None:
        raise ValueError('No feasible ensemble weight')
    return best


def combine_predictions(predictions, weights):
    if not weights or any(not np.isfinite(v) or v < 0 for v in weights.values()) or abs(sum(weights.values())-1) > 1e-8:
        raise ValueError('Weights must be finite, nonnegative and sum to one')
    missing=[name for name,weight in weights.items() if weight>0 and name not in predictions]
    if missing:
        raise ValueError(f'Missing active component predictions: {missing}')
    return sum(np.asarray(predictions[name]) * weight for name, weight in weights.items() if weight>0)


def quality_gate(per_hive, minimum_independent=5):
    """Prespecified equal-hive 5% gain; no hive may regress more than 5%."""
    reasons = []
    if not per_hive:
        return {'status': 'insufficient', 'reasons': ['fewer_than_five_independent_windows_per_hive']}
    ensemble = float(np.mean([s['ensemble_mae'] for s in per_hive.values()]))
    baseline = float(np.mean([s['baseline_mae'] for s in per_hive.values()]))
    if any(s['independent_windows'] < minimum_independent for s in per_hive.values()):
        return {'status':'insufficient','reasons':['fewer_than_five_independent_windows_per_hive'],
                'equal_hive_mae_kg':ensemble,'equal_hive_baseline_mae_kg':baseline,
                'relative_improvement':1-ensemble/baseline if baseline>1e-10 else None}
    if baseline <= 1e-10 or ensemble > baseline*.95:
        reasons.append('equal_hive_gain_below_five_percent')
    if any(s['ensemble_mae'] > s['baseline_mae']*1.05 + 1e-10 for s in per_hive.values()):
        reasons.append('individual_hive_regression_above_five_percent')
    return {'status': 'underperforming' if reasons else 'passed', 'reasons': reasons,
            'equal_hive_mae_kg': ensemble, 'equal_hive_baseline_mae_kg': baseline,
            'relative_improvement': 1-ensemble/baseline if baseline > 1e-10 else None}


class HorizonModelService:
    def __init__(self, artifact_root: Path):
        self.artifact_root = Path(artifact_root).resolve()
        self._models, self._lock = {}, RLock()

    def _path(self, relative, base=None):
        root = (base or self.artifact_root).resolve()
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Artifact path escapes bundle')
        return path

    def _manifest(self, version=None):
        path = self.artifact_root / 'index.json'
        if not path.exists():
            return None, None
        index = json.loads(path.read_text())
        chosen = str(version) if version is not None else str(index.get('default_version'))
        entry = next((v for v in index.get('versions', []) if str(v['version']) == chosen), None)
        if entry is None:
            return None, None
        manifest_path = self._path(entry['manifest'])
        if not manifest_path.is_file():
            return None, None
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('registry_id') != REGISTRY_ID or str(manifest.get('version')) != chosen:
            raise ValueError('Artifact registry/version mismatch')
        weights = manifest['weights']
        if weights.get('lstm', 0) < .05 or set(weights) != set(manifest['components']):
            raise ValueError('Bundle must retain LSTM with weight at least 0.05')
        if any(not np.isfinite(v) or v < 0 for v in weights.values()) or abs(sum(weights.values())-1) > 1e-8:
            raise ValueError('Invalid ensemble weights')
        return manifest, manifest_path.parent

    def catalog(self):
        index_path = self.artifact_root / 'index.json'
        if not index_path.exists():
            return {'registry_id': REGISTRY_ID, 'default_version': None, 'versions': [], 'status': 'model_unavailable'}
        index = json.loads(index_path.read_text())
        versions = []
        for entry in index.get('versions', []):
            manifest, _ = self._manifest(str(entry['version']))
            if manifest:
                versions.append({key: copy.deepcopy(manifest.get(key)) for key in
                    ('registry_id','version','run_id','context_hours','max_horizon_hours','components','weights','validation','created_at','evaluation_protocol')})
        return {'registry_id': REGISTRY_ID, 'default_version': index.get('default_version'),
                'supported_horizons': list(HORIZONS), 'versions': versions, 'status': 'ok' if versions else 'model_unavailable'}

    def active_version(self):
        """Select the current shared TiRex/LSTM family, with the base v2 fallback."""
        index_path = self.artifact_root / 'index.json'
        if not index_path.is_file():
            return None
        index = json.loads(index_path.read_text())
        for version in dict.fromkeys((str(index.get('default_version') or ''), '2')):
            manifest, folder = self._manifest(version)
            if manifest is None or set(manifest['components']) != {'lstm', 'tirex2'}:
                continue
            weights = manifest['weights']
            if not (.05 <= weights['lstm'] <= .95 and .05-1e-8 <= weights['tirex2'] <= .95+1e-8):
                continue
            lstm = self._path(manifest['files']['lstm'], folder)
            tirex = self._path(manifest['files']['tirex2'], folder)
            if (manifest.get('run_id') and lstm.is_file()
                    and all((tirex / name).is_file() for name in ('model.ckpt', 'model-config.yaml'))):
                return version
        return None

    def _loaded_components(self, manifest, folder):
        """Called under the model lock; both inference paths share this cache."""
        key = str(manifest['version'])
        if key not in self._models:
            loaded = {}
            for name in manifest['components']:
                if manifest['weights'][name] == 0:
                    continue
                path = self._path(manifest['files'][name], folder)
                if not path.exists():
                    raise FileNotFoundError(f'Missing {name} artifact')
                if name == 'lstm':
                    import tensorflow as tf
                    loaded[name] = tf.keras.models.load_model(path, compile=False)
                elif name == 'lightgbm':
                    import lightgbm as lgb
                    loaded[name] = lgb.Booster(model_file=str(path))
                elif name == 'chronos2':
                    loaded[name] = ChronosAdapter(path)
                elif name == 'tirex2':
                    from .tirex_adapter import TirexAdapter
                    loaded[name] = TirexAdapter(path)
                else:
                    raise ValueError(f'Unsupported component {name}')
            self._models[key] = loaded
        return self._models[key]

    def _predict_component_batch(self, manifest, folder, contexts, horizon, origins):
        with self._lock:
            loaded = self._loaded_components(manifest, folder)
            predictions = {}
            for name in manifest['components']:
                if manifest['weights'][name] == 0:
                    continue
                elif name == 'lstm':
                    x = encode_context(contexts, manifest['scaler'])
                    deltas = np.asarray(loaded[name](x, training=False))
                    if deltas.shape != (len(contexts), manifest['max_horizon_hours']):
                        raise ValueError('LSTM predictions have an invalid horizon shape')
                    predictions[name] = deltas[:, :horizon] * manifest['scaler']['weight_scale'] + contexts[:, -1, 0:1]
                elif name == 'lightgbm':
                    if origins is None or len(origins) != len(contexts):
                        raise ValueError('LightGBM batch contexts require matching origins')
                    features = np.concatenate([tree_features(context, origin, np.arange(1, horizon+1))
                        for context, origin in zip(contexts, origins)])
                    predictions[name] = loaded[name].predict(features, num_threads=1).reshape(len(contexts), horizon) + contexts[:, -1, 0:1]
                elif name in ('chronos2', 'tirex2'):
                    predictions[name] = loaded[name].predict(contexts, horizon)
                values = np.asarray(predictions[name])
                if values.shape != (len(contexts), horizon) or not np.isfinite(values).all():
                    raise ValueError(f'{name} predictions must have the expected finite batch shape')
            return predictions

    def _predict_components(self, manifest, folder, context, origin):
        predictions = self._predict_component_batch(manifest, folder, context[None],
            manifest['max_horizon_hours'], [origin])
        return {name: values[0] for name, values in predictions.items()}

    def predict_batch(self, contexts, horizon_hours=1, version=None, origins=None):
        """Forecast independent contexts in one call per active component.

        Inputs have shape (N, context_hours, 2), with weight then already observed
        temperature. Callers own timestamp alignment and must exclude targets.
        TiRex/Chronos receive the requested horizon directly; a one-hour result
        is not asserted to equal the first point of a 168-hour generation.
        """
        manifest, folder = self._manifest(version)
        if manifest is None:
            raise FileNotFoundError('Requested batch model version is unavailable')
        if isinstance(horizon_hours, bool) or horizon_hours not in HORIZONS or horizon_hours > manifest['max_horizon_hours']:
            raise ValueError('Supported forecast horizons: 1, 24, 72, 168 hours')
        values = np.asarray(contexts, dtype=np.float32)
        if values.ndim != 3 or not len(values) or values.shape[1:] != (manifest['context_hours'], 2):
            raise ValueError('Batch contexts must have shape (N, context_hours, 2)')
        if not np.isfinite(values).all():
            raise ValueError('Batch contexts must contain finite values')
        if ((values[:, :, 0] <= 0) | (values[:, :, 0] > 300)).any() or ((values[:, :, 1] < -50) | (values[:, :, 1] > 60)).any():
            raise ValueError('Batch contexts must be within supported physical bounds')
        predictions = self._predict_component_batch(manifest, folder, values, horizon_hours, origins)
        result = np.asarray(combine_predictions(predictions, manifest['weights']))
        if result.shape != (len(values), horizon_hours) or not np.isfinite(result).all():
            raise ValueError('Batch predictions must have the expected finite horizon shape')
        return result

    def predict(self, rows: list[dict], horizon_hours: int, version: str | None = None):
        manifest, folder = self._manifest(version)
        model = ({k: copy.deepcopy(manifest[k]) for k in ('registry_id','version','run_id','context_hours','max_horizon_hours','components','weights')}
                 if manifest else {'registry_id': REGISTRY_ID, 'version': version})
        result = {'status': 'model_unavailable', 'model': model, 'trajectory': [],
                  'validation': {'status': 'insufficient','calibrated': False,'noise_kg': None}, 'reasons': []}
        if horizon_hours not in HORIZONS or (manifest and horizon_hours > manifest['max_horizon_hours']):
            result.update(status='unsupported_horizon', reasons=['supported_horizons_are_1_24_72_168'])
            return result
        if manifest is None:
            result['reasons'] = ['requested_model_version_or_artifacts_unavailable']
            return result
        validation = copy.deepcopy(manifest['validation'])
        validation['context_notes'] = []
        validation['role'] = 'informational'
        specific = validation.get('horizons', {}).get(str(horizon_hours), {})
        validation.update({k: v for k,v in specific.items() if k in ('status','calibrated','reasons','independent_windows','origin_count')})
        result['validation'] = validation
        n = int(manifest['context_hours'])
        if len(rows) < n:
            result.update(status='insufficient_history', reasons=[f'requires_{n}_continuous_hours'])
            return result
        context_rows = rows[-n:]
        frame = pd.DataFrame(context_rows)
        frame['timestamp'] = pd.to_datetime(frame.timestamp, utc=True, errors='raise')
        if frame.hive_id.nunique() != 1 or not frame.timestamp.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all():
            result.update(status='insufficient_history', reasons=['context_is_not_one_continuous_hive'])
            return result
        validation['input_event_counts'] = {str(event):int(count) for event,count in
            frame.loc[frame.event.ne('normal'),'event'].value_counts().items()}
        context = frame[['weight_kg','temperature_c']].to_numpy(np.float32)
        if not np.isfinite(context).all():
            raise ValueError('Context must contain finite values')
        if ((context[:,0] <= 0) | (context[:,0] > 300)).any() or ((context[:,1] < -50) | (context[:,1] > 60)).any():
            raise ValueError('Context observations must be within supported physical bounds')
        hive = str(frame.hive_id.iloc[-1]); origin = frame.timestamp.iloc[-1]
        hive_evidence = specific.get('per_hive', {}).get(hive)
        if 'per_hive' in specific and hive_evidence is None:
            validation.update(status='insufficient', calibrated=False)
            validation['context_notes'].append('hive_has_no_independent_validation')
        elif hive_evidence:
            validation['hive_evidence'] = hive_evidence
            if hive_evidence.get('independent_windows', 0) < 5:
                validation['status'] = 'insufficient'
            if hive_evidence.get('ensemble_mae', 0) > hive_evidence.get('baseline_mae', 0)*1.05:
                validation['status'] = 'underperforming'
        cutoff = manifest.get('bundle_available_after') or manifest.get('training_calendar_cutoff')
        if 'training_hive_ids' in manifest and hive not in manifest['training_hive_ids']:
            validation.update(status='insufficient', calibrated=False)
            validation['context_notes'].append('hive_was_not_represented_in_model_training')
        if cutoff and origin <= pd.Timestamp(cutoff):
            validation.update(status='insufficient', calibrated=False)
            if manifest.get('bundle_available_after'):
                validation['context_notes'].append('model_selection_calibration_or_quality_assessment_postdates_origin')
            else:
                validation['context_notes'].append('retrospective_transfer_model_training_postdates_forecast_origin')
        try:
            predictions = self._predict_components(manifest, folder, context, origin)
        except (ImportError, FileNotFoundError) as exc:
            result['reasons'].append(f'model_dependency_or_artifact_unavailable: {exc}')
            return result
        trajectory = combine_predictions(predictions, manifest['weights'])
        if trajectory.shape != (manifest['max_horizon_hours'],) or not np.isfinite(trajectory).all():
            raise ValueError('Model predictions must have the expected finite horizon shape')
        radii = manifest.get('interval_radii_kg', [])
        result['trajectory'] = [{'timestamp': (origin + timedelta(hours=i+1)).isoformat(),
            'weight_kg': float(value),
            'lower_kg': float(value-radii[i]) if validation.get('calibrated') and len(radii)>i and radii[i] is not None else None,
            'upper_kg': float(value+radii[i]) if validation.get('calibrated') and len(radii)>i and radii[i] is not None else None}
            for i,value in enumerate(trajectory[:horizon_hours])]
        result['status'] = 'ok'
        return result


def _chronos_worker(checkpoint, input_path, output_path, horizon):
    from chronos import Chronos2Pipeline
    import torch
    torch.set_num_threads(2)
    pipeline=Chronos2Pipeline.from_pretrained(checkpoint,device_map='cpu',local_files_only=True)
    contexts=np.load(input_path,allow_pickle=False)
    inputs=[{'target':x[:,0].astype(np.float32),'past_covariates':{'temperature_c':x[:,1].astype(np.float32)}} for x in contexts]
    _,medians=pipeline.predict_quantiles(inputs,prediction_length=int(horizon),quantile_levels=[.1,.5,.9],batch_size=32,cross_learning=False)
    np.save(output_path,np.stack([value[0].detach().cpu().numpy() for value in medians]),allow_pickle=False)


if __name__=='__main__':
    if len(sys.argv)==6 and sys.argv[1]=='--chronos-worker':
        _chronos_worker(*sys.argv[2:])
    else:
        raise SystemExit('Only the internal --chronos-worker entry point is supported')
