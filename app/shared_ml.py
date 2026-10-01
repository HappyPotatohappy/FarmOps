"""One immutable LSTM registry for all hives; split and window before pooling."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import logging
from pathlib import Path
import tempfile
from threading import Lock, RLock
import numpy as np
import pandas as pd
from mlflow import keras as mlflow_keras
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
import tensorflow as tf
from .ml import (ALIAS, FEATURES, WINDOW, MINIMUM_ROWS, FORECAST_POLICY, QUALITY_DEMO_POLICY,
                 QUALITY_DEMO_REASON, COMPARISON_REASON, Bundle, ModelService, prepare_splits,
                 make_windows, baseline_scores, evaluate_gate)

LOGGER = logging.getLogger(__name__)
REGISTRY_NAME = 'BeeOPS_Common_Weight'

@dataclass(frozen=True)
class PooledSplits:
    windows: dict[str, np.ndarray]
    targets: dict[str, np.ndarray]
    hive_ids: dict[str, np.ndarray]
    mean: np.ndarray
    scale: np.ndarray
    per_hive_ranges: dict[str, dict]
    snapshot_id: str


def prepare_pool(frames, after=None, skip_insufficient=False):
    """First eligible clean block per hive, capped672, with disjoint partitions.

    Scaler statistics use train rows only. Input windows never cross hive,
    partition, operation or gap boundaries. Years differ across source hives.
    """
    windows = {name: [] for name in ('train', 'validation', 'test')}
    targets = {name: [] for name in windows}
    hive_ids = {name: [] for name in windows}
    ranges, training_rows = {}, []
    for hive_id, frame in sorted(frames.items()):
        required = {'timestamp', 'hive_id', 'event', *FEATURES}
        if not required.issubset(frame.columns):
            raise ValueError(f'Training frame requires columns: {sorted(required)}')
        data = frame.copy(deep=True)
        if data.empty or data.hive_id.isna().any() or set(data.hive_id.astype(str)) != {str(hive_id)}:
            raise ValueError(f'Training frame hive identity does not match {hive_id}')
        data['timestamp'] = pd.to_datetime(data.timestamp, utc=True, errors='raise')
        if data.timestamp.isna().any() or data.timestamp.duplicated().any() or not data.timestamp.is_monotonic_increasing:
            raise ValueError('Training timestamps must be unique and ordered')
        if not data.timestamp.eq(data.timestamp.dt.floor('h')).all():
            raise ValueError('Training timestamps must fall on exact hours')
        values = data[FEATURES].to_numpy(dtype=float)
        if not np.isfinite(values).all() or ((values[:, 0] <= 0) | (values[:, 0] > 300)).any() or ((values[:, 1] < -50) | (values[:, 1] > 60)).any():
            raise ValueError('Training features must be finite and within physical bounds')
        if after and after.get(hive_id):
            data = data[data.timestamp > pd.Timestamp(after[hive_id])]
        normal = data[data.event.eq('normal')]
        blocks = normal.timestamp.diff().ne(pd.Timedelta(hours=1)).cumsum()
        eligible = [block for _, block in normal.groupby(blocks) if len(block) >= MINIMUM_ROWS]
        if not eligible:
            if skip_insufficient:
                continue
            raise ValueError(f'Hive {hive_id} requires at least {MINIMUM_ROWS} contiguous clean hourly rows')
        selected = eligible[0].iloc[:672].reset_index(drop=True)
        parts = prepare_splits(selected)
        digest = hashlib.sha256(pd.util.hash_pandas_object(selected, index=False).values.tobytes()).hexdigest()
        info = {'snapshot_id': digest, 'rows': len(selected),
                'data_source': str(frame.attrs.get('source', 'uploaded_or_generated'))}
        for name in windows:
            block = getattr(parts, name)
            x, y = make_windows(block)
            windows[name].append(x); targets[name].append(y)
            hive_ids[name].append(np.full(len(y), str(hive_id)))
            info.update({f'{name}_start': block.timestamp.iloc[0].isoformat(),
                         f'{name}_end': block.timestamp.iloc[-1].isoformat(),
                         f'{name}_target_start': block.timestamp.iloc[WINDOW].isoformat(),
                         f'{name}_rows': len(block), f'{name}_samples': len(y)})
        info['monitoring_after'] = info['test_end']
        ranges[str(hive_id)] = info
        training_rows.append(parts.train[FEATURES].to_numpy(dtype=np.float64))
    if not ranges:
        raise ValueError(f'At least one hive needs {MINIMUM_ROWS} clean new rows for shared training')
    rows = np.concatenate(training_rows)
    mean, scale = rows.mean(axis=0), rows.std(axis=0)
    scale[scale < 1e-8] = 1.0
    snapshot = hashlib.sha256(json.dumps(ranges, sort_keys=True).encode()).hexdigest()
    return PooledSplits({k: np.concatenate(v) for k, v in windows.items()},
                        {k: np.concatenate(v) for k, v in targets.items()},
                        {k: np.concatenate(v) for k, v in hive_ids.items()}, mean, scale, ranges, snapshot)


class SharedModelService(ModelService):
    """Reuse the raw(N,24,2) prediction contract, with one global champion."""
    def __init__(self, runtime):
        self.runtime = Path(runtime).resolve()
        self.runtime.mkdir(parents=True, exist_ok=True)
        self._promotion_lock, self._training_lock = RLock(), Lock()
        self._cache, self._metadata_cache, self._champion = {}, {}, None
        self._state_path = self.runtime / 'training_state.json'
        self._state = json.loads(self._state_path.read_text()) if self._state_path.exists() else {}
        self.tracking_uri = f"sqlite:///{self.runtime / 'mlflow.db'}"
        self.client = MlflowClient(tracking_uri=self.tracking_uri, registry_uri=self.tracking_uri)
        experiment = self.client.get_experiment_by_name('BeeOPS_Common')
        self.experiment_id = experiment.experiment_id if experiment else self.client.create_experiment(
            'BeeOPS_Common', artifact_location=(self.runtime / 'artifacts').as_uri())
        try:
            registry = self.client.get_registered_model(REGISTRY_NAME)
        except MlflowException as exc:
            if exc.error_code != 'RESOURCE_DOES_NOT_EXIST':
                raise
            registry = self.client.create_registered_model(REGISTRY_NAME)
        if registry.aliases.get(ALIAS):
            self._champion = self._load_bundle(str(registry.aliases[ALIAS]))
            self._cache[self._champion.version] = self._champion
        # Failed attempts reserve evaluated observations, but synthetic demo
        # timestamps must never advance real hives' monitoring cutoffs.
        for run in self.client.search_runs([self.experiment_id], max_results=10000):
            if run.data.params.get('reason') != QUALITY_DEMO_REASON:
                self._reserve(self._run_training_ranges(run), persist=False)

    def _run_training_ranges(self, run):
        """Recover length-limited run params only from that run's full bundle."""
        try:
            return json.loads(run.data.params.get('per_hive_ranges', '{}'))
        except json.JSONDecodeError:
            try:
                path = Path(self.client.download_artifacts(run.info.run_id, 'bundle/metadata.json'))
                metadata = json.loads(path.read_text())
                if metadata.get('run_id') != run.info.run_id or metadata.get('registry_id') != REGISTRY_NAME:
                    raise ValueError('Training metadata belongs to another run or registry')
                ranges = metadata['per_hive_ranges']
                if not isinstance(ranges, dict) or not ranges:
                    raise ValueError('Complete per-hive training ranges are required')
                for hive_id, info in ranges.items():
                    if not isinstance(hive_id, str) or not hive_id or not isinstance(info, dict):
                        raise ValueError('Invalid training range identity')
                    cutoff = pd.Timestamp(info['test_end'])
                    if pd.isna(cutoff) or cutoff.tzinfo is None:
                        raise ValueError('Training range requires a timezone-aware cutoff')
                return ranges
            except (OSError, ValueError, TypeError, KeyError, MlflowException) as exc:
                raise RuntimeError(f'Cannot recover complete training ranges for run {run.info.run_id}') from exc

    def _reserve(self, ranges, persist=True):
        consumed = self._state.setdefault('consumed_through_by_hive', {})
        for hive_id, info in ranges.items():
            cutoff = info['test_end']
            if hive_id not in consumed or pd.Timestamp(cutoff) > pd.Timestamp(consumed[hive_id]):
                consumed[hive_id] = cutoff
        if persist:
            self._save_state()

    def status(self):
        with self._promotion_lock:
            bundle = self._champion
            return {'ready': bundle is not None, 'version': bundle.version if bundle else None,
                'model_scope': 'shared', 'registry_id': REGISTRY_NAME, 'registry_name': REGISTRY_NAME,
                'model_source': 'mlflow_shared' if bundle else None, 'data_source': 'pooled_hives',
                'promotion_policy': FORECAST_POLICY, 'gate_policy_version': FORECAST_POLICY,
                'gate_mae_kg': None, 'minimum_training_rows': MINIMUM_ROWS,
                'consumed_through': None, 'trained_through': None,
                'consumed_through_by_hive': dict(self._state.get('consumed_through_by_hive', {})),
                'per_hive_ranges': bundle.metadata.get('per_hive_ranges', {}) if bundle else {},
                'last_training': self._state.get('last_training'),
                'run_id': bundle.metadata.get('run_id') if bundle else None,
                'model_sha256': bundle.metadata.get('model_sha256') if bundle else None}

    def bootstrap_pool(self, frames):
        return self.ensure_ready(frames)

    def ensure_ready(self, frames):
        if not self._training_lock.acquire(blocking=False):
            raise RuntimeError('A shared model training job is already running')
        try:
            registered = {str(r.version) for r in self.client.search_model_versions(f"name='{REGISTRY_NAME}'")}
            if {'1', '2', '3'}.issubset(registered):
                self._verify_initial_versions()
                return self._ready_result('already_available', [])
            if registered not in (set(), {'1'}, {'1', '2'}):
                raise RuntimeError('Incomplete shared registry requires inspection; existing versions will not be overwritten')
            results, pool = [], prepare_pool(frames, skip_insufficient=True)
            if not registered:
                results.append(self._train_pool(pool, 'bootstrap_pool', None, 12, promote=True))
                registered.add('1')
            first = self._cache.get('1') or self._load_bundle('1')
            if first.metadata['snapshot_id'] != pool.snapshot_id:
                raise ValueError('Resume shared initialization with the same hive training snapshots')
            if '2' not in registered:
                results.append(self._train_pool(pool, COMPARISON_REASON, first, 8, promote=False))
                registered.add('2')
            if '3' not in registered:
                second = self._cache.get('2') or self._load_bundle('2')
                results.append(self._train_pool(self._quality_demo_pool(), QUALITY_DEMO_REASON, second, 8, promote=False))
            self._verify_initial_versions()
            return self._ready_result('created', results)
        finally:
            self._training_lock.release()

    def _ready_result(self, status, results):
        return {'status': status, 'ready': self.status()['ready'], 'version': self.status()['version'],
                'versions': ['1', '2', '3'], 'comparison_ready': True, 'results': results,
                'registry_id': REGISTRY_NAME, 'model_scope': 'shared'}

    def _verify_initial_versions(self):
        for version in ('1', '2', '3'):
            record = self.client.get_model_version(REGISTRY_NAME, version)
            loaded = self._cache.get(version) or self._load_bundle(version)
            if version in ('1', '2') and record.tags.get('gate_passed') != 'true':
                raise RuntimeError(f'Shared version {version} is not deployment-ready')
            if version == '3' and record.tags.get('gate_passed') != 'false':
                raise RuntimeError('Shared quality demo did not produce a naturally rejected candidate')
            self._cache[version] = loaded
        if self._champion is None:
            self.rollback('1')  # Complete an artifact commit interrupted before first alias.

    @staticmethod
    def _quality_demo_pool():
        rng = np.random.default_rng(1729)
        data = pd.DataFrame({'timestamp': pd.date_range('2035-01-01', periods=336, freq='h', tz='UTC'),
            'hive_id': '__shared_quality_demo__', 'weight_kg': 40 + rng.uniform(-3, 3, 336),
            'temperature_c': 20 + rng.uniform(-2, 2, 336), 'event': 'normal'})
        data.attrs['source'] = 'seeded_unpredictable_synthetic_quality_gate_demo'
        return prepare_pool({'__shared_quality_demo__': data})

    def train_candidate_pool(self, frames, reason, snapshot_id):
        if not self._training_lock.acquire(blocking=False):
            raise RuntimeError('A shared model training job is already running')
        try:
            if self._champion is None:
                raise RuntimeError('Shared model service is not ready; bootstrap first')
            pool = self._quality_demo_pool() if reason == QUALITY_DEMO_REASON else prepare_pool(
                frames, self._state.get('consumed_through_by_hive', {}), skip_insufficient=True)
            return self._train_pool(pool, reason, self._champion, 8, promote=True, request_snapshot_id=snapshot_id)
        finally:
            self._training_lock.release()

    def _fit_pool(self, model, mean, scale, pool, epochs):
        train_x, train_y = pool.windows['train'], pool.targets['train']
        x = ((train_x - mean) / scale).astype(np.float32)
        y = (train_y - train_x[:, -1, 0]).astype(np.float32)
        temporary = Bundle('candidate', model, mean, scale, {})
        best, weights, selected = float('inf'), None, 0
        order = np.random.default_rng(42).permutation(len(x))
        for epoch in range(1, epochs + 1):
            for start in range(0, len(order), 32):
                indices = order[start:start + 32]
                model.train_on_batch(x[indices], y[indices])
            mae = float(np.abs(self._predict_bundle(temporary, pool.windows['validation']) - pool.targets['validation']).mean())
            if np.isfinite(mae) and mae < best:
                best, weights, selected = mae, model.get_weights(), epoch
        if weights is None:
            raise RuntimeError('Training produced no finite trained checkpoint')
        model.set_weights(weights)
        return selected

    def _evaluate(self, bundle, pool):
        metrics, per_hive = {}, {hive: {} for hive in pool.per_hive_ranges}
        for name in ('validation', 'test'):
            x, y = pool.windows[name], pool.targets[name]
            predictions = self._predict_bundle(bundle, x)
            if not np.isfinite(predictions).all():
                raise RuntimeError('Model evaluation produced nonfinite predictions')
            scores = baseline_scores(x, y)
            metrics.update({f'{name}_mae': float(np.abs(predictions - y).mean()),
                f'{name}_samples': len(y), f'{name}_persistence_mae': scores['persistence'],
                f'{name}_seasonal_24h_mae': scores['seasonal_24h']})
            for hive_id, values in per_hive.items():
                mask = pool.hive_ids[name] == hive_id
                baselines = baseline_scores(x[mask], y[mask])
                values.update({f'{name}_mae': float(np.abs(predictions[mask] - y[mask]).mean()),
                    f'{name}_samples': int(mask.sum()), f'{name}_persistence_mae': baselines['persistence'],
                    f'{name}_seasonal_24h_mae': baselines['seasonal_24h']})
            metrics[f'{name}_macro_hive_mae'] = float(np.mean([v[f'{name}_mae'] for v in per_hive.values()]))
        metrics.update(mae=metrics['validation_mae'], baseline_mae=metrics['validation_persistence_mae'],
                       test_baseline_mae=metrics['test_persistence_mae'], train_samples=len(pool.targets['train']))
        return metrics, per_hive

    def _train_pool(self, pool, reason, parent, epochs, promote, request_snapshot_id=None):
        quality_demo, comparison = reason == QUALITY_DEMO_REASON, reason == COMPARISON_REASON
        policy = QUALITY_DEMO_POLICY if quality_demo else FORECAST_POLICY
        expected_champion = self._champion
        mean, scale = (parent.mean.copy(), parent.scale.copy()) if parent else (pool.mean, pool.scale)
        if parent:
            model = tf.keras.models.clone_model(parent.model)
            model.set_weights(parent.model.get_weights())
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.001), loss='mse')
        else:
            model = self._new_model()
        metadata = {'registry_id': REGISTRY_NAME, 'model_scope': 'shared', 'snapshot_id': pool.snapshot_id,
            'request_snapshot_id': str(request_snapshot_id or pool.snapshot_id), 'reason': reason,
            'parent_version': parent.version if parent else 'none', 'epochs_completed': epochs,
            'architecture': 'LSTM(8) -> Dense(1), residual kg + last observed kg',
            'data_source': 'synthetic_quality_gate_demo' if quality_demo else 'pooled_hives',
            'promotion_policy': policy, 'gate_policy_version': policy,
            'performance_metrics_role': 'quality_gate_demo' if quality_demo else 'descriptive_only',
            'reused_observations_for_demo': comparison,
            'evaluation_mode': 'reused_training_observations_for_version_comparison' if comparison else 'per_hive_chronological_disjoint_partitions',
            'checkpoint_policy': 'minimum pooled validation MAE among trained epochs 1..N; epoch zero excluded',
            'test_policy': 'fixed per-hive held-out test; never used for selection; v2 comparison reuses the same test as v1',
            'aggregation': 'sample-weighted pooled MAE, plus equal-hive macro MAE and per-hive scores',
            # Warm-started weights retain information learned from parent
            # hives, even when only another hive provides fresh observations.
            'per_hive_ranges': {**(parent.metadata.get('per_hive_ranges', {}) if parent else {}),
                                **pool.per_hive_ranges},
            'evaluated_hive_ids': sorted(pool.per_hive_ranges),
            'scaler_fit_ranges': parent.metadata['scaler_fit_ranges'] if parent else {
                hive: {'start': r['train_start'], 'end': r['train_end'], 'rows': r['train_rows']}
                for hive, r in pool.per_hive_ranges.items()},
            'scaler_source_version': parent.metadata.get('scaler_source_version', parent.version) if parent else '1'}
        run = self.client.create_run(self.experiment_id, tags={'mlflow.runName': f'common-{reason}',
            'promotion_policy': policy, 'gate_policy_version': policy, 'model_scope': 'shared'})
        run_id = run.info.run_id
        try:
            for key, value in metadata.items():
                self.client.log_param(run_id, key, json.dumps(value, sort_keys=True) if isinstance(value, dict) else value)
            if not quality_demo:
                self._reserve(pool.per_hive_ranges)
            selected = self._fit_pool(model, mean, scale, pool, epochs)
            candidate = Bundle('candidate', model, mean, scale, metadata)
            metrics, per_hive = self._evaluate(candidate, pool)
            metrics['selected_epoch'] = selected
            incumbent_mae = float(np.abs(self._predict_bundle(parent, pool.windows['validation']) - pool.targets['validation']).mean()) if parent else None
            if incumbent_mae is not None:
                metrics['incumbent_mae'] = incumbent_mae
            if not np.isfinite(list(metrics.values())).all():
                raise RuntimeError('Model evaluation produced nonfinite scores')
            for key, value in metrics.items():
                self.client.log_metric(run_id, key, float(value))
            metadata.update(run_id=run_id, selected_epoch=selected, per_hive_metrics=per_hive)
            reasons = evaluate_gate(metrics['validation_mae'], incumbent_mae,
                metrics['validation_persistence_mae'], metrics['validation_seasonal_24h_mae']) if quality_demo else []
            with tempfile.TemporaryDirectory(prefix='shared-model-', dir=self.runtime) as directory:
                path = Path(directory) / 'bundle'; path.mkdir()
                example = np.zeros((1, WINDOW, 2), dtype=np.float32)
                mlflow_keras.save_model(model, str(path / 'model'),
                    signature=infer_signature(example, np.zeros((1, 1), dtype=np.float32)),
                    pip_requirements=[f'tensorflow=={tf.__version__}'],
                    metadata={'output_semantics': 'delta_kg; wrapper adds last observed weight'})
                files = sorted((path / 'model').rglob('*.keras'))
                if len(files) != 1:
                    raise RuntimeError('Expected exactly one serialized Keras model artifact')
                metadata['model_sha256'] = hashlib.sha256(files[0].read_bytes()).hexdigest()
                metadata['weights_sha256'] = hashlib.sha256(b''.join(w.tobytes() for w in model.get_weights())).hexdigest()
                (path / 'scaler.json').write_text(json.dumps({'mean': mean.tolist(), 'scale': scale.tolist(),
                    'fit_ranges': metadata['scaler_fit_ranges']}, indent=2))
                (path / 'schema.json').write_text(json.dumps({'features': FEATURES, 'window': WINDOW,
                    'input_units': ['kg', 'celsius'], 'model_output': 'delta_kg',
                    'served_output': 'last_weight_kg + delta_kg', 'schema_version': 1, 'model_scope': 'shared'}, indent=2))
                (path / 'metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False))
                self.client.log_artifacts(run_id, str(path), artifact_path='bundle')
            record = self.client.create_model_version(REGISTRY_NAME, source=f'{run.info.artifact_uri}/bundle',
                run_id=run_id, tags={'promotion_policy': policy, 'gate_policy_version': policy,
                    'model_scope': 'shared', 'gate_passed': 'false'})
            version = str(record.version)
            # Rejected demo bundles must also be real/loadable. All fallible
            # artifact and primary reporting work precedes the alias switch.
            loaded = self._load_bundle(version)
            np.testing.assert_allclose(self._predict_bundle(loaded, pool.windows['validation'][:2]),
                self._predict_bundle(candidate, pool.windows['validation'][:2]), atol=1e-5, rtol=1e-5)
            result = {'version': version, 'run_id': run_id, 'parent_version': parent.version if parent else None,
                'promoted': False, 'gate_passed': not reasons, 'deployable': not reasons,
                'gate_reasons': reasons, 'mae': metrics['validation_mae'], 'test_mae': metrics['test_mae'],
                'baseline_mae': metrics['baseline_mae'], 'selected_epoch': selected,
                'promotion_policy': policy, 'gate_policy_version': policy,
                'per_hive_ranges': metadata['per_hive_ranges'], 'per_hive_metrics': per_hive,
                'snapshot_id': pool.snapshot_id, 'model_sha256': metadata['model_sha256'],
                'reused_observations_for_demo': comparison, 'model_scope': 'shared', 'registry_id': REGISTRY_NAME}
            self.client.set_model_version_tag(REGISTRY_NAME, version, 'gate_passed', str(not reasons).lower())
            self.client.set_tag(run_id, 'gate_reasons', json.dumps(reasons))
            self.client.set_tag(run_id, 'promoted', 'false')
            self.client.set_terminated(run_id, 'FINISHED')
            with self._promotion_lock:
                self._state['last_training'] = result
                self._save_state()
                self._cache[version] = loaded
                if promote and not reasons:
                    if self._champion is not expected_champion:
                        raise RuntimeError('Champion changed during shared training; candidate remains registered')
                    self._set_alias_safely(version, expected_champion.version if expected_champion else None)
                    self._champion = loaded
                    result['promoted'] = True
                    try:
                        self._save_state()
                        self.client.set_tag(run_id, 'promoted', 'true')
                    except Exception:
                        LOGGER.exception('Shared alias committed; secondary reporting deferred')
            return result
        except Exception:
            try:
                self.client.set_terminated(run_id, 'FAILED')
            except Exception:
                LOGGER.exception('Could not mark failed shared training run')
            raise

    def _load_bundle(self, version):
        record = self.client.get_model_version(REGISTRY_NAME, str(version))
        path = Path(self.client.download_artifacts(record.run_id, 'bundle'))
        schema = json.loads((path / 'schema.json').read_text())
        if schema.get('features') != FEATURES or schema.get('window') != WINDOW or schema.get('model_output') != 'delta_kg' or schema.get('model_scope') != 'shared':
            raise ValueError('Incompatible shared model bundle schema')
        scaler = json.loads((path / 'scaler.json').read_text())
        mean, scale = np.asarray(scaler['mean']), np.asarray(scaler['scale'])
        if mean.shape != (2,) or scale.shape != (2,) or not np.isfinite([mean, scale]).all() or (scale <= 0).any():
            raise ValueError('Invalid shared model bundle scaler')
        metadata = json.loads((path / 'metadata.json').read_text())
        files = sorted((path / 'model').rglob('*.keras'))
        if len(files) != 1 or hashlib.sha256(files[0].read_bytes()).hexdigest() != metadata.get('model_sha256'):
            raise ValueError('Shared model artifact checksum mismatch')
        model = mlflow_keras.load_model(str(path / 'model'), load_model_kwargs={'compile': False})
        mean.setflags(write=False); scale.setflags(write=False)
        bundle = Bundle(str(version), model, mean, scale, metadata)
        smoke = self._predict_bundle(bundle, np.tile(mean, (1, WINDOW, 1)).astype(np.float32))
        if smoke.shape != (1,) or not np.isfinite(smoke).all():
            raise ValueError('Shared bundle failed inference smoke test')
        self._metadata_cache[str(version)] = metadata
        return bundle

    def version_metadata(self, version):
        version = str(version)
        record = self.client.get_model_version(REGISTRY_NAME, version)
        metadata = self._metadata_cache.get(version)
        if metadata is None:
            path = Path(self.client.download_artifacts(record.run_id, 'bundle/metadata.json'))
            metadata = json.loads(path.read_text()); self._metadata_cache[version] = metadata
        return {**metadata, 'version': version, 'run_id': record.run_id, 'monitoring_after': None,
                'deployable': record.tags.get('gate_passed') == 'true'}

    def versions(self):
        current, result = self.status()['version'], []
        for record in self.client.search_model_versions(f"name='{REGISTRY_NAME}'"):
            run = self.client.get_run(record.run_id); metadata = self.version_metadata(record.version)
            passed = record.tags.get('gate_passed') == 'true'
            result.append({'version': str(record.version), 'run_id': record.run_id,
                'champion': str(record.version) == current, 'created_at': record.creation_timestamp,
                'metrics': dict(run.data.metrics), 'params': dict(run.data.params),
                'gate_passed': passed, 'deployable': passed, 'inference_available': True,
                'gate_reasons': json.loads(run.data.tags.get('gate_reasons', '[]')),
                'promotion_policy': metadata['promotion_policy'], 'run_status': run.info.status,
                'parent_version': metadata['parent_version'], 'data_source': metadata['data_source'],
                'source': record.source, 'model_sha256': metadata['model_sha256'],
                'weights_sha256': metadata['weights_sha256'], 'per_hive_ranges': metadata['per_hive_ranges'],
                'per_hive_metrics': metadata['per_hive_metrics'], 'registry_id': REGISTRY_NAME, 'model_scope': 'shared'})
        return sorted(result, key=lambda row: int(row['version']), reverse=True)

    def _set_alias_safely(self, version, previous):
        try:
            self.client.set_registered_model_alias(REGISTRY_NAME, ALIAS, version)
        except Exception:
            registry = self.client.get_registered_model(REGISTRY_NAME)
            current = registry.aliases.get(ALIAS)
            if current is not None and str(current) != previous:
                if previous is None:
                    self.client.delete_registered_model_alias(REGISTRY_NAME, ALIAS)
                else:
                    self.client.set_registered_model_alias(REGISTRY_NAME, ALIAS, previous)
            raise

    def rollback(self, version):
        version = str(version)
        record = self.client.get_model_version(REGISTRY_NAME, version)
        if record.tags.get('gate_passed') != 'true':
            raise ValueError('Version did not pass the quality gate')
        loaded = self._cache.get(version) or self._load_bundle(version)
        with self._promotion_lock:
            previous = self._champion.version if self._champion else None
            self._save_state()
            self._set_alias_safely(version, previous)
            self._champion = loaded; self._cache[version] = loaded
        return {'version': version, 'previous_version': previous, 'rolled_back': True,
                'model_scope': 'shared', 'registry_id': REGISTRY_NAME}
