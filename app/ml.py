"""Local, single-worker LSTM lifecycle with immutable serving bundles.

Validation selects a trained checkpoint. Ordinary forecast demonstrations
deploy finite, loadable models; only the explicit quality-gate demonstration
uses performance to block promotion. A disjoint final test is reported once.
The model predicts a next-hour delta in kg, added to the last observed weight.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
import tempfile
from threading import Lock, RLock
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from mlflow import keras as mlflow_keras
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
import tensorflow as tf
from .store import MINIMUM_BASELINE_GAIN, BASELINE_ZERO_TOLERANCE_KG, atomic_json

LOGGER = logging.getLogger(__name__)
FEATURES = ["weight_kg", "temperature_c"]
WINDOW = 24
MINIMUM_ROWS = 168
GATE_MAE_KG = 0.15  # Provisional educational threshold, not a health diagnosis.
REGISTRY_NAME = "BeeOPS_Weight"
ALIAS = "champion"
FORECAST_POLICY = "forecast_demo_v1"
QUALITY_DEMO_POLICY = "quality_gate_demo_v1"
QUALITY_DEMO_REASON = "synthetic_unpredictable_gate_test"
COMPARISON_REASON = "version_comparison_demo"

try:
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except RuntimeError:
    pass  # Another component may already have initialized the TF runtime.


@dataclass(frozen=True)
class Splits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True)
class Bundle:
    version: str
    model: Any
    mean: np.ndarray
    scale: np.ndarray
    metadata: dict


def prepare_splits(frame: pd.DataFrame, after: str | None = None) -> Splits:
    """Use only the newest uninterrupted normal run, after the prior snapshot.

Each held-out block has its own 24 context rows plus at least 24 targets.
No raw row is shared between any partitions, including their input contexts.
"""
    required = {"timestamp", "hive_id", "event", *FEATURES}
    if not required.issubset(frame.columns):
        raise ValueError(f"Training frame requires columns: {sorted(required)}")
    data = frame.copy(deep=True)
    data["timestamp"] = pd.to_datetime(data.timestamp, utc=True, errors="raise")
    if data.empty or data.timestamp.isna().any():
        raise ValueError(f"At least {MINIMUM_ROWS} clean new hourly rows are required")
    if data.hive_id.nunique(dropna=False) != 1 or data.hive_id.isna().any():
        raise ValueError("Training requires exactly one hive_id")
    if data.timestamp.duplicated().any() or not data.timestamp.is_monotonic_increasing:
        raise ValueError("Training timestamps must be unique and ordered")
    if not data.timestamp.eq(data.timestamp.dt.floor("h")).all():
        raise ValueError("Training timestamps must fall on exact hours")
    values = data[FEATURES].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Training values must be finite")
    if ((values[:, 0] <= 0) | (values[:, 0] > 300)).any():
        raise ValueError("weight_kg must be in (0, 300]")
    if ((values[:, 1] < -50) | (values[:, 1] > 60)).any():
        raise ValueError("temperature_c must be in [-50, 60]")
    if after is not None:
        data = data[data.timestamp > pd.Timestamp(after)].copy()
    # Mark operations/faults as breaks; never interpolate across them or a gap.
    normal = data[data.event.eq("normal")].copy()
    if not normal.empty:
        blocks = normal.timestamp.diff().ne(pd.Timedelta(hours=1)).cumsum()
        normal = normal[blocks == blocks.iloc[-1]].reset_index(drop=True)
    if len(normal) < MINIMUM_ROWS:
        raise ValueError(f"At least {MINIMUM_ROWS} contiguous clean new hourly rows are required; got {len(normal)}")
    heldout = max(48, len(normal) // 5)
    split = len(normal) - 2 * heldout
    return Splits(normal.iloc[:split].copy(), normal.iloc[split:-heldout].copy(), normal.iloc[-heldout:].copy())


def make_windows(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    values = frame[FEATURES].to_numpy(dtype=np.float32)
    windows = np.stack([values[i - WINDOW:i] for i in range(WINDOW, len(values))])
    return windows, values[WINDOW:, 0].copy()


def baseline_scores(windows: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    """Both baselines predict the same targets from already available inputs."""
    return {"persistence": float(np.abs(windows[:, -1, 0] - targets).mean()),
            "seasonal_24h": float(np.abs(windows[:, 0, 0] - targets).mean())}


def baseline_skill(mae: float, baseline_mae: float) -> float | None:
    return 1-mae/baseline_mae if baseline_mae > BASELINE_ZERO_TOLERANCE_KG else None


def evaluate_gate(mae: float, incumbent_mae: float | None, baseline_mae: float,
                  seasonal_baseline_mae: float | None = None) -> list[str]:
    """Explicit quality-gate demos need >=5% gain over the strongest baseline.

    This is a provisional educational policy. Equality with persistence, an
    untrained zero-residual checkpoint, or a perfect seasonal baseline does not
    establish added value. Existing registry versions are never re-tagged here.
    """
    values = [mae, baseline_mae] + ([] if incumbent_mae is None else [incumbent_mae])
    if seasonal_baseline_mae is not None:
        values.append(seasonal_baseline_mae)
    if not np.isfinite(values).all() or min(values) < 0:
        return ["nonfinite_metric"]
    reasons = []
    if mae > GATE_MAE_KG:
        reasons.append("absolute_mae")
    if incumbent_mae is not None and mae > incumbent_mae + 1e-7:
        reasons.append("incumbent_regression")
    if mae > baseline_mae + 1e-7:
        reasons.append("persistence_regression")
    if seasonal_baseline_mae is not None and mae > seasonal_baseline_mae + 1e-7:
        reasons.append("seasonal_24h_regression")
    strongest = min(baseline_mae, seasonal_baseline_mae) if seasonal_baseline_mae is not None else baseline_mae
    gain = baseline_skill(mae, strongest)
    if gain is None or gain+1e-12 < MINIMUM_BASELINE_GAIN:
        reasons.append("baseline_improvement_below_minimum")
    return reasons


class ModelService:
    def __init__(self, runtime: Path):
        self.runtime = Path(runtime).resolve()
        self.runtime.mkdir(parents=True, exist_ok=True)
        self._promotion_lock = RLock()
        self._training_lock = Lock()
        self._comparison_lock = Lock()
        self._champion: Bundle | None = None
        self._cache: dict[str, Bundle] = {}
        self._state_path = self.runtime / "training_state.json"
        self._state = json.loads(self._state_path.read_text()) if self._state_path.exists() else {}
        self.tracking_uri = f"sqlite:///{self.runtime / 'mlflow.db'}"
        self.client = MlflowClient(tracking_uri=self.tracking_uri, registry_uri=self.tracking_uri)
        experiment = self.client.get_experiment_by_name("BeeOPS")
        self.experiment_id = experiment.experiment_id if experiment else self.client.create_experiment(
            "BeeOPS", artifact_location=(self.runtime / "artifacts").as_uri())
        try:
            registry = self.client.get_registered_model(REGISTRY_NAME)
        except MlflowException as exc:
            if exc.error_code != "RESOURCE_DOES_NOT_EXIST":
                raise
            registry = self.client.create_registered_model(REGISTRY_NAME)
        version = registry.aliases.get(ALIAS)
        if version is not None:
            self._champion = self._load_bundle(str(version))
            self._cache[str(version)] = self._champion
        # MLflow run parameters are a durable second record of consumed ranges,
        # including failed attempts and a process crash before sidecar commit.
        cutoffs = [self._state["consumed_through"]] if self._state.get("consumed_through") else []
        for run in self.client.search_runs([self.experiment_id], max_results=10000):
            if run.data.params.get("test_end"):
                cutoffs.append(run.data.params["test_end"])
            if run.data.params.get("reason") == "bootstrap" and run.data.params.get("reused_observations_for_demo", "").lower() == "true":
                self._state["demo_reinitialization_attempted"] = True
        if cutoffs:
            self._state["consumed_through"] = max(cutoffs, key=pd.Timestamp)
        if self._state.pop("promotion_pending", None):
            result = self._state.get("last_training", {})
            result["promoted"] = self._champion is not None and self._champion.version == result.get("version")
            if not result["promoted"] and not result.get("gate_reasons"):
                result["gate_reasons"] = ["promotion_interrupted"]
            self._save_state()

    def _can_reinitialize_for_demo(self) -> bool:
        """One new-policy attempt for an old, performance-rejected bootstrap.

        This does not relax a historical version's tags or reuse observations
        after a current-policy/artifact failure or an explicit gate demo.
        """
        if self._champion is not None or self._state.get("demo_reinitialization_attempted"):
            return False
        previous = self._state.get("last_training", {})
        performance_reasons = {"absolute_mae", "incumbent_regression", "persistence_regression",
                               "seasonal_24h_regression", "baseline_improvement_below_minimum"}
        reasons = previous.get("gate_reasons") or []
        if previous.get("promoted") or not reasons or not set(reasons).issubset(performance_reasons):
            return False
        if previous.get("parent_version") not in (None, "none") or not previous.get("version") or not previous.get("run_id"):
            return False
        try:
            if pd.Timestamp(self._state.get("consumed_through")) != pd.Timestamp(previous.get("test_end")):
                return False
            record = self.client.get_model_version(REGISTRY_NAME, str(previous["version"]))
            if record.run_id != previous["run_id"] or record.tags.get("gate_passed") != "false":
                return False
            params = self.client.get_run(record.run_id).data.params
            policies = [previous.get("promotion_policy"), previous.get("gate_policy_version"),
                        params.get("promotion_policy"), params.get("gate_policy_version")]
            return (params.get("reason") == "bootstrap" and params.get("parent_version", "none") == "none"
                    and pd.Timestamp(params.get("test_end")) == pd.Timestamp(previous["test_end"])
                    and not any(policy in (FORECAST_POLICY, QUALITY_DEMO_POLICY) for policy in policies))
        except (KeyError, ValueError, TypeError, MlflowException):
            return False

    def status(self) -> dict:
        with self._promotion_lock:
            bundle = self._champion
            policy = bundle.metadata.get("promotion_policy", bundle.metadata.get("gate_policy_version", "legacy_quality_gate")) if bundle else FORECAST_POLICY
            return {
                "version": bundle.version if bundle else None,
                "ready": bundle is not None,
                "model_source": "mlflow" if bundle else None,
                "data_source": bundle.metadata.get("data_source", "unknown") if bundle else None,
                "registry_name": REGISTRY_NAME,
                "promotion_policy": policy,
                "gate_policy_version": policy,
                "gate_mae_kg": GATE_MAE_KG if policy != FORECAST_POLICY else None,
                "can_reinitialize_for_demo": self._can_reinitialize_for_demo(),
                "minimum_training_rows": MINIMUM_ROWS,
                "trained_through": bundle.metadata.get("train_end") if bundle else None,
                "consumed_through": self._state.get("consumed_through"),
                "last_training": self._state.get("last_training"),
            }

    def predict(self, windows: np.ndarray, version: str | None = None) -> tuple[np.ndarray, str]:
        windows = np.asarray(windows, dtype=np.float32)
        if windows.ndim != 3 or windows.shape[1:] != (WINDOW, 2) or not len(windows):
            raise ValueError("Expected a nonempty array with shape (N, 24, 2)")
        if not np.isfinite(windows).all():
            raise ValueError("Prediction inputs must be finite")
        # Capture one immutable bundle; never consult a mutable alias mid-request.
        with self._promotion_lock:
            if version is None:
                bundle = self._champion
            else:
                version = str(version)
                bundle = self._cache.get(version)
                if bundle is None:
                    bundle = self._load_bundle(version)
                    self._cache[version] = bundle
        if bundle is None:
            raise RuntimeError("Model service is not ready")
        predictions = self._predict_bundle(bundle, windows)
        if not np.isfinite(predictions).all():
            raise RuntimeError("Model produced nonfinite predictions")
        if ((predictions <= 0) | (predictions > 300)).any():
            raise RuntimeError("Model produced predictions outside physical weight bounds (0, 300]")
        return predictions, bundle.version

    @staticmethod
    def _predict_bundle(bundle: Bundle, windows: np.ndarray) -> np.ndarray:
        scaled = ((windows - bundle.mean) / bundle.scale).astype(np.float32)
        delta = np.asarray(bundle.model(scaled, training=False)).reshape(-1)
        return windows[:, -1, 0].astype(np.float64) + delta.astype(np.float64)

    def bootstrap(self, frame: pd.DataFrame) -> dict:
        with self._promotion_lock:
            if self._champion is not None:
                return {"promoted": False, "version": self._champion.version,
                        "gate_reasons": ["existing_champion"], "ready": True,
                        "promotion_policy": self.status()["promotion_policy"],
                        "gate_policy_version": self.status()["gate_policy_version"]}
        digest = hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).values.tobytes()).hexdigest()
        return self._train(frame, "bootstrap", digest, bootstrap=True)

    def train_candidate(self, frame: pd.DataFrame, reason: str, snapshot_id: str) -> dict:
        if not self.status()["ready"]:
            raise RuntimeError("Model service is not ready; bootstrap first")
        return self._train(frame, reason, snapshot_id, bootstrap=False)

    def ensure_comparison_version(self, frame: pd.DataFrame) -> dict:
        """Prepare exact v1/v2 inference without changing the current champion.

        A new v2 is a real eight-epoch warm start on the supplied educational
        data. Its reused-data evaluation is explicitly descriptive. Existing
        versions, including historically blocked but loadable ones, are never
        rewritten or made eligible for deployment by this method.
        """
        with self._comparison_lock:
            registered = {str(item.version) for item in self.client.search_model_versions(f"name='{REGISTRY_NAME}'")}
            if {"1", "2"}.issubset(registered):
                loaded = {}
                for version in ("1", "2"):
                    try:
                        loaded[version] = self._cache.get(version) or self._load_bundle(version)
                    except Exception as exc:
                        raise RuntimeError(f"Comparison version {version} could not be loaded: {exc}") from exc
                with self._promotion_lock:
                    self._cache.update(loaded)
                return {"status": "already_available", "versions": ["1", "2"],
                        "version": "2", "comparison_version": "2", "comparison_ready": True,
                        "promoted": False, "deployment_action": "comparison_only",
                        "promotion_policy": loaded["2"].metadata.get("promotion_policy", loaded["2"].metadata.get("gate_policy_version", "legacy_quality_gate"))}
            if registered != {"1"} or self.status()["version"] != "1":
                raise RuntimeError("Bootstrap a ready version 1 before preparing comparison version 2")
            snapshot_id = hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).values.tobytes()).hexdigest()
            result = self.train_candidate(frame, COMPARISON_REASON, snapshot_id)
            if not result.get("comparison_ready"):
                raise RuntimeError(f"Comparison version {result['version']} is not ready: {result['gate_reasons']}")
            return {**result, "status": "created", "versions": ["1", "2"], "comparison_version": "2"}

    @staticmethod
    def _new_model() -> Any:
        tf.keras.utils.set_random_seed(42)
        inputs = tf.keras.Input(shape=(WINDOW, 2), name="normalized_weight_temperature")
        hidden = tf.keras.layers.LSTM(8, name="hourly_lstm")(inputs)
        delta = tf.keras.layers.Dense(1, kernel_initializer="zeros", bias_initializer="zeros", name="next_hour_delta_kg")(hidden)
        model = tf.keras.Model(inputs, delta)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.003), loss="mse")
        return model

    def _fit(self, model, mean, scale, parts, epochs) -> tuple[int, float]:
        train_x, train_y = make_windows(parts.train)
        val_x, val_y = make_windows(parts.validation)
        x = ((train_x - mean) / scale).astype(np.float32)
        y = (train_y - train_x[:, -1, 0]).astype(np.float32)
        temporary = Bundle("candidate", model, mean, scale, {})
        # Epoch zero is initialization, not a trained LSTM checkpoint. A
        # constant series can still legitimately learn a zero next-hour delta.
        best_mae, best_weights, selected_epoch = float("inf"), None, 0
        # train_on_batch avoids an unbounded tf.data threadpool in small containers.
        for epoch in range(1, epochs + 1):
            for start in range(0, len(x), 32):
                model.train_on_batch(x[start:start + 32], y[start:start + 32])
            mae = float(np.mean(np.abs(self._predict_bundle(temporary, val_x) - val_y)))
            if np.isfinite(mae) and mae < best_mae:
                best_mae, best_weights, selected_epoch = mae, model.get_weights(), epoch
        if best_weights is None:
            raise RuntimeError("Training produced no finite trained checkpoint")
        model.set_weights(best_weights)
        return selected_epoch, best_mae

    def _train(self, frame, reason, snapshot_id, bootstrap) -> dict:
        if not self._training_lock.acquire(blocking=False):
            raise RuntimeError("A model training job is already running")
        run_id = None
        try:
            comparison_demo = reason == COMPARISON_REASON
            with self._promotion_lock:
                parent = self._champion
                cutoff = self._state.get("consumed_through")
                reused_for_demo = bootstrap and self._can_reinitialize_for_demo()
                if reused_for_demo:
                    cutoff = None
                if comparison_demo:
                    registered = {str(item.version) for item in self.client.search_model_versions(f"name='{REGISTRY_NAME}'")}
                    if registered != {"1"} or parent is None or parent.version != "1":
                        raise ValueError("Reused comparison training is allowed only to create version 2 from the sole ready version 1")
                    cutoff = None
            parts = prepare_splits(frame, after=cutoff)
            if parent is None:
                mean = parts.train[FEATURES].to_numpy(dtype=np.float64).mean(axis=0)
                scale = parts.train[FEATURES].to_numpy(dtype=np.float64).std(axis=0)
                scale[scale < 1e-8] = 1.0
                model = self._new_model()
            else:
                # Warm-start keeps the parent's fitted scaler; refitting it would
                # change the meaning of its learned weights before optimization.
                mean, scale = parent.mean.copy(), parent.scale.copy()
                model = tf.keras.models.clone_model(parent.model)
                model.set_weights(parent.model.get_weights())
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")
            epochs = 12 if bootstrap else 8
            quality_demo = reason == QUALITY_DEMO_REASON
            policy = QUALITY_DEMO_POLICY if quality_demo else FORECAST_POLICY
            metadata = {
                "snapshot_id": str(snapshot_id), "reason": str(reason),
                "parent_version": parent.version if parent else "none",
                "data_source": str(frame.attrs.get("source", "uploaded_or_generated")),
                "architecture": "LSTM(8) -> Dense(1), residual kg + last observed kg",
                "promotion_policy": policy,
                "gate_policy_version": policy,
                "reused_observations_for_demo": reused_for_demo or comparison_demo,
                "evaluation_mode": "reused_training_observations_for_version_comparison" if comparison_demo else "chronological_disjoint_partitions",
                "gate_policy": ("validation <= 0.15kg and <= incumbent and >=5% improvement over min(persistence,seasonal_24h)"
                    if quality_demo else "trained LSTM checkpoint, finite evaluation and artifact load/smoke checks; accuracy is descriptive, not a promotion barrier"),
                "performance_metrics_role": "quality_gate_demo" if quality_demo else "descriptive_only",
                "checkpoint_policy": "minimum validation MAE among trained epochs 1..N; epoch zero excluded",
                "test_policy": ("chronological partitions within reused demonstration data; parent has already seen this dataset; not independent generalization evidence"
                    if comparison_demo else "disjoint final report only; never used for checkpoint selection or gate"),
                "scaler_fit_start": parent.metadata["scaler_fit_start"] if parent else parts.train.timestamp.iloc[0].isoformat(),
                "scaler_fit_end": parent.metadata["scaler_fit_end"] if parent else parts.train.timestamp.iloc[-1].isoformat(),
                "scaler_source_version": parent.metadata.get("scaler_source_version", parent.version) if parent else "self",
                "epochs_completed": epochs,
            }
            if quality_demo:
                metadata.update(gate_minimum_relative_gain_pct=MINIMUM_BASELINE_GAIN*100,
                    gate_baselines="persistence,seasonal_24h; identical validation targets",
                    gate_threshold_status="provisional_educational_not_field_validated")
            for name in ("train", "validation", "test"):
                block = getattr(parts, name)
                metadata[f"{name}_start"] = block.timestamp.iloc[0].isoformat()
                metadata[f"{name}_end"] = block.timestamp.iloc[-1].isoformat()
                metadata[f"{name}_target_start"] = block.timestamp.iloc[WINDOW].isoformat()
            run = self.client.create_run(self.experiment_id, tags={"mlflow.runName": f"beeops-{reason}",
                "snapshot_id": str(snapshot_id), "promotion_policy": policy, "gate_policy_version": policy})
            run_id = run.info.run_id
            for key, value in metadata.items():
                self.client.log_param(run_id, key, value)
            with self._promotion_lock:
                # Reserve the entire snapshot before fitting or evaluating it.
                # Artifact/bookkeeping failures must not enable its reuse in the
                # same process; logged run parameters recover this after a crash.
                self._state["consumed_through"] = max(
                    filter(None, [self._state.get("consumed_through"), metadata["test_end"]]), key=pd.Timestamp)
                if reused_for_demo:
                    self._state["demo_reinitialization_attempted"] = True
                self._save_state()
            selected_epoch, mae = self._fit(model, mean, scale, parts, epochs)
            candidate = Bundle("candidate", model, mean, scale, metadata)
            val_x, val_y = make_windows(parts.validation)
            test_x, test_y = make_windows(parts.test)
            validation_baselines = baseline_scores(val_x, val_y)
            test_baselines = baseline_scores(test_x, test_y)
            baseline_mae = validation_baselines["persistence"]
            best_baseline_name = min(validation_baselines, key=validation_baselines.get)
            best_baseline_mae = validation_baselines[best_baseline_name]
            incumbent_mae = float(np.abs(self._predict_bundle(parent, val_x) - val_y).mean()) if parent else None
            test_mae = float(np.abs(self._predict_bundle(candidate, test_x) - test_y).mean())
            finite_scores = [mae, test_mae, *validation_baselines.values(), *test_baselines.values()]
            if incumbent_mae is not None:
                finite_scores.append(incumbent_mae)
            if not np.isfinite(finite_scores).all():
                raise RuntimeError("Model evaluation produced nonfinite scores")
            metrics = {
                "mae": mae, "validation_mae": mae, "baseline_mae": baseline_mae,
                "validation_persistence_mae": baseline_mae,
                "validation_seasonal_24h_mae": validation_baselines["seasonal_24h"],
                "validation_best_baseline_mae": best_baseline_mae,
                "validation_mae_skill_vs_best_baseline": baseline_skill(mae, best_baseline_mae),
                "test_mae": test_mae, "test_baseline_mae": test_baselines["persistence"],
                "test_persistence_mae": test_baselines["persistence"],
                "test_seasonal_24h_mae": test_baselines["seasonal_24h"],
                "test_best_baseline_mae": min(test_baselines.values()),
                "test_mae_skill_vs_best_baseline": baseline_skill(test_mae, min(test_baselines.values())),
                "validation_samples": len(val_y), "test_samples": len(test_y),
                "train_samples": len(parts.train) - WINDOW, "selected_epoch": selected_epoch,
            }
            if parent:
                metrics["incumbent_mae"] = incumbent_mae
                metrics["test_incumbent_mae"] = float(np.abs(self._predict_bundle(parent, test_x) - test_y).mean())
            for key, value in metrics.items():
                if value is not None and np.isfinite(value):
                    self.client.log_metric(run_id, key, float(value))
            metadata["run_id"] = run_id
            metadata["selected_epoch"] = selected_epoch
            metadata["validation_best_baseline_name"] = best_baseline_name
            self.client.log_param(run_id, "validation_best_baseline_name", best_baseline_name)
            with tempfile.TemporaryDirectory(prefix="beeops-model-", dir=self.runtime) as directory:
                path = Path(directory) / "bundle"
                path.mkdir()
                example = np.zeros((1, WINDOW, 2), dtype=np.float32)
                mlflow_keras.save_model(model, str(path / "model"),
                    signature=infer_signature(example, np.zeros((1, 1), dtype=np.float32)),
                    pip_requirements=[f"tensorflow=={tf.__version__}"],
                    metadata={"output_semantics": "delta_kg; wrapper adds last observed weight"})
                (path / "scaler.json").write_text(json.dumps({"mean": mean.tolist(), "scale": scale.tolist(),
                    "fit_start": metadata["scaler_fit_start"], "fit_end": metadata["scaler_fit_end"]}, indent=2))
                (path / "schema.json").write_text(json.dumps({"features": FEATURES, "window": WINDOW,
                    "input_units": ["kg", "celsius"], "model_output": "delta_kg",
                    "served_output": "last_weight_kg + delta_kg", "schema_version": 1}, indent=2))
                (path / "metadata.json").write_text(json.dumps(metadata, indent=2))
                self.client.log_artifacts(run_id, str(path), artifact_path="bundle")
            version_record = self.client.create_model_version(REGISTRY_NAME,
                source=f"{run.info.artifact_uri}/bundle", run_id=run_id,
                tags={"snapshot_id": str(snapshot_id), "parent_version": metadata["parent_version"],
                      "promotion_policy": policy, "gate_policy_version": policy})
            version = str(version_record.version)
            reasons = evaluate_gate(mae, incumbent_mae, baseline_mae, validation_baselines["seasonal_24h"]) if quality_demo else []
            loaded = None
            if not reasons:
                try:
                    loaded = self._load_bundle(version)
                    np.testing.assert_allclose(self._predict_bundle(loaded, val_x[:2]),
                                               self._predict_bundle(candidate, val_x[:2]), rtol=1e-5, atol=1e-5)
                except Exception as exc:
                    LOGGER.exception("Candidate %s failed load/smoke verification", version)
                    reasons.append("load_failed")
                    self.client.set_tag(run_id, "load_error", str(exc)[:1000])
            result = {"promoted": False, "version": version,
                "parent_version": parent.version if parent else None, "run_id": run_id,
                "mae": mae, "incumbent_mae": incumbent_mae, "baseline_mae": baseline_mae,
                "seasonal_24h_mae": validation_baselines["seasonal_24h"],
                "best_baseline_name": best_baseline_name, "best_baseline_mae": best_baseline_mae,
                "validation_mae_skill_vs_best_baseline": baseline_skill(mae, best_baseline_mae),
                "promotion_policy": policy, "gate_policy_version": policy,
                "reused_observations_for_demo": reused_for_demo or comparison_demo,
                "minimum_relative_gain_pct": MINIMUM_BASELINE_GAIN * 100 if quality_demo else None,
                "test_mae": metrics["test_mae"], "test_baseline_mae": metrics["test_baseline_mae"],
                "test_seasonal_24h_mae": test_baselines["seasonal_24h"],
                "test_mae_skill_vs_best_baseline": metrics["test_mae_skill_vs_best_baseline"],
                "gate_reasons": reasons, "snapshot_id": str(snapshot_id),
                **{key: metadata[key] for key in ("train_start", "train_end", "validation_start", "validation_end", "test_start", "test_end")}}
            if comparison_demo:
                result.update(comparison_ready=not reasons, deployment_action="comparison_only")
            self.client.set_model_version_tag(REGISTRY_NAME, version, "gate_passed", str(not reasons).lower())
            self.client.set_tag(run_id, "promoted", "false")
            self.client.set_tag(run_id, "gate_reasons", json.dumps(reasons))
            self.client.set_terminated(run_id, "FINISHED")
            with self._promotion_lock:
                self._state["consumed_through"] = max(
                    filter(None, [self._state.get("consumed_through"), metadata["test_end"]]), key=pd.Timestamp)
                self._state["last_training"] = result
                self._state["promotion_pending"] = not reasons and not comparison_demo
                # This durable preparation must succeed before the alias moves.
                self._save_state()
                if not reasons and comparison_demo:
                    self._cache[version] = loaded
                elif not reasons:
                    if self._champion is not parent:
                        reasons.append("champion_changed_during_training")
                    else:
                        try:
                            self._set_alias_safely(version, parent.version if parent else None)
                        except Exception:
                            LOGGER.exception("Candidate %s alias update failed", version)
                            reasons.append("alias_update_failed")
                        else:
                            self._cache[version] = loaded
                            self._champion = loaded
                            result["promoted"] = True
                self._state.pop("promotion_pending", None)
                # Promotion has committed. Secondary reporting failures cannot
                # turn a successful serving switch into a failed training job.
                try:
                    self._save_state()
                    self.client.set_tag(run_id, "promoted", str(result["promoted"]).lower())
                    self.client.set_tag(run_id, "gate_reasons", json.dumps(reasons))
                except Exception:
                    LOGGER.exception("Post-promotion reporting deferred; startup reconciles the prepared state with the alias")
            return result
        except Exception:
            if run_id:
                try:
                    self.client.set_terminated(run_id, "FAILED")
                except Exception:
                    LOGGER.exception("Could not mark failed run %s", run_id)
            raise
        finally:
            self._training_lock.release()

    def _save_state(self):
        atomic_json(self._state_path, self._state)

    def _set_alias_safely(self, version: str, previous: str | None):
        try:
            self.client.set_registered_model_alias(REGISTRY_NAME, ALIAS, version)
        except Exception:
            # A storage adapter can raise after a committed write. Reconcile
            # before reporting failure, so cache and registry remain consistent.
            registry = self.client.get_registered_model(REGISTRY_NAME)
            current = registry.aliases.get(ALIAS)
            if current is not None and str(current) != previous:
                if previous is None:
                    self.client.delete_registered_model_alias(REGISTRY_NAME, ALIAS)
                else:
                    self.client.set_registered_model_alias(REGISTRY_NAME, ALIAS, previous)
            raise

    def _load_bundle(self, version: str) -> Bundle:
        record = self.client.get_model_version(REGISTRY_NAME, str(version))
        # Resolve this exact version's run, never a global/latest scaler file.
        path = Path(self.client.download_artifacts(record.run_id, "bundle"))
        schema = json.loads((path / "schema.json").read_text())
        if schema.get("features") != FEATURES or schema.get("window") != WINDOW or schema.get("model_output") != "delta_kg":
            raise ValueError("Incompatible model bundle schema")
        scaler = json.loads((path / "scaler.json").read_text())
        mean, scale = np.asarray(scaler["mean"]), np.asarray(scaler["scale"])
        if mean.shape != (2,) or scale.shape != (2,) or not np.isfinite([mean, scale]).all() or (scale <= 0).any():
            raise ValueError("Invalid model bundle scaler")
        mean.setflags(write=False)
        scale.setflags(write=False)
        model = mlflow_keras.load_model(str(path / "model"), load_model_kwargs={"compile": False})
        bundle = Bundle(str(version), model, mean, scale, json.loads((path / "metadata.json").read_text()))
        smoke = self._predict_bundle(bundle, np.tile(mean, (1, WINDOW, 1)).astype(np.float32))
        if smoke.shape != (1,) or not np.isfinite(smoke).all():
            raise ValueError("Model bundle failed inference smoke test")
        return bundle

    def versions(self) -> list[dict]:
        current = self.status()["version"]
        result = []
        for record in self.client.search_model_versions(f"name='{REGISTRY_NAME}'"):
            run = self.client.get_run(record.run_id)
            result.append({"version": str(record.version), "run_id": record.run_id,
                "champion": str(record.version) == current, "created_at": record.creation_timestamp,
                "metrics": dict(run.data.metrics), "params": dict(run.data.params),
                "gate_passed": record.tags.get("gate_passed") == "true",
                "gate_reasons": json.loads(run.data.tags.get("gate_reasons", "[]")),
                "promotion_policy": run.data.params.get("promotion_policy", run.data.params.get("gate_policy_version", "legacy_quality_gate")),
                "run_status": run.info.status, "parent_version": run.data.params.get("parent_version"),
                "data_source": run.data.params.get("data_source"),
                "source": record.source})
        return sorted(result, key=lambda row: int(row["version"]), reverse=True)

    def version_metadata(self, version: str) -> dict:
        """Exact-version ranges; never substitute the current champion's cutoff."""
        record = self.client.get_model_version(REGISTRY_NAME, str(version))
        run = self.client.get_run(record.run_id)
        return {**dict(run.data.params), "version": str(record.version),
                "run_id": record.run_id, "monitoring_after": run.data.params.get("test_end")}

    def rollback(self, version: str) -> dict:
        version = str(version)
        try:
            record = self.client.get_model_version(REGISTRY_NAME, version)
            if record.tags.get("gate_passed") != "true":
                raise ValueError("Version did not pass the quality gate")
            loaded = self._load_bundle(version)
        except Exception as exc:
            raise ValueError(f"Version {version} could not be loaded: {exc}") from exc
        with self._promotion_lock:
            previous = self._champion.version if self._champion else None
            self._save_state()
            self._set_alias_safely(version, previous)
            self._cache[version] = loaded
            self._champion = loaded
        return {"version": version, "previous_version": previous, "rolled_back": True}
