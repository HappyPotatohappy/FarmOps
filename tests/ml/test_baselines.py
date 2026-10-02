"""Added-value checks use identical targets and never tune on the held-out test."""
from datetime import datetime, timedelta, timezone
import math

import numpy as np
import pytest

from app.store import Store


def observations(count=48, seasonal=False):
    return [
        {"timestamp": (datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i)).isoformat(),
         "hive_id": "BASELINE", "weight_kg": 40 + (.2 * math.sin(i * math.pi / 12) if seasonal else .01 * i),
         "temperature_c": 20, "event": "normal"}
        for i in range(count)
    ]


def scored_store(tmp_path, model_error=.02, seasonal=False, targets=24):
    rows = observations(24 + targets, seasonal)
    store = Store(tmp_path)
    store.ingest(rows, "known hourly measurements")
    for i in range(24, len(rows)):
        store.record_prediction(rows[i-24:i], rows[i]["weight_kg"] + model_error, "1")
    return store


def test_small_absolute_error_does_not_mask_worse_than_persistence(tmp_path):
    quality = scored_store(tmp_path).quality("1")
    assert quality["mae_kg"] < quality["threshold_kg"]
    assert quality["status"] == "underperforming"
    assert quality["absolute_status"] == "ok"
    assert quality["baseline_status"] == "underperforming"
    assert quality["persistence_count"] == quality["seasonal_24h_count"] == 24
    assert quality["persistence_mae_kg"] == pytest.approx(.01)
    assert quality["seasonal_24h_mae_kg"] == pytest.approx(.24)
    assert quality["baseline_name"] == "persistence"
    assert quality["skill_score"] == pytest.approx(-1)
    assert quality["relative_gain_pct"] == pytest.approx(-100)


def test_seasonal_baseline_prevents_claiming_value_for_a_daily_copy(tmp_path):
    quality = scored_store(tmp_path, model_error=.01, seasonal=True).quality("1")
    assert quality["mae_kg"] < quality["persistence_mae_kg"]
    assert quality["baseline_name"] == "seasonal_24h"
    assert quality["status"] == "underperforming"
    assert quality["baseline_reason"] == "baseline_is_perfect"
    assert quality["skill_score"] is None  # Never divide by a zero-error baseline.


def test_insufficient_pairs_show_metrics_without_positive_or_negative_verdict(tmp_path):
    quality = scored_store(tmp_path, targets=3).quality("1")
    assert quality["count"] == quality["baseline_count"] == 3
    assert quality["status"] == quality["baseline_status"] == "insufficient_data"
    assert quality["persistence_mae_kg"] == pytest.approx(.01)
    assert quality["relative_gain_pct"] == pytest.approx(-100)


def test_material_improvement_is_useful_but_absolute_degradation_keeps_alert_status(tmp_path):
    quality = scored_store(tmp_path / "useful", model_error=.005).quality("1")
    assert quality["status"] == "ok" and quality["baseline_status"] == "useful"
    assert quality["relative_gain_pct"] == pytest.approx(50)
    degraded = scored_store(tmp_path / "degraded", model_error=.2).quality("1")
    assert degraded["status"] == "degraded"
    assert degraded["absolute_status"] == "degraded"
    assert degraded["baseline_status"] == "underperforming"


def test_version_exclusions_apply_equally_to_both_baselines(tmp_path):
    store = scored_store(tmp_path, targets=28)
    cutoff = observations(50)[49]["timestamp"]
    store.set_model_cutoff("1", cutoff)
    quality = store.quality("1")
    assert quality["count"] == quality["persistence_count"] == quality["seasonal_24h_count"] == 2
    assert quality["baseline_status"] == "insufficient_data"
    assert store.quality("missing")["baseline_count"] == 0


def test_event_targets_and_input_contexts_do_not_enter_any_quality_score(tmp_path):
    rows = observations(50)
    rows[24]["event"] = "inspection"
    store = Store(tmp_path)
    store.ingest(rows, "observed inspection")
    for i in range(24, len(rows)):
        store.record_prediction(rows[i-24:i], rows[i]["weight_kg"] + .02, "1")
    quality = store.quality("1")
    assert quality["count"] == quality["persistence_count"] == quality["seasonal_24h_count"] == 1
    assert quality["baseline_status"] == "insufficient_data"


def test_gate_requires_material_gain_not_a_zero_residual_or_tie():
    from app.ml import evaluate_gate
    assert "baseline_improvement_below_minimum" in evaluate_gate(.01, None, .01, .2)
    assert "baseline_improvement_below_minimum" in evaluate_gate(0, None, 0, 0)
    assert "baseline_improvement_below_minimum" in evaluate_gate(.0096, None, .01, .2)
    assert evaluate_gate(.0094, None, .01, .2) == []


def test_gate_uses_strongest_baseline_and_does_not_accept_only_persistence_gain():
    from app.ml import evaluate_gate
    reasons = evaluate_gate(.04, None, .10, .02)
    assert "seasonal_24h_regression" in reasons
    assert "baseline_improvement_below_minimum" in reasons
    assert evaluate_gate(.018, None, .10, .02) == []


def test_validation_baselines_take_last_and_exactly_24_hour_old_input():
    from app.ml import baseline_scores
    windows = np.zeros((2, 24, 2), dtype=np.float32)
    windows[:, -1, 0] = [10, 20]
    windows[:, 0, 0] = [11, 21]
    scores = baseline_scores(windows, np.array([11, 21]))
    assert scores == {"persistence": 1.0, "seasonal_24h": 0.0}


def test_constant_observations_can_prepare_demo_model_without_claiming_baseline_gain(tmp_path):
    import pandas as pd
    from app.ml import ModelService
    frame = pd.DataFrame(observations(168))
    frame["weight_kg"] = 40.0
    model = ModelService(tmp_path)
    result = model.bootstrap(frame)
    assert result["mae"] == 0
    assert result["promoted"] and model.status()["ready"]
    assert result['gate_reasons']==[]
    version = model.versions()[0]
    assert version["champion"] and version["gate_passed"]
    assert version["params"]["gate_policy_version"] == "forecast_demo_v1"
    assert version['metrics']['selected_epoch']>=1
    assert version["metrics"]["validation_persistence_mae"] == 0
    assert version["metrics"]["validation_seasonal_24h_mae"] == 0
    assert version["metrics"]["test_seasonal_24h_mae"] == 0
    assert ModelService(tmp_path).versions()[0]["gate_passed"] is True
