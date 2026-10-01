"""Contract tests for real LSTM, temporal isolation and safe MLflow promotion."""
from importlib import import_module, util
import hashlib
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import Event, current_thread

import numpy as np
import pandas as pd
import pytest


def ml_module():
    assert util.find_spec("app.ml") is not None, "BeeOPS model service is not implemented"
    return import_module("app.ml")


def frame(rows=240, start="2026-01-01", weight=40.0):
    # A learnable 10 g/hour trend, rather than a constant series on which an
    # untrained zero-residual LSTM merely ties the perfect persistence baseline.
    return pd.DataFrame({
        "timestamp": pd.date_range(start, periods=rows, freq="h", tz="UTC"),
        "hive_id": "test-hive",
        "weight_kg": weight + (np.arange(rows) - rows) * .01,
        "temperature_c": 20 + 3 * np.sin(np.arange(rows) * np.pi / 12),
        "event": "normal",
    })


def test_temporal_partitions_have_disjoint_raw_rows_and_clean_new_data():
    ml = ml_module()
    data = frame()
    parts = ml.prepare_splits(data)
    assert len(parts.train) == 144
    assert len(parts.validation) == len(parts.test) == 48
    assert parts.train.timestamp.max() < parts.validation.timestamp.min()
    assert parts.validation.timestamp.max() < parts.test.timestamp.min()
    assert ml.make_windows(parts.validation)[0].shape == (24, 24, 2)
    with pytest.raises(ValueError, match="168"):
        ml.prepare_splits(data, after=data.timestamp.iloc[100].isoformat())
    data.loc[100, "event"] = "harvest"
    with pytest.raises(ValueError, match="168"):
        ml.prepare_splits(data)
    data = frame()
    data.loc[100, "timestamp"] += pd.Timedelta(minutes=1)
    with pytest.raises(ValueError, match="hour"):
        ml.prepare_splits(data)


def test_gate_rejects_absolute_or_relative_regression_and_nonfinite_scores():
    gate = ml_module().evaluate_gate
    assert gate(0.10, 0.12, 0.11) == []
    assert "absolute_mae" in gate(0.16, 0.20, 0.20)
    assert "incumbent_regression" in gate(0.10, 0.08, 0.11)
    assert "persistence_regression" in gate(0.10, 0.12, 0.09)
    assert "nonfinite_metric" in gate(float("nan"), 0.12, 0.11)


def test_prediction_rejects_shape_nonfinite_and_unready(tmp_path):
    service = ml_module().ModelService(tmp_path)
    for invalid in (np.zeros((24, 2)), np.zeros((1, 23, 2)), np.full((1, 24, 2), np.nan)):
        with pytest.raises(ValueError):
            service.predict(invalid)
    with pytest.raises(RuntimeError, match="ready"):
        service.predict(np.full((1, 24, 2), 20.0))


@pytest.fixture
def trained(tmp_path):
    service = ml_module().ModelService(tmp_path)
    result = service.bootstrap(frame())
    assert result["promoted"] is True
    return service


def test_real_lstm_registry_scaler_restart_new_training_and_rollback(trained):
    service = trained
    window = frame().iloc[-24:][["weight_kg", "temperature_c"]].to_numpy()[None]
    before, version = service.predict(window)
    assert before.shape == (1,)
    assert before[0] == pytest.approx(40, abs=0.001)
    entry = service.versions()[0]
    assert entry['gate_reasons']==[]
    assert service.version_metadata(version)['monitoring_after']==entry['params']['test_end']
    run = service.client.get_run(entry["run_id"])
    bundle = Path(service.client.download_artifacts(entry["run_id"], "bundle"))
    scaler = json.loads((bundle / "scaler.json").read_text())
    schema = json.loads((bundle / "schema.json").read_text())
    assert scaler["mean"][0] == pytest.approx(frame().iloc[:144].weight_kg.mean(), abs=1e-12)
    assert schema["features"] == ["weight_kg", "temperature_c"]
    assert (bundle / "model" / "MLmodel").is_file()
    assert run.data.metrics["validation_samples"] >= 24
    assert run.data.metrics["test_samples"] >= 24
    assert pd.Timestamp(run.data.params["train_end"]) < pd.Timestamp(run.data.params["validation_start"])
    assert pd.Timestamp(run.data.params["validation_end"]) < pd.Timestamp(run.data.params["test_start"])
    assert run.data.params["scaler_fit_end"] == run.data.params["train_end"]
    restarted = type(service)(service.runtime)
    np.testing.assert_allclose(restarted.predict(window)[0], before, atol=1e-6)
    result = service.train_candidate(frame(start="2026-01-11", weight=41), "drift", "new-snapshot")
    assert result["promoted"] is True
    assert result["parent_version"] == version
    assert pd.Timestamp(result["train_start"]) > pd.Timestamp("2026-01-10T23:00:00Z")
    assert result["version"] != version
    assert str(service.client.get_model_version_by_alias("BeeOPS_Weight", "champion").version) == result["version"]
    np.testing.assert_allclose(service.predict(window, version=version)[0], before, atol=1e-6)
    service.rollback(version)
    assert service.status()["version"] == version
    assert str(service.client.get_model_version_by_alias("BeeOPS_Weight", "champion").version) == version
    with pytest.raises(ValueError, match="168"):
        service.train_candidate(frame(start="2026-01-11", weight=41), "repeat", "same-snapshot")


def test_failed_quality_gate_keeps_champion(trained):
    service = trained
    incumbent = service.status()["version"]
    noisy = frame(start="2026-01-11")
    noisy.loc[144:, "weight_kg"] = 40 + np.random.default_rng(12).uniform(-2, 2, 96)
    result = service.train_candidate(noisy, "synthetic_unpredictable_gate_test", "bad-candidate")
    assert result["promoted"] is False
    assert "absolute_mae" in result["gate_reasons"]
    assert result['promotion_policy']==result['gate_policy_version']=='quality_gate_demo_v1'
    assert service.status()["version"] == incumbent
    assert str(service.client.get_model_version_by_alias("BeeOPS_Weight", "champion").version) == incumbent
    with pytest.raises(ValueError, match="gate"):
        service.rollback(result["version"])
    before=service.client.get_model_version('BeeOPS_Weight',result['version']).tags
    restarted=type(service)(service.runtime)
    assert restarted.client.get_model_version('BeeOPS_Weight',result['version']).tags==before
    with pytest.raises(ValueError,match='gate'):
        restarted.rollback(result['version'])


def test_normal_bootstrap_registers_trained_lstm_despite_poor_absolute_performance(tmp_path):
    ml=ml_module(); service=ml.ModelService(tmp_path)
    data=frame()
    data.loc[144:,'weight_kg']=40+np.random.default_rng(12).uniform(-2,2,96)
    result=service.bootstrap(data)
    assert result['mae']>.15
    assert result['promoted'] and service.status()['ready']
    assert result['gate_reasons']==[]
    assert result['promotion_policy']==result['gate_policy_version']=='forecast_demo_v1'
    assert service.status()['promotion_policy']=='forecast_demo_v1'
    version=service.versions()[0]
    assert version['promotion_policy']=='forecast_demo_v1'
    assert version['params']['gate_policy_version']=='forecast_demo_v1'
    run=service.client.get_run(version['run_id'])
    assert run.data.tags['promotion_policy']=='forecast_demo_v1'
    assert service.client.get_model_version('BeeOPS_Weight',result['version']).tags['promotion_policy']=='forecast_demo_v1'
    assert version['metrics']['selected_epoch']>=1
    prediction,used=service.predict(data.iloc[-24:][['weight_kg','temperature_c']].to_numpy()[None])
    assert used==result['version'] and np.isfinite(prediction).all()


def test_normal_retraining_does_not_block_incumbent_or_baseline_regression(trained):
    service=trained; original=service.status()['version']
    data=frame(start='2026-01-11')
    data.loc[144:,'weight_kg']=40+np.random.default_rng(12).uniform(-2,2,96)
    result=service.train_candidate(data,'operator_request','ordinary-demo-training')
    assert result['mae']>.15
    assert result['promoted'] and service.status()['version']==result['version']!=original
    assert result['gate_reasons']==[] and result['promotion_policy']=='forecast_demo_v1'
    assert service.versions()[0]['metrics']['selected_epoch']>=1


def test_checkpoint_selection_uses_trained_epochs_even_when_epoch_zero_is_better(tmp_path):
    ml=ml_module(); service=ml.ModelService(tmp_path)
    data=frame()
    data.loc[144:,'weight_kg']=data.iloc[143].weight_kg
    result=service.bootstrap(data)
    version=service.versions()[0]
    assert version['metrics']['baseline_mae']==0
    assert version['metrics']['selected_epoch']>=1
    assert version['metrics']['validation_mae']>0
    assert result['promoted']
    window=data.iloc[-24:][['weight_kg','temperature_c']].to_numpy()[None]
    predictions,_=service.predict(window)
    assert abs(predictions[0]-window[0,-1,0])>1e-5


def test_no_finite_trained_checkpoint_preserves_incumbent(trained,monkeypatch):
    service=trained; original=service.status()['version']
    monkeypatch.setattr(service,'_predict_bundle',lambda bundle,windows:np.full(len(windows),np.nan))
    with pytest.raises(RuntimeError,match='finite trained checkpoint'):
        service.train_candidate(frame(start='2026-01-11'),'operator_request','nonfinite-training')
    assert service.status()['version']==original
    assert str(service.client.get_model_version_by_alias('BeeOPS_Weight','champion').version)==original


def legacy_rejection(tmp_path,policy='baseline_gain_v2',reason='bootstrap',gate_reasons=None):
    """Build the durable registry/state boundary of an old rejected bootstrap."""
    service=ml_module().ModelService(tmp_path);data=frame()
    run=service.client.create_run(service.experiment_id)
    cutoff=data.timestamp.iloc[-1].isoformat()
    for key,value in {'reason':reason,'gate_policy_version':policy,'parent_version':'none','test_end':cutoff}.items():
        service.client.log_param(run.info.run_id,key,value)
    version=service.client.create_model_version('BeeOPS_Weight',source=(tmp_path/'legacy-bundle').as_uri(),
        run_id=run.info.run_id,tags={'gate_passed':'false'})
    rejected={'promoted':False,'version':str(version.version),'run_id':run.info.run_id,
              'parent_version':None,'gate_policy_version':policy,'test_end':cutoff,
              'gate_reasons':gate_reasons or ['baseline_improvement_below_minimum']}
    service.client.set_tag(run.info.run_id,'gate_reasons',json.dumps(rejected['gate_reasons']))
    service.client.set_terminated(run.info.run_id,'FINISHED')
    service._state={'consumed_through':cutoff,'last_training':rejected};service._save_state()
    return type(service)(tmp_path),data,rejected


def test_old_initial_performance_rejection_reinitializes_once_without_retagging(tmp_path):
    service,data,old=legacy_rejection(tmp_path)
    assert service.status()['can_reinitialize_for_demo'] is True
    old_tags=dict(service.client.get_model_version('BeeOPS_Weight',old['version']).tags)
    result=service.bootstrap(data)
    assert result['promoted'] and result['version']!=old['version']
    assert result['promotion_policy']=='forecast_demo_v1'
    assert result['reused_observations_for_demo'] is True
    assert service.version_metadata(result['version'])['reused_observations_for_demo']=='True'
    assert service.status()['can_reinitialize_for_demo'] is False
    assert pd.Timestamp(service.status()['consumed_through'])>=pd.Timestamp(old['test_end'])
    assert service.client.get_model_version('BeeOPS_Weight',old['version']).tags==old_tags
    with pytest.raises(ValueError,match='gate'):
        service.rollback(old['version'])
    assert type(service)(tmp_path).status()['can_reinitialize_for_demo'] is False


@pytest.mark.parametrize('policy,reason,reasons',[
    ('quality_gate_demo_v1','synthetic_unpredictable_gate_test',['absolute_mae']),
    ('quality_gate_demo_v1','bootstrap',['baseline_improvement_below_minimum']),
    ('baseline_gain_v2','synthetic_unpredictable_gate_test',['absolute_mae']),
    ('forecast_demo_v1','bootstrap',['baseline_improvement_below_minimum']),
    ('forecast_demo_v1','bootstrap',['load_failed']),
    ('baseline_gain_v2','bootstrap',['load_failed']),
    ('baseline_gain_v2','operator_request',['baseline_improvement_below_minimum']),
])
def test_reinitialization_does_not_reuse_current_policy_explicit_gate_or_artifact_failure(tmp_path,policy,reason,reasons):
    service,data,_=legacy_rejection(tmp_path,policy,reason,reasons)
    assert service.status()['can_reinitialize_for_demo'] is False
    with pytest.raises(ValueError,match='168'):
        service.bootstrap(data)


def test_failed_reinitialization_cannot_reconsume_legacy_snapshot(tmp_path,monkeypatch):
    service,data,_=legacy_rejection(tmp_path)
    def broken_fit(*args,**kwargs): raise RuntimeError('simulated training failure')
    monkeypatch.setattr(service,'_fit',broken_fit)
    with pytest.raises(RuntimeError,match='simulated'):
        service.bootstrap(data)
    assert service.status()['can_reinitialize_for_demo'] is False
    assert type(service)(tmp_path).status()['can_reinitialize_for_demo'] is False
    with pytest.raises(ValueError,match='168'):
        service.bootstrap(data)


def test_comparison_version_is_actually_warm_trained_on_same_data_without_switching_champion(trained):
    service=trained
    window=frame().iloc[-24:][['weight_kg','temperature_c']].to_numpy()[None]
    before,first=service.predict(window)
    first_record=service.versions()[0]
    artifact=Path(service.client.download_artifacts(first_record['run_id'],'bundle'))/'model/data/model.keras'
    before_hash=hashlib.sha256(artifact.read_bytes()).hexdigest()
    result=service.ensure_comparison_version(frame())
    assert result['status']=='created' and result['versions']==['1','2']
    assert result['comparison_version']==result['version']=='2'
    assert result['comparison_ready'] and result['reused_observations_for_demo']
    assert result['promoted'] is False and result['deployment_action']=='comparison_only'
    assert service.status()['version']==first=='1'
    assert str(service.client.get_model_version_by_alias('BeeOPS_Weight','champion').version)=='1'
    records={item['version']:item for item in service.versions()}
    second=records['2'];params=second['params']
    assert second['run_id']!=first_record['run_id']
    assert second['gate_passed'] and not second['champion']
    assert params['parent_version']=='1' and params['epochs_completed']=='8'
    assert params['reason']=='version_comparison_demo'
    assert params['reused_observations_for_demo']=='True'
    assert params['evaluation_mode']=='reused_training_observations_for_version_comparison'
    assert params['train_start']==first_record['params']['train_start']
    assert params['test_end']==first_record['params']['test_end']
    assert second['metrics']['selected_epoch']>=1
    changed,used=service.predict(window,version='2')
    assert used=='2' and np.isfinite(changed).all()
    assert np.any(changed!=before)
    np.testing.assert_array_equal(service.predict(window,version='1')[0],before)
    assert hashlib.sha256(artifact.read_bytes()).hexdigest()==before_hash


def test_comparison_preparation_is_idempotent_after_restart_and_does_not_relax_ordinary_cutoff(trained):
    service=trained
    service.ensure_comparison_version(frame())
    versions=service.versions();cutoff=service.status()['consumed_through']
    again=service.ensure_comparison_version(frame())
    assert again['status']=='already_available' and again['comparison_ready']
    restarted=type(service)(service.runtime)
    assert restarted.ensure_comparison_version(frame())['status']=='already_available'
    assert restarted.versions()==versions
    assert restarted.status()['version']=='1' and restarted.status()['consumed_through']==cutoff
    with pytest.raises(ValueError,match='168'):
        restarted.train_candidate(frame(),'operator_request','same-data-not-a-comparison')
    with pytest.raises(ValueError,match='only'):
        restarted.train_candidate(frame(),'version_comparison_demo','repeat-comparison')


def test_existing_comparison_can_be_loaded_without_making_blocked_history_deployable(trained):
    service=trained
    service.ensure_comparison_version(frame())
    service.rollback('2')
    # Model the historical Latvia v1 eligibility boundary while retaining its
    # exact artifact: a comparison may load it, but must not rewrite its verdict.
    service.client.set_model_version_tag('BeeOPS_Weight','1','gate_passed','false')
    old_tags=dict(service.client.get_model_version('BeeOPS_Weight','1').tags)
    result=service.ensure_comparison_version(frame())
    assert result['status']=='already_available' and result['versions']==['1','2']
    assert service.status()['version']=='2'
    assert service.client.get_model_version('BeeOPS_Weight','1').tags==old_tags
    assert service.predict(frame().iloc[-24:][['weight_kg','temperature_c']].to_numpy()[None],version='1')[1]=='1'
    with pytest.raises(ValueError,match='gate'):
        service.rollback('1')


@pytest.mark.parametrize("failing_method", ["log_artifacts", "create_model_version", "set_model_version_tag", "set_terminated", "_save_state"])
def test_bookkeeping_failure_before_promotion_preserves_serving_and_restart(trained, monkeypatch, failing_method):
    service = trained
    incumbent = service.status()["version"]
    target = service if failing_method == "_save_state" else service.client

    def fail(*args, **kwargs):
        raise OSError("simulated durable bookkeeping failure")

    monkeypatch.setattr(target, failing_method, fail)
    with pytest.raises(OSError):
        service.train_candidate(frame(start="2026-01-11"), "drift", "bookkeeping-fault")
    assert service.status()["version"] == incumbent
    assert str(service.client.get_model_version_by_alias("BeeOPS_Weight", "champion").version) == incumbent
    assert type(service)(service.runtime).status()["version"] == incumbent
    assert pd.Timestamp(service.status()["consumed_through"]) == pd.Timestamp("2026-01-20T23:00:00Z")
    with pytest.raises(ValueError, match="168"):
        service.train_candidate(frame(rows=241, start="2026-01-11"), "overlapping-retry", "retry-plus-one")


def test_state_failure_after_committed_promotion_is_recovered_on_restart(trained, monkeypatch):
    service = trained
    original = service._save_state
    incumbent = service.status()["version"]

    def fail_final_write():
        alias = str(service.client.get_model_version_by_alias("BeeOPS_Weight", "champion").version)
        if alias != incumbent:
            raise OSError("simulated final status write failure")
        original()

    monkeypatch.setattr(service, "_save_state", fail_final_write)
    result = service.train_candidate(frame(start="2026-01-11"), "drift", "final-state-fault")
    assert result["promoted"] is True
    restarted = type(service)(service.runtime)
    assert restarted.status()["version"] == result["version"]
    assert restarted.status()["last_training"]["promoted"] is True
    assert restarted.status()["consumed_through"] == result["test_end"]
    with pytest.raises(ValueError, match="168"):
        restarted.train_candidate(frame(start="2026-01-11"), "repeat", "repeat")


def test_nonphysical_output_is_rejected_and_origin_is_separate(trained, monkeypatch):
    service = trained
    assert service.status()["model_source"] == "mlflow"
    assert "data_source" in service.status()
    monkeypatch.setattr(service, "_predict_bundle", lambda *args: np.array([-1.0]))
    window = frame().iloc[-24:][["weight_kg", "temperature_c"]].to_numpy()[None]
    with pytest.raises(RuntimeError, match="physical"):
        service.predict(window)


def test_inflight_prediction_retains_captured_version_across_rollback(trained, monkeypatch):
    service = trained
    first = service.status()["version"]
    second = service.train_candidate(frame(start="2026-01-11"), "drift", "concurrency")["version"]
    original = service._predict_bundle
    started, resume = Event(), Event()

    def held_inference(bundle, windows):
        if current_thread().name.startswith("inflight"):
            started.set()
            assert resume.wait(5)
        return original(bundle, windows)

    monkeypatch.setattr(service, "_predict_bundle", held_inference)
    window = frame().iloc[-24:][["weight_kg", "temperature_c"]].to_numpy()[None]
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="inflight") as executor:
        future = executor.submit(service.predict, window)
        assert started.wait(5)
        try:
            service.rollback(first)
        finally:
            resume.set()
        assert future.result()[1] == second
    assert service.predict(window)[1] == first


def test_alias_adapter_failure_after_write_restores_previous_champion(trained, monkeypatch):
    service = trained
    incumbent = service.status()["version"]
    original = service.client.set_registered_model_alias

    def write_then_fail(name, alias, version):
        original(name, alias, version)
        if str(version) != incumbent:
            raise OSError("simulated error after alias commit")

    monkeypatch.setattr(service.client, "set_registered_model_alias", write_then_fail)
    result = service.train_candidate(frame(start="2026-01-11"), "drift", "alias-fault")
    assert result["promoted"] is False
    assert "alias_update_failed" in result["gate_reasons"]
    assert service.status()["version"] == incumbent
    assert str(service.client.get_model_version_by_alias("BeeOPS_Weight", "champion").version) == incumbent


def test_failed_candidate_load_keeps_cache_and_alias(trained, monkeypatch):
    service = trained
    incumbent = service.status()["version"]
    original = service._load_bundle

    def fail_new(version):
        if str(version) != incumbent:
            raise OSError("simulated artifact corruption")
        return original(version)

    monkeypatch.setattr(service, "_load_bundle", fail_new)
    result = service.train_candidate(frame(start="2026-01-11"), "drift", "load-fault")
    assert result["promoted"] is False
    assert "load_failed" in result["gate_reasons"]
    assert service.status()["version"] == incumbent
    assert str(service.client.get_model_version_by_alias("BeeOPS_Weight", "champion").version) == incumbent
