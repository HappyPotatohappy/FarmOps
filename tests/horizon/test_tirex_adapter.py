"""Boundary and leakage tests for the optional, isolated TiRex-2 candidate."""
import importlib.util
import numpy as np
import pytest


def module():
    assert importlib.util.find_spec('app.tirex_adapter') is not None, 'TiRex adapter is not implemented'
    from app import tirex_adapter
    return tirex_adapter


def test_contexts_keep_observed_temperature_and_weight_order():
    original = np.stack([np.column_stack([np.arange(24) + 40., np.arange(24) + 10.])])
    actual = module().prepare_contexts(original)
    np.testing.assert_array_equal(actual, original)
    assert actual.dtype == np.float32
    assert actual.shape == (1, 24, 2)


@pytest.mark.parametrize('value', [np.zeros((24,2)), np.zeros((0,24,2)), np.zeros((1,24,3)), np.full((1,24,2), np.nan)])
def test_invalid_contexts_fail_before_model_execution(value):
    with pytest.raises(ValueError, match='context'):
        module().prepare_contexts(value)


def test_extracts_real_median_quantile_not_mean_or_first_quantile():
    raw = [np.stack([np.full(72, q) for q in range(9)])[None] for _ in range(2)]
    result = module().median_forecasts(raw, batch_size=2, horizon=72)
    np.testing.assert_array_equal(result, np.full((2,72), 4.))


@pytest.mark.parametrize('raw', [[np.zeros((1,9,12))], [np.full((1,9,24),np.nan)], [np.zeros((2,9,24))]])
def test_incorrect_or_nonfinite_model_output_is_not_a_forecast(raw):
    with pytest.raises(ValueError, match='forecast'):
        module().median_forecasts(raw, batch_size=1, horizon=24)


def test_remote_model_id_cannot_trigger_download(tmp_path):
    with pytest.raises(FileNotFoundError, match='local'):
        module().TirexAdapter(tmp_path/'not_downloaded')


def test_ensemble_weights_ignore_calibration_and_test_labels():
    assert importlib.util.find_spec('scripts.benchmark_tirex2') is not None, 'Benchmark is not implemented'
    from scripts.benchmark_tirex2 import choose_weights
    rows = [dict(partition='validation', hive_id='a', actual_kg=0., predictions_kg={'lstm':1.,'chronos2':0.,'tirex2':2.}),
            dict(partition='test', hive_id='a', actual_kg=999., predictions_kg={'lstm':1.,'chronos2':0.,'tirex2':999.})]
    weights = choose_weights(rows)
    assert weights['lstm'] == pytest.approx(.1)
    assert weights['chronos2'] == pytest.approx(.9)
    assert weights['tirex2'] == 0.
    rows[1]['actual_kg'] = -999.
    assert choose_weights(rows) == weights


def test_ensemble_requires_actual_lstm_output():
    assert importlib.util.find_spec('scripts.benchmark_tirex2') is not None, 'Benchmark is not implemented'
    from scripts.benchmark_tirex2 import choose_weights
    with pytest.raises(ValueError, match='lstm'):
        choose_weights([dict(partition='validation', hive_id='a', actual_kg=0., predictions_kg={'chronos2':0.,'tirex2':0.})])


def test_export_never_reuses_old_uncertainty_or_overwrites_version(tmp_path):
    import json
    from scripts import benchmark_tirex2 as benchmark
    assert hasattr(benchmark, 'export_bundle'), 'Immutable candidate export is not implemented'
    old = tmp_path/'models'/'1';old.mkdir(parents=True)
    (old/'lstm.keras').write_bytes(b'unit-test-artifact')
    (old/'manifest.json').write_text(json.dumps({'registry_id':'BeeOPS_Horizon_Weight','version':'1',
        'files':{'lstm':'lstm.keras'},'scaler':{'weight_scale':1},'validation':{'calibrated':True},
        'interval_radii_kg':[999.]*168,'context_hours':168,'training_hive_ids':['a']}))
    checkpoint=tmp_path/'checkpoint';checkpoint.mkdir()
    for name in ['model.ckpt','model-config.yaml','LICENSE','NOTICE']:(checkpoint/name).write_text(name)
    report={'weights':{'lstm':.1,'tirex2':.9,'chronos2':0.},'model_revision':'test','model_sha256':'test',
       'context_hours':168,'created_at':'2026-10-01T00:00:00Z','evaluation_protocol':'retrospective_regression',
       'origins':[],'scores':{'test':{},'validation':{}},'limitations':['Regression only'],
       'baseline_report_sha256':'test','data_snapshot':'test'}
    result=benchmark.export_bundle(old,checkpoint,report,tmp_path/'models','2')
    assert result['components']==['lstm','tirex2']
    assert result['weights']=={'lstm':.1,'tirex2':.9}
    assert result['validation']['calibrated'] is False
    assert result['interval_radii_kg']==[]
    assert (tmp_path/'models'/'2'/'lstm.keras').read_bytes()==(old/'lstm.keras').read_bytes()
    assert json.loads((tmp_path/'models'/'index.json').read_text())['default_version']=='2'
    with pytest.raises(FileExistsError):benchmark.export_bundle(old,checkpoint,report,tmp_path/'models','2')
