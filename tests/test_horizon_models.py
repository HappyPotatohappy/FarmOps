"""Model contribution, immutable bundle, and truthful serving contract tests."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from app import horizon_models as hm


def rows(n=30):
    return [{'timestamp': t.isoformat(), 'hive_id': 'real', 'weight_kg': 40+i*.01,
             'temperature_c': 20., 'event': 'normal'} for i,t in
            enumerate(pd.date_range('2025-01-01', periods=n, freq='h', tz='UTC'))]


def manifest(root, **overrides):
    body = {'registry_id': 'BeeOPS_Horizon_Weight', 'version': '1', 'run_id': 'unit-test',
            'context_hours': 24, 'max_horizon_hours': 168, 'components': ['lstm'],
            'weights': {'lstm': 1.}, 'scaler': {'weight_scale': 1., 'temperature_mean': 20., 'temperature_scale': 1.},
            'files': {'lstm': 'lstm.keras'}, 'validation': {'status': 'insufficient', 'calibrated': False,
             'noise_kg': .01, 'horizons': {}}, 'interval_radii_kg': [], **overrides}
    folder = root / '1'; folder.mkdir(parents=True, exist_ok=True)
    (folder / 'manifest.json').write_text(json.dumps(body))
    (root / 'index.json').write_text(json.dumps({'registry_id': 'BeeOPS_Horizon_Weight',
        'default_version': '1', 'versions': [{'version': '1', 'manifest': '1/manifest.json'}]}))
    return body


def test_constrained_selection_preserves_real_lstm_contribution():
    target = np.zeros((4,168))
    predictions = {'lstm': np.ones((4,168))*10, 'lightgbm': np.zeros((4,168))}
    weights = hm.select_weights(predictions, target)
    assert weights['lstm'] == pytest.approx(.1)
    assert weights['lightgbm'] == pytest.approx(.9)
    assert sum(weights.values()) == pytest.approx(1.)
    assert hm.combine_predictions(predictions, weights)[0,0] == pytest.approx(1.)
    with pytest.raises(ValueError, match='lstm'):
        hm.select_weights({'lightgbm': np.zeros((4,168))}, target)


def test_missing_versions_and_artifacts_do_not_fabricate_forecasts(tmp_path):
    service = hm.HorizonModelService(tmp_path)
    assert service.catalog()['versions'] == []
    assert service.predict(rows(), 24)['status'] == 'model_unavailable'
    manifest(tmp_path)
    assert service.predict(rows(), 24, 'no-such-version')['status'] == 'model_unavailable'
    assert service.predict(rows(), 48, '1')['status'] == 'unsupported_horizon'
    response = service.predict(rows(4), 24, '1')
    assert response['status'] == 'insufficient_history'
    assert response['trajectory'] == []
    assert service.predict(rows(), 24, '1')['status'] == 'model_unavailable'


def test_gap_still_requires_continuous_history(tmp_path):
    manifest(tmp_path)
    service = hm.HorizonModelService(tmp_path)
    gap = rows(60); del gap[-8]
    assert service.predict(gap, 24)['status'] == 'insufficient_history'


def test_quality_gate_cannot_hide_single_hive_regression_or_small_samples():
    evidence = {'a': {'ensemble_mae': .8, 'baseline_mae': 1., 'independent_windows': 10},
                'b': {'ensemble_mae': 1.2, 'baseline_mae': 1., 'independent_windows': 10}}
    assert hm.quality_gate(evidence)['status'] == 'underperforming'
    evidence['b']['ensemble_mae'] = .8
    assert hm.quality_gate(evidence)['status'] == 'passed'
    evidence['b']['independent_windows'] = 1
    assert hm.quality_gate(evidence)['status'] == 'insufficient'


def test_real_lstm_export_reload_predicts_identical_trajectory(tmp_path):
    tf = pytest.importorskip('tensorflow')
    body = manifest(tmp_path)
    x = np.array([[[0.,0.]]*24, [[.1,.1]]*24], dtype=np.float32)
    y = np.array([[.2]*168, [.3]*168], dtype=np.float32)
    model, history = hm.train_lstm(x,y,epochs=1,seed=7)
    assert history['epochs_completed'] == 1
    model.save(tmp_path / '1/lstm.keras')
    service = hm.HorizonModelService(tmp_path)
    prediction = service.predict(rows(), 24)
    reloaded = hm.HorizonModelService(tmp_path).predict(rows(), 24)
    assert prediction['status'] == 'ok'
    assert prediction['trajectory'] == reloaded['trajectory']
    assert len(prediction['trajectory']) == 24
    assert prediction['trajectory'][0]['timestamp'] == '2025-01-02T06:00:00+00:00'
    assert prediction['trajectory'][0]['lower_kg'] is None
    assert prediction['validation']['status'] == 'insufficient'
    assert prediction['model']['weights']['lstm'] == 1.
    assert prediction['reasons'] == []
    event_rows = rows(); event_rows[-5]['event'] = 'harvest'
    with_event = service.predict(event_rows, 24)
    assert with_event['status'] == 'ok'
    assert with_event['trajectory'] == prediction['trajectory']
    assert with_event['validation']['input_event_counts'] == {'harvest': 1}
    assert with_event['reasons'] == []


def test_invalid_nonfinite_model_output_is_an_error_not_success(tmp_path, monkeypatch):
    manifest(tmp_path)
    service = hm.HorizonModelService(tmp_path)
    # Real heavy inference is tested above. This narrow fault injection asserts
    # that invalid backend output cannot escape as a success response.
    monkeypatch.setattr(service, '_predict_components', lambda *a: {'lstm': np.full(168, np.nan)})
    with pytest.raises(ValueError, match='finite'):
        service.predict(rows(), 24)


def test_validation_is_scoped_to_hive_and_model_calendar_cutoff(tmp_path, monkeypatch):
    validation = {'status': 'passed', 'calibrated': True, 'noise_kg': .01,
                  'horizons': {'24': {'status': 'passed', 'calibrated': True,
                    'per_hive': {'real': {'independent_windows': 8, 'ensemble_mae': .1, 'baseline_mae': .2}}}}}
    manifest(tmp_path, validation=validation, interval_radii_kg=[.2]*168,
             training_calendar_cutoff='2025-02-01T00:00:00+00:00')
    service = hm.HorizonModelService(tmp_path)
    monkeypatch.setattr(service, '_predict_components', lambda *a: {'lstm': np.full(168,40.)})
    result = service.predict(rows(),24)
    assert result['validation']['status'] == 'insufficient'
    assert result['trajectory'][0]['lower_kg'] is None
    assert result['status'] == 'ok'
    assert result['reasons'] == []
    assert any('postdates' in reason for reason in result['validation']['context_notes'])
    unseen = rows()
    for item in unseen:
        item['hive_id'] = 'unseen'
        item['timestamp'] = item['timestamp'].replace('2025','2026')
    assert service.predict(unseen,24)['validation']['status'] == 'insufficient'


def test_unseen_training_hive_cannot_inherit_pooled_validation_pass(tmp_path, monkeypatch):
    validation = {'status': 'passed', 'calibrated': True, 'noise_kg': .01,
      'horizons': {'24': {'status': 'passed','calibrated': True,
       'per_hive': {'real': {'independent_windows': 10,'ensemble_mae': .1,'baseline_mae': .2}}}}}
    manifest(tmp_path,validation=validation,training_hive_ids=['other_hive'],interval_radii_kg=[.2]*168)
    service=hm.HorizonModelService(tmp_path)
    monkeypatch.setattr(service,'_predict_components',lambda *a: {'lstm':np.full(168,40.)})
    result=service.predict(rows(),24)
    assert result['validation']['status']=='insufficient'
    assert result['validation']['calibrated'] is False
    assert result['status'] == 'ok'
    assert result['reasons'] == []
    assert 'hive_was_not_represented_in_model_training' in result['validation']['context_notes']


def test_zero_weight_unavailable_component_is_not_required_for_prediction():
    actual=hm.combine_predictions({'lstm':np.array([3.,4.])},{'lstm':1.,'chronos2':0.})
    np.testing.assert_array_equal(actual,[3.,4.])
    with pytest.raises(ValueError,match='Missing'):
        hm.combine_predictions({'lstm':np.array([3.])},{'lstm':.5,'chronos2':.5})


def test_partial_lstm_targets_are_masked_without_nonfinite_training(tmp_path):
    pytest.importorskip('tensorflow')
    x=np.zeros((4,24,2),dtype=np.float32)
    target=np.full((4,168),np.nan,dtype=np.float32);target[:,:24]=.25
    model,history=hm.train_lstm(x,target,epochs=1)
    assert np.isfinite(history['training_mae_scaled']).all()
    assert np.isfinite(np.asarray(model(x,training=False))).all()


def test_validation_selection_and_calibration_availability_cutoff_holds(tmp_path,monkeypatch):
    validation={'status':'passed','calibrated':True,'noise_kg':.01,'horizons':{'24':{'status':'passed','calibrated':True}}}
    manifest(tmp_path,validation=validation,training_calendar_cutoff='2024-01-01T00:00Z',
             bundle_available_after='2025-02-01T00:00Z',interval_radii_kg=[.2]*168)
    service=hm.HorizonModelService(tmp_path)
    monkeypatch.setattr(service,'_predict_components',lambda *a:{'lstm':np.full(168,40.)})
    actual=service.predict(rows(),24)
    assert actual['validation']['status']=='insufficient'
    assert actual['validation']['calibrated'] is False
    assert actual['trajectory'][0]['lower_kg'] is None


def test_insufficient_gate_still_reports_actual_equal_hive_metrics():
    result=hm.quality_gate({'a':{'ensemble_mae':.8,'baseline_mae':1.,'independent_windows':1}})
    assert result['status']=='insufficient'
    assert result['equal_hive_mae_kg']==pytest.approx(.8)
    assert result['relative_improvement']==pytest.approx(.2)


@pytest.mark.parametrize('field,value',[('weight_kg',float('nan')),('temperature_c',float('inf')),('weight_kg',-1.),('temperature_c',100.)])
def test_invalid_observed_values_cannot_produce_a_success(tmp_path,monkeypatch,field,value):
    manifest(tmp_path)
    service=hm.HorizonModelService(tmp_path)
    monkeypatch.setattr(service,'_predict_components',lambda *a:{'lstm':np.full(168,40.)})
    observed=rows();observed[-1][field]=value
    with pytest.raises(ValueError,match='finite|physical'):
        service.predict(observed,24)
