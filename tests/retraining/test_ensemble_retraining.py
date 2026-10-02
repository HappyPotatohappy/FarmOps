"""Manual and temperature requests train one chronological shared ensemble."""
from copy import deepcopy
from types import SimpleNamespace
import threading

import numpy as np
import pandas as pd
import pytest

from app.store import Store
from test_temperature_drift import coordinator, observations


def test_minimum_new_day_has_disjoint_training_calibration_and_final_holdout():
    from app.temperature_drift import prepare_training
    rows=observations(192)
    result=prepare_training(rows,rows[167]['timestamp'],168,168)
    assert len(result['train'])==12
    assert len(result['calibration'])==6
    assert len(result['validation'])==6
    assert max(sample.target_end for sample in result['train'])<min(sample.target_start for sample in result['calibration'])
    assert max(sample.target_end for sample in result['calibration'])<min(sample.target_start for sample in result['validation'])
    assert max(sample.target_end for sample in result['validation'])==pd.Timestamp(rows[-1]['timestamp'])
    for partition in ('train','calibration','validation'):
        assert all(sample.origin<sample.target_start for sample in result[partition])
    changed=deepcopy(rows)
    for row in changed[-6:]:
        row['weight_kg']+=.5
    later=prepare_training(changed,rows[167]['timestamp'],168,168)
    for partition in ('train','calibration'):
        for original,modified in zip(result[partition],later[partition]):
            np.testing.assert_array_equal(original.context,modified.context)
            np.testing.assert_array_equal(original.target,modified.target)


def test_blend_weights_are_fitted_on_calibration_predictions_with_both_components_retained():
    from app.temperature_drift import fit_ensemble_weights
    weights=fit_ensemble_weights(np.array([4.,8.,12.]),np.array([0.,0.,0.]),
                                  np.array([1.,2.,3.]),{'lstm':.1,'tirex2':.9})
    assert weights['lstm']==pytest.approx(.25)
    assert weights['tirex2']==pytest.approx(.75)
    for targets in (np.zeros(3),np.array([4.,8.,12.])):
        constrained=fit_ensemble_weights(np.array([4.,8.,12.]),np.zeros(3),targets,weights)
        assert .05<=constrained['lstm']<=.95
        assert .05<=constrained['tirex2']<=.95
        assert sum(constrained.values())==pytest.approx(1.)


def test_manual_request_uses_existing_observations_once_despite_initialized_temperature_reference(tmp_path,monkeypatch):
    item,provider=coordinator(tmp_path)
    provider._manifest=lambda version: ({'context_hours':168,'max_horizon_hours':168,
        'components':['lstm','tirex2'],'weights':{'lstm':.1,'tirex2':.9}},tmp_path)
    service=SimpleNamespace(store=Store(tmp_path/'hive'))
    service.store.ingest(observations(),'existing case')
    item.observe(service,'TEMP-TEST')
    entered,release=threading.Event(),threading.Event()
    def train(job,rows,reference,progress):
        entered.set()
        assert release.wait(10)
        for stage in ('train','register','deploy'): progress(stage)
        return {'new_version':'3','promoted':True}
    monkeypatch.setattr(item,'_train_bundle',train)
    try:
        first=item.request_training(service,'TEMP-TEST')
        assert first['status']=='queued'
        assert first['workspace_id']=='TEMP-TEST'
        assert first['job']['reason']=='manual_retraining'
        assert entered.wait(10)
        repeated=item.request_training(service,'TEMP-TEST')
        assert repeated['status']=='already_evaluated'
        assert repeated['job']['job_id']==first['job']['job_id']
        assert len(item.retraining_status()['jobs'])==1
    finally:
        release.set()
        item.close()
    assert item.request_training(service,'TEMP-TEST')['status']=='already_evaluated'


def test_manual_request_requires_unconsumed_labels_and_matching_hive(tmp_path):
    item,provider=coordinator(tmp_path)
    provider._manifest=lambda version: ({'context_hours':168,'max_horizon_hours':168,
        'temperature_training':{'consumed_through_by_hive':{'TEMP-TEST':observations()[-1]['timestamp']}}},tmp_path)
    service=SimpleNamespace(store=Store(tmp_path/'hive'))
    service.store.ingest(observations(),'existing case')
    try:
        with pytest.raises(ValueError,match='new normal labels'):
            item.request_training(service,'TEMP-TEST')
        with pytest.raises(ValueError,match='hive'):
            item.request_training(service,'WRONG')
        assert item.retraining_status()['jobs']==[]
    finally:
        item.close()


@pytest.mark.parametrize('previous_status',['failed','interrupted'])
def test_manual_request_can_retry_failed_or_interrupted_snapshot_with_audit_preserved(tmp_path,monkeypatch,previous_status):
    from app.temperature_drift import TemperatureDriftCoordinator
    item,provider=coordinator(tmp_path)
    provider._manifest=lambda version: ({'context_hours':168,'max_horizon_hours':168,
        'components':['lstm','tirex2'],'weights':{'lstm':.1,'tirex2':.9}},tmp_path)
    service=SimpleNamespace(store=Store(tmp_path/'hive'))
    service.store.ingest(observations(),'existing case')
    def fail(*args):
        raise OSError('temporary artifact storage failure')
    monkeypatch.setattr(item,'_train_bundle',fail)
    first=item.request_training(service,'TEMP-TEST')
    item.close()
    assert item.retraining_status()['latest_job']['status']=='failed'
    if previous_status=='interrupted':
        item.state['jobs'][-1]['status']='running'
        item._save()
    restored=TemperatureDriftCoordinator(tmp_path,provider)
    def train(job,rows,reference,progress):
        for stage in ('train','register','deploy'): progress(stage)
        return {'new_version':'3','promoted':True}
    monkeypatch.setattr(restored,'_train_bundle',train)
    try:
        retry=restored.request_training(service,'TEMP-TEST')
        assert retry['status']=='queued'
        assert retry['job']['job_id']!=first['job']['job_id']
        assert retry['snapshot_id']==first['snapshot_id']
    finally:
        restored.close()
    jobs=restored.retraining_status()['jobs']
    assert [job['status'] for job in jobs]==['completed',previous_status]
    assert restored.request_training(service,'TEMP-TEST')['status']=='already_evaluated'
