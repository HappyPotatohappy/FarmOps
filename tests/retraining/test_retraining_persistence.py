"""Persistence faults must not invent active work or undo deployed outcomes."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.store import Store
from test_temperature_drift import coordinator, observations


def prepared(tmp_path):
    item,provider=coordinator(tmp_path)
    provider._manifest=lambda version: ({'context_hours':168,'max_horizon_hours':168,
        'components':['lstm','tirex2'],'weights':{'lstm':.1,'tirex2':.9}},tmp_path)
    service=SimpleNamespace(store=Store(tmp_path/'hive'))
    service.store.ingest(observations(),'baseline')
    item.observe(service,'TEMP-TEST')
    return item,provider,service


def successful_training(provider,called):
    def train(job,rows,reference,progress):
        called.append(job['job_id'])
        for stage in ('train','register','deploy'): progress(stage)
        provider.version='3'
        return {'new_version':'3','promoted':True,'reporting_warnings':[]}
    return train


@pytest.mark.parametrize('failure',['reference','final_state'])
def test_post_publication_reference_fault_keeps_completed_job_and_old_reference(tmp_path,monkeypatch,failure):
    item,provider,service=prepared(tmp_path)
    old=deepcopy(item.status('TEMP-TEST')['reference'])
    called=[]
    monkeypatch.setattr(item,'_train_bundle',successful_training(provider,called))
    if failure=='reference':
        def fail_reference(*args): raise OSError('reference disk full')
        monkeypatch.setattr(item,'_persist_reference',fail_reference)
    else:
        original_save=item._save
        failed=[]
        def fail_final_state_once():
            if item.state['jobs'][-1]['status']=='completed' and not failed:
                failed.append(True)
                raise OSError('final state write failed')
            original_save()
        monkeypatch.setattr(item,'_save',fail_final_state_once)
    service.store.ingest(observations(168,336,8.),'warm observations')
    item.observe(service,'TEMP-TEST')
    item.close()
    state=item.status('TEMP-TEST')
    assert state['active_model']['version']=='3'
    assert state['job']['status']=='completed'
    assert state['job']['promoted'] is True
    assert state['job']['new_version']=='3'
    assert state['job']['reporting_warnings']
    assert all(step['status']=='completed' for step in state['job']['steps'])
    assert state['reference']==old
    persisted=json.loads(item.path.read_text())
    assert persisted['jobs'][-1]['status']=='completed'
    assert persisted['workspaces']['TEMP-TEST']['reference']==old
    assert item.observe(service,'TEMP-TEST')['check']['trigger_status']=='already_evaluated'
    assert len(called)==len(item.retraining_status()['jobs'])==1


@pytest.mark.parametrize('failure',['snapshot','queued_state','submit'])
def test_enqueue_failure_never_leaves_phantom_work_and_manual_retry_is_possible(tmp_path,monkeypatch,failure):
    from app import temperature_drift
    item,provider,service=prepared(tmp_path)
    called=[]
    monkeypatch.setattr(item,'_train_bundle',successful_training(provider,called))
    with monkeypatch.context() as faults:
        if failure=='snapshot':
            original_json=temperature_drift.atomic_json
            def fail_snapshot(path,value):
                if Path(path).parent.name=='snapshots': raise OSError('snapshot disk full')
                original_json(path,value)
            faults.setattr(temperature_drift,'atomic_json',fail_snapshot)
        elif failure=='queued_state':
            original_save=item._save
            def fail_queued_state():
                if item.state['jobs'][-1]['status']=='queued': raise OSError('queued state disk full')
                original_save()
            faults.setattr(item,'_save',fail_queued_state)
        else:
            def fail_submit(*args): raise RuntimeError('worker unavailable')
            faults.setattr(item.executor,'submit',fail_submit)
        with pytest.raises((OSError,RuntimeError)):
            item.request_training(service,'TEMP-TEST')
        status=item.retraining_status()
        assert status['running_job'] is None
        assert status['queued_jobs']==[]
        assert status['latest_job']['status']=='failed'
        assert called==[]
        assert json.loads(item.path.read_text())['jobs'][-1]['status']=='failed'
    try:
        retry=item.request_training(service,'TEMP-TEST')
        assert retry['status']=='queued'
    finally:
        item.close()
    assert [job['status'] for job in item.retraining_status()['jobs']]==['completed','failed']
    assert len(called)==1


def test_worker_start_persistence_failure_does_not_leave_running_job(tmp_path,monkeypatch):
    item,provider,service=prepared(tmp_path)
    called=[]
    monkeypatch.setattr(item,'_train_bundle',successful_training(provider,called))
    original_save=item._save
    def fail_running_state():
        if item.state['jobs'][-1]['status']=='running': raise OSError('worker state disk full')
        original_save()
    monkeypatch.setattr(item,'_save',fail_running_state)
    service.store.ingest(observations(168,336,8.),'warm observations')
    item.observe(service,'TEMP-TEST')
    item.close()
    state=item.retraining_status()
    assert state['latest_job']['status']=='failed'
    assert state['running_job'] is None
    assert called==[]
    assert state['active_model']['version']=='2'
    assert json.loads(item.path.read_text())['jobs'][-1]['status']=='failed'
