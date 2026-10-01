"""Shared retraining visibility is global, read-only, and restart-safe."""
from copy import deepcopy
from datetime import datetime
import json

from fastapi.testclient import TestClient
import pytest

from app.main import create_app
from app.temperature_drift import TemperatureDriftCoordinator
from test_api import ForecastFixture
from test_temperature_drift import Provider, coordinator


def stored_job(job_id, status, hive='OTHER-HIVE'):
    return {'job_id':job_id, 'workspace_id':hive, 'status':status,
            'stage':'train' if status=='running' else 'trigger',
            'requested_at':'2026-01-01T00:00:00+00:00', 'steps':[]}


def test_global_api_exposes_running_work_before_newer_queue_without_writing(tmp_path):
    monitor, _=coordinator(tmp_path/'monitor')
    monitor.state['jobs']=[stored_job(f'old-{i}','completed') for i in range(12)]+[
        stored_job('failed','failed'), stored_job('active','running','ACTIVE-HIVE'),
        stored_job('queued-first','queued'), stored_job('queued-last','queued')]
    monitor._save()
    saved=monitor.path.read_bytes()
    state=deepcopy(monitor.state)
    with TestClient(create_app(tmp_path/'api',model=ForecastFixture(),temperature_monitor=monitor)) as client:
        response=client.get('/monitoring/retraining',params={'workspace_id':'DOES-NOT-EXIST'})
        assert response.status_code==200,response.text
        assert response.headers['cache-control']=='no-store'
        result=response.json()
        assert result['enabled'] is True
        assert result['active_model']['version']=='2'
        assert result['running_job']['job_id']=='active'
        assert [job['job_id'] for job in result['queued_jobs']]==['queued-first','queued-last']
        assert result['latest_job']['job_id']=='queued-last'
        assert [job['job_id'] for job in result['jobs']]==[
            'queued-last','queued-first','active','failed','old-11','old-10','old-9','old-8','old-7','old-6']
        assert datetime.fromisoformat(result['server_time']).utcoffset().total_seconds()==0
        assert monitor.state==state
        assert monitor.path.read_bytes()==saved
        assert client.get('/data/status').json()['rows']==0


def test_restarted_monitor_retains_epoch_progress_and_completed_failed_history(tmp_path):
    monitor, provider=coordinator(tmp_path)
    progress={'completed_epochs':3,'total_epochs':8,'training_windows':144,'last_loss':.25,
              'updated_at':'2026-01-01T00:00:03+00:00'}
    steps=[{'key':'trigger','status':'completed','completed_at':'2026-01-01T00:00:00+00:00'},
           {'key':'train','status':'running','started_at':'2026-01-01T00:00:01+00:00'},
           {'key':'register','status':'pending'}]
    monitor.state['jobs']=[stored_job('completed','completed'),stored_job('failed','failed'),
        {**stored_job('active','running'),'training_progress':progress,'steps':steps},stored_job('pending','queued')]
    monitor._save()
    monitor.close()
    restored=TemperatureDriftCoordinator(tmp_path,provider)
    try:
        result=restored.retraining_status()
        assert result['running_job'] is None
        assert result['queued_jobs']==[]
        assert [job['status'] for job in result['jobs']]==['interrupted','interrupted','failed','completed']
        active=next(job for job in result['jobs'] if job['job_id']=='active')
        assert active['training_progress']==progress
        assert active['steps'][0]==steps[0]
        assert active['steps'][2]==steps[2]
        assert active['steps'][1]=={**steps[1],'status':'interrupted',
                                   'error':active['error'],'completed_at':active['completed_at']}
        result['jobs'][1]['training_progress']['completed_epochs']=99
        persisted=json.loads(restored.path.read_text())
        assert persisted['jobs'][2]['training_progress']['completed_epochs']==3
        assert persisted['jobs'][2]['steps']==active['steps']
        assert restored.retraining_status()['jobs'][1]['training_progress']['completed_epochs']==3
    finally:
        restored.close()


@pytest.mark.parametrize('stub',[False,True])
def test_global_api_has_stable_empty_contract_without_supported_monitor(tmp_path,stub):
    class LegacyMonitor:
        def close(self): pass
    with TestClient(create_app(tmp_path,model=ForecastFixture(),
                              temperature_monitor=LegacyMonitor() if stub else None)) as client:
        response=client.get('/monitoring/retraining')
        assert response.status_code==200,response.text
        result=response.json()
        assert result['enabled'] is False
        assert result['active_model'] is result['running_job'] is result['latest_job'] is None
        assert result['queued_jobs']==result['jobs']==[]
        assert datetime.fromisoformat(result['server_time']).utcoffset().total_seconds()==0
