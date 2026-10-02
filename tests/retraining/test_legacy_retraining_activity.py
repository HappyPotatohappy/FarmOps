"""Manual LSTM activity is a read-only projection of loaded service records."""
from copy import deepcopy

from fastapi.testclient import TestClient
import pytest

from app.main import create_app
from app.service import Service
from app.workspaces import WorkspaceManager, workspace_directory, workspace_metadata
from test_api import ForecastFixture
from test_store import rows


@pytest.fixture
def manager(tmp_path):
    primary = Service(tmp_path, ForecastFixture())
    value = WorkspaceManager(tmp_path, primary, model_factory=lambda runtime: ForecastFixture())
    primary.store.ingest(rows(24), 'fixture observations')
    try:
        yield value
    finally:
        value.close()


def training(status='running', **updates):
    return {'status': status, 'reason': 'operator_request', 'snapshot_id': 'snapshot-1',
            'requested_at': '2026-10-01T05:00:00+00:00', 'rows': 336,
            'reporting_warnings': ['retained warning'], **updates}


def test_loaded_training_snapshots_preserve_identity_without_opening_other_workspaces(manager, monkeypatch):
    other, _ = manager.ensure('SECOND-HIVE')
    manager.primary.training = training()
    other.training = training('completed', snapshot_id='snapshot-2', version='7',
                              completed_at='2026-10-01T06:00:00+00:00')
    manager._entries['UNLOADED-HIVE'] = {
        **workspace_metadata('UNLOADED-HIVE'),
        'path': workspace_directory(manager.root_runtime, 'UNLOADED-HIVE')}
    original = deepcopy([manager.primary.training, other.training])

    def forbidden(*args, **kwargs):
        raise AssertionError('Activity reads must not load models, open services, or write observations')

    monkeypatch.setattr(manager, 'get', forbidden)
    monkeypatch.setattr(manager, 'model_factory', forbidden)
    for service in (manager.primary, other):
        monkeypatch.setattr(service, 'health', forbidden)
        monkeypatch.setattr(service.model, 'status', forbidden)
        monkeypatch.setattr(service.store, 'ingest', forbidden)
        monkeypatch.setattr(service.store, 'set_meta', forbidden)

    jobs = manager.training_jobs()
    assert [job['workspace_id'] for job in jobs] == ['SECOND-HIVE', 'BEE-01']
    assert [job['job_id'] for job in jobs] == [
        'legacy:SECOND-HIVE:snapshot-2', 'legacy:BEE-01:snapshot-1']
    assert jobs[0]['new_version'] == '7'
    assert all(job['source'] == 'legacy_lstm' and job['stage'] == 'train' for job in jobs)
    assert all(job['steps'] == [] and 'training_progress' not in job for job in jobs)
    assert 'UNLOADED-HIVE' not in manager._services
    jobs[0]['reporting_warnings'].append('caller edit')
    assert [manager.primary.training, other.training] == original


@pytest.mark.parametrize('status', ['queued', 'running', 'completed', 'failed', 'partial', 'interrupted'])
def test_real_legacy_job_statuses_remain_distinct(manager, status):
    manager.primary.training = training(status)
    assert manager.training_jobs()[0]['status'] == status


@pytest.mark.parametrize('record', [{'status': 'idle'}, {'status': 'completed', 'reason': 'initial_upload'},
                                   {'status': 'unknown', 'snapshot_id': 'not-a-job'}])
def test_idle_and_unidentified_bootstrap_records_do_not_invent_jobs(manager, record):
    manager.primary.training = record
    assert manager.training_jobs() == []


def test_requested_time_identifies_a_job_when_snapshot_is_absent(manager):
    manager.primary.training = {'status': 'failed', 'requested_at': '2026-10-01T05:00:00+00:00',
                                'error': 'model registry unavailable'}
    record = manager.training_jobs()[0]
    assert record['job_id'] == 'legacy:BEE-01:2026-10-01T05:00:00+00:00'
    assert record['error'] == 'model registry unavailable'
    assert 'new_version' not in record


def test_retraining_api_adds_global_legacy_jobs_without_mutating_temperature_status(tmp_path):
    common = {'enabled': True, 'active_model': {'version': '4', 'run_id': 'common-4'},
              'running_job': {'job_id': 'common-job', 'status': 'running'}, 'queued_jobs': [],
              'latest_job': {'job_id': 'common-job', 'status': 'running'},
              'jobs': [{'job_id': 'common-job', 'status': 'running'}],
              'server_time': '2026-10-01T07:00:00+00:00'}

    class Monitor:
        def retraining_status(self):
            return common

        def close(self):
            pass

    app = create_app(tmp_path, model=ForecastFixture(), model_factory=lambda runtime: ForecastFixture(),
                     temperature_monitor=Monitor(), dashboard_dataset='')
    with TestClient(app) as client:
        manager = app.state.workspaces
        manager.primary.store.ingest(rows(24), 'fixture observations')
        manager.primary.training = training()
        other, _ = manager.ensure('SECOND-HIVE')
        other.training = training('completed', snapshot_id='latest', version='7',
                                  completed_at='2026-10-01T06:00:00+00:00')
        expected = deepcopy(common)
        responses = [client.get('/monitoring/retraining', params={'workspace_id': choice})
                     for choice in ('BEE-01', 'SECOND-HIVE', 'UNKNOWN-HIVE')]
        assert all(response.status_code == 200 for response in responses)
        assert all(response.headers['cache-control'] == 'no-store' for response in responses)
        records = [response.json() for response in responses]
        assert records[0] == records[1] == records[2]
        assert {key: records[0][key] for key in common} == expected
        assert [job['workspace_id'] for job in records[0]['legacy_jobs']] == ['SECOND-HIVE', 'BEE-01']
        assert common == expected


def test_disabled_temperature_monitor_still_exposes_loaded_manual_training(tmp_path):
    app = create_app(tmp_path, model=ForecastFixture(), model_factory=lambda runtime: ForecastFixture(),
                     dashboard_dataset='')
    with TestClient(app) as client:
        app.state.service.store.ingest(rows(24), 'fixture observations')
        app.state.service.training = training()
        result = client.get('/monitoring/retraining').json()
        assert result['enabled'] is False
        assert result['active_model'] is None
        assert result['running_job'] is None and result['jobs'] == []
        assert result['legacy_jobs'][0]['job_id'] == 'legacy:BEE-01:snapshot-1'
