"""Primary retraining must update the ensemble served by the dashboard."""
from fastapi.testclient import TestClient
import pytest

from app.main import create_app
from test_api import ForecastFixture
from test_store import rows


class EnsembleMonitor:
    def __init__(self, error=None):
        self.requests = []
        self.error = error

    def observe(self, service, workspace_id):
        return {'enabled': True}

    def request_training(self, service, workspace_id):
        self.requests.append((workspace_id, service.store.data_status()['rows']))
        if self.error:
            raise self.error
        return {'status': 'queued', 'workspace_id': workspace_id,
                'job': {'job_id': 'actual-ensemble', 'status': 'queued',
                        'registry_id': 'BeeOPS_Horizon_Weight'}}

    def close(self):
        pass


def test_primary_retrain_uses_actual_ensemble_and_resolves_default_hive(tmp_path, monkeypatch):
    monitor = EnsembleMonitor()
    with TestClient(create_app(tmp_path, model=ForecastFixture(), temperature_monitor=monitor)) as client:
        assert client.post('/observations', json={'observations': rows(24)}).status_code == 200
        def forbidden_legacy(*args, **kwargs):
            pytest.fail('Primary control must not run the separate legacy LSTM')
        monkeypatch.setattr(client.app.state.service, 'request_training', forbidden_legacy)
        response = client.post('/models/retrain', json={})
        assert response.status_code == 202, response.text
        assert monitor.requests == [('BEE-01', 24)]
        assert response.json()['job']['registry_id'] == 'BeeOPS_Horizon_Weight'
        assert response.json()['job']['status'] == 'queued'
        assert client.post('/models/retrain?workspace_id=missing', json={}).status_code == 404
        assert len(monitor.requests) == 1


def test_missing_ensemble_trainer_never_falls_back_to_legacy(tmp_path, monkeypatch):
    with TestClient(create_app(tmp_path, model=ForecastFixture())) as client:
        monkeypatch.setattr(client.app.state.service, 'request_training',
                            lambda: pytest.fail('No legacy fallback'))
        response = client.post('/models/retrain', json={})
        assert response.status_code == 503
        assert '앙상블' in response.json()['detail']


@pytest.mark.parametrize('error,status', [(ValueError('Need new observed labels'), 422),
                                        (RuntimeError('No active ensemble'), 503)])
def test_training_errors_do_not_claim_job_was_completed(tmp_path, error, status):
    with TestClient(create_app(tmp_path, model=ForecastFixture(),
                              temperature_monitor=EnsembleMonitor(error))) as client:
        response = client.post('/models/retrain', json={})
        assert response.status_code == status
        assert response.json() == {'detail': str(error)}
