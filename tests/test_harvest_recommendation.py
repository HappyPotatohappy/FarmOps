"""The product recommendation has one shared model and one automatic window."""
from test_horizon_api import horizon_client
from test_store import rows


def test_automatic_recommendation_ignores_old_model_and_period_selections(horizon_client):
    client, provider = horizon_client
    client.app.state.service.store.ingest(rows(80), 'fixture')
    response = client.get('/harvest/recommendation', params={
        'model_version': '1', 'horizon_hours': 24})
    assert response.status_code == 200, response.text
    result = response.json()
    assert response.headers['cache-control'] == 'no-store'
    assert result['selection'] == {
        'mode': 'automatic', 'model_version': '2', 'horizon_hours': 168}
    assert result['model']['version'] == result['forecast']['model']['version'] == '2'
    assert result['horizon_hours'] == result['requested_horizon_hours'] == 168
    assert len(result['forecast']['trajectory']) == 168
    assert result['candidate_window'] is not None
    assert result['forecast']['forecast_id'] == result['forecast_id']
    assert result['planning'] and 'preferences' in result
    # The provider still defaults to v1; product policy must explicitly select v2.
    assert provider.calls[-1][1:] == (168, '2')


def test_automatic_recommendation_keeps_the_common_model_for_other_hive(horizon_client):
    client, _ = horizon_client
    client.app.state.service.store.ingest(rows(50), 'primary')
    other, _ = client.app.state.workspaces.ensure('OTHER')
    sample = rows(50)
    for row in sample:
        row['hive_id'] = 'OTHER'
        row['weight_kg'] += 10
    other.store.ingest(sample, 'other')
    primary = client.get('/harvest/recommendation').json()
    result = client.get('/harvest/recommendation', params={'workspace_id': 'OTHER'}).json()
    assert result['workspace_id'] == result['hive_id'] == 'OTHER'
    assert result['model'] == primary['model']
    assert result['forecast_id'] != primary['forecast_id']
    assert result['forecast']['trajectory'][0]['weight_kg'] > 60
    assert client.get('/harvest/recommendation', params={'workspace_id': 'MISSING'}).status_code == 404


def test_automatic_recommendation_does_not_invent_forecast_for_short_input(horizon_client):
    client, _ = horizon_client
    client.app.state.service.store.ingest(rows(12), 'short')
    response = client.get('/harvest/recommendation')
    assert response.status_code == 200, response.text
    result = response.json()
    assert result['candidate_window'] is None
    assert result['forecast']['status'] == 'insufficient_history'
    assert result['forecast']['trajectory'] == []
    assert result['selection']['model_version'] == '2'


def test_automatic_recommendation_exposes_model_failure_without_substitution(horizon_client):
    client, provider = horizon_client
    client.app.state.service.store.ingest(rows(50), 'fixture')
    provider.fail = True
    response = client.get('/harvest/recommendation')
    assert response.status_code == 503
    assert 'artifact' in response.text
