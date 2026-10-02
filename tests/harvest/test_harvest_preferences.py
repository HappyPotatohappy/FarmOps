"""Field records change planning, never model output or the prediction ledger."""
from test_horizon_api import horizon_client, dump
from test_store import rows
import pytest


def test_defaults_are_unknown_and_read_does_not_write(horizon_client):
    client, _ = horizon_client
    store = client.app.state.service.store
    before = dump(store)
    response = client.get('/harvest/preferences')
    assert response.status_code == 200
    assert response.headers['cache-control'] == 'no-store'
    assert response.json()['updated_at'] is None
    assert all(v is None for v in response.json()['preferences'].values())
    assert dump(store) == before


def test_save_is_scoped_to_one_hive_and_persists(horizon_client):
    client, _ = horizon_client
    client.app.state.service.store.ingest(rows(80), 'fixture')
    client.app.state.workspaces.ensure('OTHER')
    payload = {'recorded_timezone':'Asia/Seoul', 'working_hours':{'start':'05:00','end':'09:00'},
               'honey_moisture_pct':17.2, 'capped_ratio_pct':83, 'reserves_confirmed':True,
               'measured_at':'2025-01-01T08:00:00+09:00'}
    saved = client.put('/harvest/preferences', json=payload)
    assert saved.status_code == 200, saved.text
    assert saved.json()['preferences']['honey_moisture_pct'] == 17.2
    assert saved.json()['updated_at']
    assert client.get('/harvest/preferences').json() == saved.json()
    other = client.get('/harvest/preferences', params={'workspace_id':'OTHER'}).json()
    assert other['workspace_id'] == 'OTHER'
    assert all(v is None for v in other['preferences'].values())
    assert client.get('/harvest/preferences', params={'workspace_id':'MISSING'}).status_code == 404


def test_field_conditions_do_not_block_or_recompute_forecast(horizon_client):
    client, provider = horizon_client
    store = client.app.state.service.store
    store.ingest(rows(80), 'fixture')
    first = client.get('/harvest/report').json()
    assert 'planning' in first
    assert first['planning']['work_windows'] == []
    saved = client.put('/harvest/preferences', json={
        'recorded_timezone':'Asia/Seoul', 'working_hours':{'start':'05:00','end':'09:00'},
        'honey_moisture_pct':23, 'capped_ratio_pct':10, 'reserves_confirmed':False})
    assert saved.status_code == 200
    after_save = dump(store)
    second = client.get('/harvest/report').json()
    assert second['candidate_window'] == first['candidate_window']
    assert second['status'] == first['status'] == 'inspection_window'
    assert second['forecast'] == first['forecast']
    assert second['forecast_id'] == first['forecast_id']
    assert second['planning']['work_windows']
    assert second['planning']['field_checks'] != first['planning']['field_checks']
    assert len(provider.calls) == 1
    assert dump(store) == after_save


@pytest.mark.parametrize('payload',[
    {'recorded_timezone':'Not/AZone'}, {'honey_moisture_pct':-1},
    {'capped_ratio_pct':101}, {'reserves_confirmed':'yes'},
    {'honey_moisture_pct':True}, {'unexpected':'value'},
    {'working_hours':{'start':'25:00','end':'09:00'}},
    {'working_hours':{'start':'09:00','end':'09:00'}},
    {'measured_at':'2025-01-01T08:00:00'},
])
def test_invalid_field_input_cannot_replace_previous_settings(horizon_client, payload):
    client, _ = horizon_client
    assert client.put('/harvest/preferences', json={'recorded_timezone':'Europe/Riga'}).status_code == 200
    before = client.get('/harvest/preferences').json()
    assert client.put('/harvest/preferences', json=payload).status_code == 422
    assert client.get('/harvest/preferences').json() == before


def test_empty_object_resets_all_fields(horizon_client):
    client, _ = horizon_client
    assert client.put('/harvest/preferences', json={'reserves_confirmed':True}).status_code == 200
    response = client.put('/harvest/preferences', json={})
    assert response.status_code == 200
    assert all(v is None for v in response.json()['preferences'].values())


def test_timezone_area_name_is_validation_error_not_500(horizon_client):
    client, _ = horizon_client
    client.app.state.service.store.ingest(rows(80), 'fixture')
    assert client.put('/harvest/preferences', json={'recorded_timezone':'Europe'}).status_code == 422
