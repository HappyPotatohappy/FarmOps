"""Observed inputs are the actual dashboard workspaces, not a gallery fixture."""
import csv
import io

from fastapi.testclient import TestClient

from app.main import create_app
from app.store import Store
from test_api import ForecastFixture
from test_horizon_api import HorizonFixture
from test_store import rows


def test_observed_profile_seeds_actual_ledgers_and_all_default_routes(tmp_path):
    old = rows(24)
    Store(tmp_path).ingest(old,'old data')
    app = create_app(tmp_path,model=ForecastFixture(),model_factory=lambda _:ForecastFixture(),
                     horizon_model=HorizonFixture(),dashboard_dataset='observed')
    with TestClient(app) as client:
        catalog = client.get('/workspaces').json()
        assert len(catalog['workspaces'])==3
        default = catalog['default_workspace_id']
        assert default=='trend_ufc_apis2_dec_jan'
        assert [w['rows'] for w in catalog['workspaces']]==[609,456,477]
        assert all(w['kind']=='public_real' for w in catalog['workspaces'])
        assert old[0]['hive_id'] not in [w['workspace_id'] for w in catalog['workspaces']]
        assert client.get('/health').json()['data']['hive_id']==default
        assert client.get('/config').json()['hive_id']==default
        assert client.get('/config').json()['runtime_type']=='public_real'
        observations = client.get('/observations?limit=720').json()
        assert len(observations)==609 and observations[-1]['hive_id']==default
        report = client.get('/harvest/recommendation').json()
        assert report['workspace_id']==report['hive_id']==default
        assert len(report['forecast']['trajectory'])==168
        assert len(client.get('/models/catalog').json())==3
        samples = client.get('/data/samples').json()
        assert len(samples['samples'])==3
        for sample in samples['samples']:
            actual = list(csv.DictReader(io.StringIO(client.get(sample['download_url']).text)))
            assert len(actual)==sample['rows']
            assert actual[0]['hive_id']==sample['hive_id']
        exported = list(csv.DictReader(io.StringIO(client.get('/data/export').text)))
        assert len(exported)==609 and exported[-1]['hive_id']==default
        # Prior data remains intact and accessible by its explicit identity.
        assert app.state.workspaces.primary.store.observations()==old


def test_observed_profile_seed_is_repeatable_without_duplicate_observations(tmp_path):
    for _ in range(2):
        app = create_app(tmp_path,model=ForecastFixture(),model_factory=lambda _:ForecastFixture(),
                         horizon_model=HorizonFixture(),dashboard_dataset='observed')
        with TestClient(app) as client:
            assert [w['rows'] for w in client.get('/workspaces').json()['workspaces']]==[609,456,477]
