"""The browser's complete CSV workflow against actual HTTP and SQLite boundaries."""
import csv
import io
import time

from fastapi.testclient import TestClient
from app.main import create_app
from test_api import ForecastFixture
from test_store import rows


def csv_file(data):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(data[0]))
    writer.writeheader()
    writer.writerows(data)
    return {'file': ('observations.csv', stream.getvalue(), 'text/csv')}


def await_job(client, job_id, attempts=100):
    for _ in range(attempts):
        job = client.get('/imports/jobs/' + job_id).json()
        if job['status'] not in ('queued', 'running'):
            return job
        time.sleep(.02)
    raise AssertionError('Import job did not finish')


def test_upload_preview_commit_results_and_restart(tmp_path):
    with TestClient(create_app(tmp_path, model=ForecastFixture(), model_factory=lambda path: ForecastFixture())) as client:
        preview = client.post('/imports/preview', files=csv_file(rows(30))).json()
        assert preview['can_commit'] is True
        assert preview['insert_count'] == 30
        assert client.get('/data/status').json()['rows'] == 0
        response = client.post('/imports/commit', json={'preview_id': preview['preview_id']})
        assert response.status_code == 202
        job = await_job(client, response.json()['job_id'])
        assert job['status'] == 'completed', job
        assert job['result']['ingest']['inserted'] == 30
        assert job['result']['forecast']['prediction']['actual_weight_kg'] is None
        workspace = preview['workspace_id']
        scope = {'workspace_id': workspace}
        assert client.get('/data/status', params=scope).json()['rows'] == 30
        page = client.get('/data/rows', params={**scope, 'page_size': 10}).json()
        assert page['total'] == 30 and len(page['items']) == 10
        assert page['items'][0]['timestamp'] > page['items'][-1]['timestamp']
        exported = client.get('/data/export', params=scope)
        assert exported.status_code == 200
        assert len(list(csv.DictReader(io.StringIO(exported.text)))) == 30
        duplicate = client.post('/imports/preview', files=csv_file(rows(30))).json()
        assert duplicate['insert_count'] == 0 and duplicate['duplicate_count'] == 30
        duplicate_job = client.post('/imports/commit', json={'preview_id': duplicate['preview_id']}).json()
        assert await_job(client, duplicate_job['job_id'])['result']['ingest']['inserted'] == 0
        repeated = client.post('/imports/commit', json={'preview_id': preview['preview_id']}).json()
        assert repeated['job_id'] == job['job_id']
    with TestClient(create_app(tmp_path, model=ForecastFixture(), model_factory=lambda path: ForecastFixture())) as client:
        assert client.get('/imports/jobs/' + job['job_id']).json()['status'] == 'completed'
        assert client.get('/data/status', params=scope).json()['rows'] == 30


def test_cross_hive_workspaces_are_isolated_and_catalog_visible(tmp_path):
    with TestClient(create_app(tmp_path, model=ForecastFixture(), model_factory=lambda path: ForecastFixture())) as client:
        client.post('/observations', json={'observations': rows(24)})
        other = rows(25)
        for row in other:
            row['hive_id'] = 'ANOTHER'
        preview = client.post('/imports/preview', files=csv_file(other)).json()
        job = client.post('/imports/commit', json={'preview_id': preview['preview_id']}).json()
        assert await_job(client, job['job_id'])['status'] == 'completed'
        catalog = client.get('/workspaces').json()
        assert {x['hive_id'] for x in catalog['workspaces']} == {'BEE-01', 'ANOTHER'}
        assert client.get('/data/status').json()['hive_id'] == 'BEE-01'
        assert client.get('/data/status', params={'workspace_id': 'ANOTHER'}).json()['rows'] == 25
        assert {x['workspace_id'] for x in client.get('/models/catalog').json()} == {'BEE-01', 'ANOTHER'}
        assert client.get('/health', params={'workspace_id': 'missing'}).status_code == 404


def test_invalid_preview_and_oversize_never_mutate(tmp_path):
    with TestClient(create_app(tmp_path, model=ForecastFixture())) as client:
        preview = client.post('/imports/preview', files={'file': ('bad.csv', 'timestamp,hive_id\ninvalid,BEE\n', 'text/csv')}).json()
        assert preview['can_commit'] is False and preview['errors']
        assert client.post('/imports/commit', json={'preview_id': preview['preview_id']}).status_code in (409, 422)
        assert client.get('/data/status').json()['rows'] == 0
        assert client.post('/imports/preview', files={'file': ('big.csv', b'x' * (5 * 1024 * 1024 + 1))}).status_code == 413
        assert client.get('/data/sample.csv').status_code == 200


def test_public_csv_creates_a_real_model_and_forecast_from_empty_runtime(tmp_path):
    from pathlib import Path
    from app.ml import ModelService
    source = Path(__file__).resolve().parents[2] / 'data' / 'real_hive.csv'
    with TestClient(create_app(tmp_path, model=ModelService(tmp_path))) as client:
        preview = client.post('/imports/preview', files={'file': ('real_hive.csv', source.read_bytes(), 'text/csv')}).json()
        assert preview['can_commit'] and preview['insert_count'] == 724
        response = client.post('/imports/commit', json={'preview_id': preview['preview_id']})
        assert response.status_code == 202
        job = await_job(client, response.json()['job_id'], attempts=3000)
        assert job['status'] == 'completed', job
        assert job['result']['ingest']['inserted'] == 724
        assert job['result']['forecast']['prediction']['model_version'] == '1'
        scope = {'workspace_id': 'ufc_apis_2'}
        health = client.get('/health', params=scope).json()
        assert health['model_loaded'] and health['quality']['status'] == 'insufficient_data'
        versions = client.get('/models', params=scope).json()
        assert {v['version'] for v in versions} == {'1','2'}
        assert job['result']['model_versions']['versions'] == ['1','2']
        initial = next(v for v in versions if v['version']=='1')
        second = next(v for v in versions if v['version']=='2')
        assert initial['champion'] and not second['champion']
        assert second['parent_version']=='1'
        assert int(second['metrics']['selected_epoch'])>=1
        cutoff = initial['params']['test_end']
        predictions = client.get('/predictions', params=scope).json()
        assert predictions and all(p['target_timestamp'] > cutoff for p in predictions)
        payload=client.get('/demo/payload',params=scope).json()
        outputs=[]
        for version in ('1','2'):
            prediction=client.post('/predict',params={**scope,'model_version':version},json=payload)
            assert prediction.status_code==200,prediction.text
            assert prediction.json()['model_version']==version
            outputs.append(prediction.json()['predicted_weight_kg'])
        assert abs(outputs[1]-outputs[0])>1e-8
        duplicate=client.post('/imports/preview',files={'file':('real_hive.csv',source.read_bytes(),'text/csv')}).json()
        repeated=client.post('/imports/commit',json={'preview_id':duplicate['preview_id']}).json()
        assert await_job(client,repeated['job_id'],attempts=3000)['status']=='completed'
        assert len(client.get('/models',params=scope).json())==2
