"""Real CSV and real MLflow model: new observations change inference context."""
import csv
import io
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import create_app
from app.ml import ModelService
from test_workflow_api import await_job, csv_file


def test_real_csv_incremental_upload_updates_inference_and_duplicate_is_unchanged(tmp_path):
    data = list(csv.DictReader(io.StringIO(
        (Path(__file__).resolve().parents[2] / 'data/real_hive.csv').read_text())))
    assert len(data) == 724
    scope = {'workspace_id': 'ufc_apis_2'}
    with TestClient(create_app(tmp_path, model=ModelService(tmp_path))) as client:
        def upload(part):
            response = client.post('/imports/preview', files=csv_file(part))
            assert response.status_code == 200
            preview = response.json()
            assert preview['can_commit'], preview
            response = client.post('/imports/commit', json={'preview_id': preview['preview_id']})
            assert response.status_code == 202
            job = await_job(client, response.json()['job_id'], attempts=3000)
            assert job['status'] == 'completed', job
            return preview, job['result']

        def report():
            response = client.get('/forecast/report', params=scope)
            assert response.status_code == 200, response.text
            return response.json()

        _, initial = upload(data[:672])
        assert initial['ingest']['inserted'] == 672
        before = report()
        assert before['next_forecast']['target_timestamp'] == data[672]['timestamp']
        assert before['metrics']['post_cutoff_clean']['count'] == 0
        registered_before = client.get('/models', params=scope).json()

        _, appended = upload(data[672:698])
        assert appended['ingest']['inserted'] == 26
        middle = report()
        assert middle['snapshot_id'] != before['snapshot_id']
        assert middle['next_forecast']['target_timestamp'] == data[698]['timestamp']
        assert middle['next_forecast']['input_end'] == data[697]['timestamp']
        assert middle['next_forecast']['predicted_weight_kg'] != before['next_forecast']['predicted_weight_kg']

        _, final = upload(data[698:])
        assert final['ingest']['inserted'] == 26
        after = report()
        assert after['snapshot_id'] != middle['snapshot_id']
        assert after['next_forecast']['input_end'] == data[-1]['timestamp']
        assert after['next_forecast']['predicted_weight_kg'] != middle['next_forecast']['predicted_weight_kg']
        assert after['model']['version'] == before['model']['version'] == '1'
        assert after['model']['run_id'] == before['model']['run_id']
        assert client.get('/models', params=scope).json() == registered_before

        preview, duplicate = upload(data)
        assert preview['insert_count'] == 0 and preview['duplicate_count'] == 724
        assert duplicate['ingest']['inserted'] == 0
        repeated = report()
        for key in ('snapshot_id', 'rows', 'next_forecast', 'metrics', 'model'):
            assert repeated[key] == after[key], key
        history = client.get('/predictions', params=scope).json()
        report()
        assert client.get('/predictions', params=scope).json() == history
