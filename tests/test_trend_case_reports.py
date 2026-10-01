"""Selected observation cases still use real snapshot-bound prediction paths."""
import csv
import hashlib
import io
import json
from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest

from test_horizon_api import HorizonFixture


def fixture_cases(tmp_path):
    folder = tmp_path / 'cases'
    folder.mkdir()
    path = folder / 'case-a.csv'
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    rows = [dict(timestamp=(start+timedelta(hours=i)).isoformat(), hive_id='CASE-A',
                 weight_kg=50+i*.01, temperature_c=25, event='normal') for i in range(240)]
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    item = dict(id='case-a', label='관측 사례', hive_id='CASE-A', source_hive_id='apis-source',
                start=rows[0]['timestamp'], end=rows[-1]['timestamp'], rows=len(rows),
                file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                source_url='https://example.org/data')
    manifest = {'default_case_id':'case-a', 'cases':[item]}
    (folder/'manifest.json').write_text(json.dumps(manifest))
    return folder, rows, manifest


def reports(tmp_path):
    from app.trend_cases import TrendCaseReports
    folder, rows, manifest = fixture_cases(tmp_path)
    provider = HorizonFixture()
    return TrendCaseReports(provider, folder), provider, folder, rows, manifest


def test_case_report_uses_exact_observations_and_current_model_without_ingestion(tmp_path):
    service, provider, folder, rows, _ = reports(tmp_path)
    before = {p.name:p.read_bytes() for p in folder.iterdir()}
    result = service.report('case-a')
    assert provider.calls[0] == (rows,168,'2')
    assert result['observations'] == rows
    assert result['forecast']['as_of'] == rows[-1]['timestamp']
    assert result['forecast']['trajectory'][0]['timestamp'] == '2025-01-11T00:00:00+00:00'
    assert result['model']['version'] == '2'
    assert result['forecast_id'] == result['forecast']['forecast_id']
    assert result['case']['download_url'] == '/data/trend-cases/case-a.csv'
    assert {p.name:p.read_bytes() for p in folder.iterdir()} == before


def test_case_cache_follows_active_model_and_returns_independent_objects(tmp_path):
    service, provider, _, _, _ = reports(tmp_path)
    provider.active_version = lambda:'1'
    first = service.report('case-a')
    first['forecast']['trajectory'].clear()
    assert len(service.report('case-a')['forecast']['trajectory']) == 168
    assert len(provider.calls) == 1
    provider.active_version = lambda:'2'
    newer = service.report('case-a')
    assert newer['model']['version'] == '2'
    assert newer['forecast_id'] != first['forecast_id']
    assert len(provider.calls) == 2


def test_unknown_case_and_tampered_observations_are_not_served(tmp_path):
    service, _, folder, _, _ = reports(tmp_path)
    with pytest.raises(KeyError): service.report('../case-a')
    (folder/'case-a.csv').write_text('bad')
    with pytest.raises(ValueError, match='checksum'): service.report('case-a')


def test_case_file_cannot_escape_manifest_directory(tmp_path):
    service, _, folder, _, manifest = reports(tmp_path)
    manifest['cases'][0]['file'] = '../outside.csv'
    (folder/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='directory'): service.file('case-a')


def test_csv_export_contains_prediction_values_and_only_recommended_hours(tmp_path):
    service, _, _, _, _ = reports(tmp_path)
    result = service.report('case-a')
    exported = list(csv.DictReader(io.StringIO(service.forecast_csv('case-a'))))
    assert len(exported) == 168
    assert [float(row['predicted_weight_kg']) for row in exported] == [
        row['weight_kg'] for row in result['forecast']['trajectory']]
    selected = [row for row in exported if row['recommended_harvest']=='true']
    assert 1 <= len(selected) <= 72
    assert selected[0]['timestamp'] == result['candidate_window']['start']
    assert selected[-1]['timestamp'] == result['candidate_window']['end']


def test_inference_failure_or_wrong_model_is_not_replaced_with_observed_curve(tmp_path):
    service, provider, _, _, _ = reports(tmp_path)
    provider.fail = True
    with pytest.raises(RuntimeError, match='broken'): service.report('case-a')
    provider.fail = False; provider.wrong = True
    with pytest.raises(RuntimeError, match='different'): service.report('case-a')


def test_case_http_routes_and_downloads_are_read_only(tmp_path):
    from fastapi.testclient import TestClient
    from app.main import create_app
    from test_api import ForecastFixture
    service, provider, _, rows, _ = reports(tmp_path)
    app = create_app(tmp_path/'runtime',model=ForecastFixture(),horizon_model=provider)
    with TestClient(app) as client:
        app.state.trend_cases = service
        before = client.get('/workspaces').json()
        catalog = client.get('/data/trend-cases')
        assert catalog.status_code == 200
        assert catalog.json()['cases'][0]['id']=='case-a'
        assert client.get('/data/trend-cases/case-a.csv').status_code==200
        report = client.get('/analysis/trend-cases/case-a')
        assert report.status_code==200
        assert report.headers['cache-control']=='no-store'
        assert report.json()['observations']==rows
        exported = client.get('/analysis/trend-cases/case-a/forecast.csv')
        assert exported.status_code==200
        assert len(list(csv.DictReader(io.StringIO(exported.text))))==168
        assert client.get('/analysis/trend-cases/missing').status_code==404
        assert client.get('/data/trend-cases/missing.csv').status_code==404
        assert client.get('/workspaces').json()==before
