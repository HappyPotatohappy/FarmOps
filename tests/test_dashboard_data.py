"""Observed dashboard defaults preserve every existing observation ledger."""
from copy import deepcopy
import csv
import hashlib
import json

import pytest

from app.service import Service
from app.trend_cases import TrendCaseReports
from app.workspaces import WorkspaceManager
from test_store import rows


class ReadOnlyModel:
    def __init__(self, runtime=None):
        pass

    def status(self):
        return {'ready': True, 'version': '2'}

    def versions(self):
        return []

    def bootstrap(self, *args, **kwargs):
        raise AssertionError('Seeding must not train models')

    def predict(self, *args, **kwargs):
        raise AssertionError('Seeding must only ingest observations')


def case_rows(hive_id, count=168, start=0):
    values = rows(count, start=start)
    for row in values:
        row['hive_id'] = hive_id
    return values


@pytest.fixture
def observed_dashboard(tmp_path):
    folder = tmp_path/'cases'
    folder.mkdir()
    entries = []
    for index in range(3):
        case_id, hive_id = f'case-{index}', f'OBSERVED-{index}'
        observations = case_rows(hive_id)
        path = folder/f'{case_id}.csv'
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(observations[0]))
            writer.writeheader()
            writer.writerows(observations)
        entries.append(dict(id=case_id, hive_id=hive_id, label=f'실측 벌통 {index}',
                            file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            rows=len(observations), start=observations[0]['timestamp'],
                            end=observations[-1]['timestamp']))
    (folder/'manifest.json').write_text(json.dumps({'default_case_id':'case-1', 'cases':entries}))
    runtime = tmp_path/'runtime'
    manager = WorkspaceManager(runtime, Service(runtime, ReadOnlyModel()), model_factory=ReadOnlyModel)
    try:
        yield manager, TrendCaseReports(None, folder), entries
    finally:
        manager.close()


def test_seed_exposes_only_three_observed_hives_without_deleting_legacy_data(observed_dashboard):
    from app.dashboard_data import DashboardData
    manager, cases, _ = observed_dashboard
    manager.primary.store.ingest(rows(24), 'legacy')
    legacy = manager.primary.store.observations()
    extra, _ = manager.ensure('OLD-UPLOAD')
    extra.store.ingest(case_rows('OLD-UPLOAD', 24), 'old upload')
    dashboard = DashboardData(manager, cases)
    dashboard.seed()
    items = manager.catalog()
    original_items = deepcopy(items)
    visible = dashboard.catalog(items)
    assert [item['workspace_id'] for item in visible] == ['OBSERVED-0', 'OBSERVED-1', 'OBSERVED-2']
    assert [item['name'] for item in visible] == ['실측 벌통 0', '실측 벌통 1', '실측 벌통 2']
    assert all(item['kind']=='public_real' and item['rows']==168 for item in visible)
    assert dashboard.default_workspace_id == 'OBSERVED-1'
    assert manager.primary.store.observations() == legacy
    assert extra.store.data_status()['rows'] == 24
    assert items == original_items
    for hive in ('OBSERVED-0', 'OBSERVED-1', 'OBSERVED-2'):
        assert manager.get(hive).store.observations() == case_rows(hive)


def test_restart_seeding_preserves_followups_and_explicit_upload_visibility(observed_dashboard):
    from app.dashboard_data import DashboardData
    manager, cases, _ = observed_dashboard
    dashboard = DashboardData(manager, cases)
    dashboard.seed()
    first = manager.get('OBSERVED-0').store
    first.ingest(case_rows('OBSERVED-0', 2, 168), 'followup')
    uploaded, _ = manager.ensure('NEW-UPLOAD')
    uploaded.store.ingest(case_rows('NEW-UPLOAD', 24), 'user CSV')
    assert 'NEW-UPLOAD' not in {item['workspace_id'] for item in dashboard.catalog(manager.catalog())}
    dashboard.register('NEW-UPLOAD')
    dashboard.register('NEW-UPLOAD')
    restarted = DashboardData(manager, cases)
    restarted.seed()
    restarted.seed()
    assert first.observations() == case_rows('OBSERVED-0', 170)
    assert [item['workspace_id'] for item in restarted.catalog(manager.catalog())] == [
        'OBSERVED-0', 'OBSERVED-1', 'OBSERVED-2', 'NEW-UPLOAD']
    assert (manager.root_runtime/'dashboard_data.json').is_file()


def test_seed_conflicts_preserve_existing_values(observed_dashboard):
    from app.dashboard_data import DashboardData
    manager, cases, _ = observed_dashboard
    existing, _ = manager.ensure('OBSERVED-0')
    altered = case_rows('OBSERVED-0')
    altered[20]['weight_kg'] += .5
    existing.store.ingest(altered, 'prior observations')
    before = existing.store.observations()
    with pytest.raises(ValueError, match='Conflicting'):
        DashboardData(manager, cases).seed()
    assert existing.store.observations() == before


def test_seed_checks_all_source_hashes_before_ingesting_any_case(observed_dashboard):
    from app.dashboard_data import DashboardData
    manager, cases, _ = observed_dashboard
    (cases.root/'case-2.csv').write_text('tampered')
    with pytest.raises(ValueError, match='checksum'):
        DashboardData(manager, cases).seed()
    assert manager.primary.store.data_status()['rows'] == 0
    assert len(manager.catalog()) == 1


@pytest.mark.parametrize('field,value', [('hive_id','WRONG-HIVE'), ('rows',169),
                                        ('start','2024-01-01T00:00:00+00:00')])
def test_seed_rejects_case_metadata_that_does_not_match_observations(observed_dashboard,field,value):
    from app.dashboard_data import DashboardData
    manager, cases, _ = observed_dashboard
    path = cases.root/'manifest.json'
    manifest = json.loads(path.read_text())
    manifest['cases'][0][field] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='identity'):
        DashboardData(manager, cases).seed()
    assert manager.primary.store.data_status()['rows'] == 0


@pytest.mark.parametrize('hive_id', ['../escape', '', 'bad hive', None])
def test_invalid_registration_does_not_change_visibility(observed_dashboard,hive_id):
    from app.dashboard_data import DashboardData
    manager, cases, _ = observed_dashboard
    dashboard = DashboardData(manager, cases)
    dashboard.seed()
    state = (manager.root_runtime/'dashboard_data.json').read_bytes()
    with pytest.raises(ValueError):
        dashboard.register(hive_id)
    assert (manager.root_runtime/'dashboard_data.json').read_bytes() == state
