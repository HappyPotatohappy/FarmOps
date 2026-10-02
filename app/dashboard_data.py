"""Seed observed dashboard hives while retaining all pre-existing workspaces."""
from copy import deepcopy
import csv
import json
from threading import RLock

from .contracts import validate_series
from .store import atomic_json
from .workspaces import workspace_directory


class DashboardData:
    def __init__(self, manager, cases):
        self.manager = manager
        self.cases = cases
        self.path = manager.root_runtime/'dashboard_data.json'
        self._lock = RLock()
        catalog = cases.catalog()
        self._cases = catalog['cases']
        if not self._cases:
            raise ValueError('Observed dashboard requires at least one case')
        self._case_by_hive = {case['hive_id']:case for case in self._cases}
        by_id = {case['id']:case for case in self._cases}
        if len(self._case_by_hive)!=len(self._cases) or len(by_id)!=len(self._cases):
            raise ValueError('Observed cases require unique identities')
        for hive_id in self._case_by_hive:
            workspace_directory(manager.root_runtime, hive_id)
        default = catalog.get('default_case_id') or self._cases[0]['id']
        if default not in by_id:
            raise ValueError('Observed dashboard default case is unavailable')
        self.default_workspace_id = by_id[default]['hive_id']
        saved = []
        if self.path.exists():
            state = json.loads(self.path.read_text())
            if (not isinstance(state, dict) or state.get('schema_version')!=1
                    or not isinstance(state.get('workspace_ids'), list)):
                raise ValueError('Invalid dashboard workspace visibility state')
            saved = state['workspace_ids']
            for hive_id in saved:
                workspace_directory(manager.root_runtime, hive_id)
        self._visible = list(dict.fromkeys([*self._case_by_hive, *saved]))

    def _save(self, workspace_ids):
        atomic_json(self.path, {'schema_version':1, 'workspace_ids':workspace_ids})

    def seed(self):
        """Validate every source first; duplicate ingestion preserves later rows."""
        prepared = []
        for case in self._cases:
            with self.cases.file(case['id']).open(newline='') as handle:
                observations = validate_series(list(csv.DictReader(handle)), minimum=168)
            if (observations[0]['hive_id']!=case['hive_id'] or len(observations)!=case['rows']
                    or observations[0]['timestamp']!=case['start']
                    or observations[-1]['timestamp']!=case['end']):
                raise ValueError('Case observations do not match their manifest identity')
            prepared.append((case, observations))
        result = {}
        with self._lock:
            for case, observations in prepared:
                service, _ = self.manager.ensure(case['hive_id'])
                result[case['hive_id']] = service.store.ingest(observations, f"공개 관측: {case['label']}")
                service.store.set_meta('dashboard_name', case['label'])
                service.store.set_meta('dashboard_kind', 'public_real')
            self._save(self._visible)
        return result

    def catalog(self, items):
        """Show seeded cases and explicitly registered uploads in stable order."""
        by_id = {item['workspace_id']:item for item in items}
        result = []
        with self._lock:
            for hive_id in self._visible:
                if hive_id not in by_id:
                    continue
                item = deepcopy(by_id[hive_id])
                if hive_id in self._case_by_hive:
                    item.update(name=self._case_by_hive[hive_id]['label'], kind='public_real')
                result.append(item)
        return result

    def register(self, hive_id):
        workspace_directory(self.manager.root_runtime, hive_id)
        with self._lock:
            if hive_id not in self._visible:
                visible = [*self._visible, hive_id]
                self._save(visible)
                self._visible = visible
