"""Read-only observation examples with forecasts from the current shared model."""
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime, timezone
import csv
import hashlib
import io
import json
from pathlib import Path
from threading import RLock

from .contracts import validate_series
from .harvest import recommend_harvest
from .horizon_service import HorizonReports, REGISTRY_ID
from .store import digest, utcnow

DATA_ROOT = Path(__file__).resolve().parents[1] / 'data' / 'trend_cases'


class TrendCaseReports:
    def __init__(self, provider, root=DATA_ROOT):
        self.provider = provider
        self.root = Path(root).resolve()
        self._lock = RLock()
        self._cache = OrderedDict()

    def _manifest(self):
        return json.loads((self.root/'manifest.json').read_text())

    def _case(self, case_id):
        case = next((c for c in self._manifest()['cases'] if c['id']==case_id), None)
        if case is None:
            raise KeyError(case_id)
        return case

    @staticmethod
    def _summary(case):
        public = {k:deepcopy(case[k]) for k in ('id','label','hive_id','source_hive_id',
            'start','end','rows','source_url','source_label','license','weight_min_kg','weight_max_kg') if k in case}
        public['download_url'] = f"/data/trend-cases/{case['id']}.csv"
        public['forecast_download_url'] = f"/analysis/trend-cases/{case['id']}/forecast.csv"
        return public

    def catalog(self):
        manifest = self._manifest()
        return {'default_case_id':manifest.get('default_case_id'),
                'cases':[self._summary(c) for c in manifest['cases']],
                'download_url':'/data/trend-cases.zip'}

    def file(self, case_id):
        case = self._case(case_id)
        path = (self.root/case['file']).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('Case file escapes its data directory')
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != case['sha256']:
            raise ValueError('Case file checksum does not match its manifest')
        return path

    def report(self, case_id):
        case = self._case(case_id)
        path = self.file(case_id)
        with path.open(newline='') as handle:
            observations = validate_series(list(csv.DictReader(handle)),minimum=168)
        if (len(observations)!=case['rows'] or observations[0]['hive_id']!=case['hive_id']
                or observations[0]['timestamp']!=case['start'] or observations[-1]['timestamp']!=case['end']):
            raise ValueError('Case observations do not match their manifest identity')
        catalog = HorizonReports(self.provider).catalog()
        version = catalog['active_version']
        model = next((item for item in catalog['versions'] if str(item['version'])==version), None)
        if model is None:
            raise RuntimeError('Shared forecasting model is unavailable')
        snapshot = digest(observations)
        fingerprint = digest({'case_id':case_id,'snapshot_id':snapshot,'model':model,'horizon_hours':168})
        with self._lock:
            if fingerprint in self._cache:
                self._cache.move_to_end(fingerprint)
                return deepcopy(self._cache[fingerprint])
            forecast = deepcopy(self.provider.predict(observations,168,version=version))
            as_of = observations[-1]['timestamp']
            HorizonReports._validate(forecast,version,model,as_of,168)
            forecast.update(workspace_id=case['hive_id'],hive_id=case['hive_id'],snapshot_id=snapshot,
                forecast_id=fingerprint,as_of=as_of,generated_at=utcnow(),horizon_hours=168,
                requested_horizon_hours=168,time_basis='recorded_timestamp',
                historical=datetime.fromisoformat(as_of)<datetime.now(timezone.utc))
            source_hive = case['source_hive_id']
            species = 'meliponini' if 'meliponini' in source_hive else 'apis'
            report = recommend_harvest(forecast,observations,species)
            report.update(case=self._summary(case),observations=observations,forecast=forecast)
            if forecast['status']=='ok':
                self._cache[fingerprint] = deepcopy(report)
                while len(self._cache)>12:
                    self._cache.popitem(last=False)
            return report

    def forecast_csv(self, case_id):
        report = self.report(case_id)
        forecast = report['forecast']
        if forecast['status']!='ok':
            raise RuntimeError('Case forecast is unavailable')
        window = report['candidate_window']
        output = io.StringIO(newline='')
        writer = csv.DictWriter(output,fieldnames=['timestamp','hive_id','predicted_weight_kg',
            'lower_kg','upper_kg','recommended_harvest','model_version'])
        writer.writeheader()
        for point in forecast['trajectory']:
            stamp = point['timestamp']
            recommended = bool(window and window['start']<=stamp<=window['end'])
            writer.writerow(dict(timestamp=stamp,hive_id=report['hive_id'],
                predicted_weight_kg=point['weight_kg'],lower_kg=point.get('lower_kg'),
                upper_kg=point.get('upper_kg'),recommended_harvest=str(recommended).lower(),
                model_version=report['model']['version']))
        return output.getvalue()
