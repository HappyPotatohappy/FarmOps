"""Snapshot-bound multi-day reports; the one-hour ledger remains independent."""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from functools import lru_cache
import csv
import json
import math
from pathlib import Path
from threading import RLock

from .store import digest, utcnow

REGISTRY_ID = 'BeeOPS_Horizon_Weight'
HORIZONS = (1, 24, 72, 168)
ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def _public_observations():
    """Known sample identity needs matching observations, not just a chosen hive ID."""
    result = {}
    for path in [ROOT / 'data/real_hive.csv', *(ROOT / 'data/samples').glob('*.csv')]:
        with path.open(newline='') as handle:
            for row in csv.DictReader(handle):
                if not {'hive_id', 'timestamp', 'weight_kg', 'temperature_c'}.issubset(row):
                    continue
                result.setdefault(row['hive_id'], {})[datetime.fromisoformat(row['timestamp'])] = (
                    float(row['weight_kg']), float(row['temperature_c']))
    return result


def species_for(hive_id, observations, context_hours=72):
    if hive_id == 'BEE-DEMO':
        return 'synthetic'
    species = {'ufc_apis_1': 'apis', 'ufc_apis_2': 'apis', 'vecauce_2021_3': 'apis',
               'ufc_meliponini_s3': 'meliponini'}.get(hive_id, 'unknown')
    if species == 'unknown':
        return species
    known = _public_observations().get(hive_id, {})
    required = max(72, context_hours)
    if len(observations) < required:
        return 'unknown'
    for row in observations[-required:]:
        reference = known.get(datetime.fromisoformat(row['timestamp']))
        if reference is None or not all(math.isclose(float(row[key]), value, rel_tol=0, abs_tol=1e-8)
                                       for key, value in zip(('weight_kg', 'temperature_c'), reference)):
            return 'unknown'
    return species


class HorizonReports:
    def __init__(self, provider):
        self.provider = provider
        self._lock = RLock()
        self._cache = OrderedDict()

    def catalog(self):
        try:
            catalog = deepcopy(self.provider.catalog())
            if not isinstance(catalog, dict):
                raise ValueError('Catalog must be an object')
            json.dumps(catalog, allow_nan=False)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise RuntimeError(f'Horizon catalog unavailable: {exc}') from exc
        if catalog.get('registry_id') != REGISTRY_ID:
            raise RuntimeError('Unexpected horizon model registry')
        choose = getattr(self.provider, 'active_version', None)
        active = choose() if callable(choose) else '2'
        catalog['active_version'] = str(active) if active is not None else None
        return catalog

    @staticmethod
    def _validate(result, selected, expected, as_of, horizon_hours):
        statuses = {'ok', 'insufficient_history', 'event_hold', 'model_unavailable', 'unsupported_horizon'}
        if (not isinstance(result, dict) or result.get('status') not in statuses
                or not isinstance(result.get('model'), dict)
                or not isinstance(result.get('validation'), dict)
                or not isinstance(result.get('trajectory'), list)
                or not isinstance(result.get('reasons'), list)):
            raise RuntimeError('Invalid horizon provider envelope')
        try:
            json.dumps(result, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise RuntimeError('Horizon provider returned invalid or nonfinite metadata') from exc
        model = result.get('model') or {}
        if model.get('version') is not None and str(model['version']) != str(selected):
            raise RuntimeError('Horizon provider returned a different requested version')
        if result.get('status') != 'ok':
            if result.get('trajectory'):
                raise RuntimeError('Unavailable forecast must not contain a trajectory')
            return
        if expected is None or not model.get('version') or str(model['version']) != str(selected):
            raise RuntimeError('Successful horizon forecast requires a catalog-matched version')
        if model.get('registry_id') != REGISTRY_ID or not model.get('run_id'):
            raise RuntimeError('Horizon model identity is incomplete')
        if expected and model.get('run_id') != expected.get('run_id'):
            raise RuntimeError('Horizon provider returned a different model run')
        trajectory = result.get('trajectory')
        if not isinstance(trajectory, list) or len(trajectory) != horizon_hours or not as_of:
            raise RuntimeError('Horizon prediction path has an invalid length')
        origin = datetime.fromisoformat(as_of)
        for hour, point in enumerate(trajectory, start=1):
            try:
                stamp = datetime.fromisoformat(point['timestamp'])
                value = float(point['weight_kg'])
                lower, upper = point.get('lower_kg'), point.get('upper_kg')
                valid = stamp == origin + timedelta(hours=hour) and math.isfinite(value) and 0 < value <= 300
                if (lower is None) != (upper is None):
                    valid = False
                elif lower is not None:
                    lower, upper = float(lower), float(upper)
                    valid = valid and math.isfinite(lower) and math.isfinite(upper) and lower <= value <= upper
                if not valid:
                    raise ValueError('invalid point')
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise RuntimeError('Invalid horizon prediction timestamp, value or interval') from exc

    def _forecast(self, service, workspace_id, horizon_hours, version):
        if isinstance(horizon_hours, bool) or horizon_hours not in HORIZONS:
            raise ValueError('Supported forecast horizons: 1, 24, 72, 168 hours')
        with service.store.lock:
            data = service.store.data_status()
            observations = deepcopy(service.store.observations(720))
        hive = data.get('hive_id')
        if observations and any(row['hive_id'] != hive for row in observations):
            raise RuntimeError('Observation snapshot contains a different hive')
        catalog = self.catalog()
        selected = str(version or catalog.get('default_version') or '') or None
        expected = next((item for item in catalog.get('versions', [])
                         if str(item['version']) == selected), None)
        identity = {'workspace_id': workspace_id, 'hive_id': hive, 'snapshot_id': data['snapshot_id'],
                    'as_of': data.get('end'), 'horizon_hours': horizon_hours,
                    'requested_horizon_hours': horizon_hours}
        fingerprint = digest({**identity, 'model': expected, 'requested_version': selected,
                              'registry_id': REGISTRY_ID})
        with self._lock:
            if fingerprint in self._cache:
                result = deepcopy(self._cache[fingerprint])
                self._cache.move_to_end(fingerprint)
            else:
                try:
                    result = deepcopy(self.provider.predict(observations, horizon_hours, version=selected))
                    self._validate(result, selected, expected, identity['as_of'], horizon_hours)
                except (ImportError, OSError, ValueError, TypeError, KeyError) as exc:
                    raise RuntimeError(f'Horizon model unavailable: {exc}') from exc
                result.update(identity, forecast_id=fingerprint, generated_at=utcnow(),
                              time_basis='recorded_timestamp',
                              time_note='공개 자료의 +00:00은 원본 시간대 미상에 대한 표기 가정입니다.')
                if result['status'] in ('ok', 'insufficient_history', 'event_hold'):
                    self._cache[fingerprint] = deepcopy(result)
                    while len(self._cache) > 24:
                        self._cache.popitem(last=False)
        origin = datetime.fromisoformat(identity['as_of']) if identity['as_of'] else None
        result['historical'] = bool(origin and origin < datetime.now(timezone.utc) - timedelta(hours=48))
        return result, observations

    def forecast(self, service, workspace_id, horizon_hours=168, version=None):
        return self._forecast(service, workspace_id, horizon_hours, version)[0]

    def harvest(self, service, workspace_id, horizon_hours=168, version=None):
        from .harvest import recommend_harvest
        forecast, observations = self._forecast(service, workspace_id, horizon_hours, version)
        context = forecast.get('model', {}).get('context_hours', 72)
        species = species_for(forecast['hive_id'], observations, context if isinstance(context, int) else 72)
        result = recommend_harvest(forecast, observations, species)
        for key in ('workspace_id', 'hive_id', 'snapshot_id', 'forecast_id', 'as_of', 'generated_at',
                    'horizon_hours', 'requested_horizon_hours', 'time_basis', 'historical', 'model'):
            result[key] = deepcopy(forecast.get(key))
        result['forecast'] = forecast
        return result

    def recommendation(self, service, workspace_id):
        """Use the selected shared model over the full supported planning period.

        Model and horizon choices remain available on the technical forecast
        API; the product recommendation always compares the next seven days
        using the current shared TiRex-2 + LSTM artifact.
        """
        version = self.catalog()['active_version']
        if version is None:
            raise RuntimeError('No available shared TiRex-2 + LSTM model')
        result = self.harvest(service, workspace_id, horizon_hours=168, version=version)
        result['selection'] = {'mode': 'automatic', 'model_version': version, 'horizon_hours': 168}
        return result
