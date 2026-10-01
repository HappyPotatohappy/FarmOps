"""Read-only rolling-origin comparison using the currently active shared bundle.

This is retrospective inference, never an insertion into the live prediction
ledger. The current artifact may have learned from the displayed time period;
the resulting MAE is descriptive and is not a deployment/holdout quality gate.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from datetime import datetime, timedelta
import math
from threading import RLock

import numpy as np

from .horizon_service import HorizonReports
from .store import digest, utcnow


class HorizonHistoryReports:
    def __init__(self, provider):
        self.provider = provider
        self._catalog = HorizonReports(provider)
        self._cache = OrderedDict()
        self._lock = RLock()

    @staticmethod
    def _current_ensemble(model):
        weights = model.get('weights') or {}
        return (set(model.get('components') or []) == {'lstm', 'tirex2'}
                and set(weights) == {'lstm', 'tirex2'}
                and all(math.isfinite(value) and .05-1e-8 <= value <= .95+1e-8 for value in weights.values())
                and math.isclose(sum(weights.values()), 1., abs_tol=1e-8)
                and bool(model.get('run_id')))

    @staticmethod
    def _windows(observations, context_hours, limit, prediction_limit):
        candidates = []
        for index in range(max(context_hours, len(observations)-limit), len(observations)):
            # The target observation is checked and scored separately. It is
            # deliberately absent from the context passed to inference.
            context = observations[index-context_hours:index]
            target = observations[index]
            points = [*context, target]
            try:
                stamps = [datetime.fromisoformat(row['timestamp']) for row in points]
                values = np.asarray([[row['weight_kg'], row['temperature_c']] for row in points], dtype=float)
                valid = (all(stamp.tzinfo is not None for stamp in stamps)
                    and all(right-left == timedelta(hours=1) for left, right in zip(stamps, stamps[1:]))
                    and len({row['hive_id'] for row in points}) == 1
                    and np.isfinite(values).all()
                    and ((values[:, 0] > 0) & (values[:, 0] <= 300)).all()
                    and ((values[:, 1] >= -50) & (values[:, 1] <= 60)).all())
            except (KeyError, ValueError, TypeError, OverflowError):
                valid = False
            if valid:
                candidates.append((values[:-1], stamps[-2], target))
        return candidates[-prediction_limit:]

    def report(self, service, workspace_id, limit=168, prediction_limit=48):
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 720:
            raise ValueError('History limit must be between 1 and 720 observations')
        if isinstance(prediction_limit, bool) or not isinstance(prediction_limit, int) or not 1 <= prediction_limit <= 72:
            raise ValueError('Prediction limit must be between 1 and 72 targets')
        active = getattr(self.provider, 'active_version', None)
        version = str((active() if callable(active) else '2') or '')
        catalog = self._catalog.catalog()
        model = deepcopy(next((item for item in catalog.get('versions', [])
                              if str(item['version']) == version), {}))
        context_hours = model.get('context_hours', 168)
        if isinstance(context_hours, bool) or not isinstance(context_hours, int) or not 1 <= context_hours <= 720:
            raise RuntimeError('Invalid history model context length')
        with service.store.lock:
            data = deepcopy(service.store.data_status())
            observations = deepcopy(service.store.observations(context_hours+limit))
        if any(row['hive_id'] != data.get('hive_id') for row in observations):
            raise RuntimeError('Observation snapshot contains a different hive')
        identity = {'workspace_id': workspace_id, 'hive_id': data.get('hive_id'),
                    'snapshot_id': data['snapshot_id'], 'as_of': data.get('end'),
                    'model': model, 'limit': limit, 'prediction_limit': prediction_limit}
        fingerprint = digest(identity)
        with self._lock:
            if fingerprint in self._cache:
                self._cache.move_to_end(fingerprint)
                return deepcopy(self._cache[fingerprint])
            result = {**identity, 'status': 'insufficient_history', 'history_id': fingerprint,
                'generated_at': utcnow(), 'time_basis': 'recorded_timestamp',
                'horizon_hours': 1, 'context_hours': context_hours, 'retrospective': True,
                'observations': observations[-limit:], 'predictions': [], 'reasons': [],
                'metrics': {'count': 0, 'mae_kg': None,
                    'evaluation': 'retrospective_rolling_origin', 'is_live_performance': False},
                'evaluation_note': '현재 공통 모델로 각 시점 직전의 관측만 입력해 1시간 뒤를 다시 예측했습니다. MAE는 과거 구간의 재계산 오차이며 실시간 운영·미사용 시험 성능이 아닙니다.'}
            if not self._current_ensemble(model):
                result.update(status='model_unavailable', reasons=['active_tirex2_lstm_bundle_unavailable'])
                return result
            windows = self._windows(observations, context_hours, limit, prediction_limit)
            if not windows:
                result['reasons'] = [f'requires_{context_hours}_continuous_context_hours_before_a_target']
            else:
                try:
                    predicted = np.asarray(self.provider.predict_batch(
                        np.stack([window[0] for window in windows]), horizon_hours=1, version=version,
                        origins=[window[1] for window in windows]), dtype=float)
                    if (predicted.shape != (len(windows), 1) or not np.isfinite(predicted).all()
                            or (predicted <= 0).any() or (predicted > 300).any()):
                        raise ValueError('Invalid historical prediction shape or values')
                except (ImportError, OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
                    raise RuntimeError(f'Historical prediction unavailable: {exc}') from exc
                for (_, origin, target), value in zip(windows, predicted[:, 0]):
                    actual = float(target['weight_kg'])
                    result['predictions'].append({'origin_timestamp': origin.isoformat(),
                        'target_timestamp': target['timestamp'], 'predicted_weight_kg': float(value),
                        'actual_weight_kg': actual, 'absolute_error_kg': abs(actual-float(value)),
                        'model_version': version, 'event': target.get('event', 'normal')})
                result['metrics'].update(count=len(windows),
                    mae_kg=sum(point['absolute_error_kg'] for point in result['predictions'])/len(windows))
                result['status'] = 'ok'
            self._cache[fingerprint] = deepcopy(result)
            while len(self._cache) > 24:
                self._cache.popitem(last=False)
            return result
