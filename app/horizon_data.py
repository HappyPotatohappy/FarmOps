"""Real hourly data and origin-safe features for retrospective horizon experiments.

No serving/training dependency is imported here. Source clocks use a bookkeeping
UTC offset; clock features are recorded-clock cycles, not inferred local time.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

HORIZONS = (1, 24, 72, 168)
COLUMNS = ['timestamp', 'hive_id', 'weight_kg', 'temperature_c', 'event']


@dataclass(frozen=True)
class Window:
    hive_id: str
    origin: pd.Timestamp
    target_start: pd.Timestamp
    target_end: pd.Timestamp
    context: np.ndarray
    target: np.ndarray
    operational: bool

    @property
    def key(self):
        return f'{self.hive_id}|{self.origin.isoformat()}'


def canonical_real_rows(frames):
    data = pd.concat(frames, ignore_index=True).copy()
    if 'source' not in data:
        raise ValueError('An explicit real source is required')
    data = data[~data.source.astype(str).str.contains('synthetic|demo', case=False)
                & ~data.hive_id.astype(str).str.contains('BEE-DEMO', case=False)].copy()
    data.timestamp = pd.to_datetime(data.timestamp, utc=True, errors='raise')
    if data[COLUMNS].isna().any().any():
        raise ValueError('Null observation')
    values = data[['weight_kg', 'temperature_c']].to_numpy(float)
    if not np.isfinite(values).all() or ((values[:, 0] <= 0) | (values[:, 0] > 300)).any() or ((values[:, 1] < -50) | (values[:, 1] > 60)).any():
        raise ValueError('Invalid physical observation')
    if not data.timestamp.eq(data.timestamp.dt.floor('h')).all():
        raise ValueError('Observations must represent completed exact hours')
    unique = data.drop_duplicates(COLUMNS)
    if unique.duplicated(['hive_id', 'timestamp']).any():
        raise ValueError('conflicting observations for the same hive and timestamp')
    return unique.sort_values(['hive_id', 'timestamp']).reset_index(drop=True)


def load_real_sources(root: Path):
    """Expand every raw hive and occupied segment using existing physical filters.

    Published checksums are verified. No coverage/accuracy winner selection and
    no interpolation: gaps and >1 kg numerical events are preserved.
    """
    from scripts.prepare_additional_ufc_data import hourly_sensor
    from scripts.prepare_vecauce_data import load_hourly
    root = Path(root)
    raw_root = root / 'raw'
    zenodo = json.loads((raw_root / 'zenodo_20399470_metadata.json').read_text())
    vec_metadata = json.loads((raw_root / 'vecauce2021/metadata.json').read_text())
    frames, provenance = [], []
    def add(hourly, hive_id, source, species, details):
        frame = hourly.rename(columns={'weight': 'weight_kg', 'ext_temperature': 'temperature_c'}).copy()
        adjacent = frame.index.to_series().diff().eq(pd.Timedelta(hours=1))
        frame['event'] = np.where(adjacent & frame.weight_kg.diff().abs().gt(1), 'colony_alert', 'normal')
        frame = frame.reset_index(names='timestamp')
        frame['timestamp'] = pd.to_datetime(frame.timestamp).dt.tz_localize('UTC')
        frame['hive_id'], frame['source'], frame['species'] = hive_id, source, species
        frames.append(frame[COLUMNS + ['source', 'species']])
        provenance.append({'hive_id': hive_id, 'source': source, 'species': species,
                           'hourly_rows': len(frame), 'event_rows': int(frame.event.ne('normal').sum()),
                           'start': frame.timestamp.min().isoformat(), 'end': frame.timestamp.max().isoformat(), **details})
    for filename in ['dadosColmeia1Apis.csv', 'dadosColmeia2Apis.csv', 'DadosMeliponas.csv']:
        path = raw_root / filename
        content = path.read_bytes()
        info = next(item for item in zenodo['files'] if item['key'] == filename)
        if info['checksum'] != 'md5:' + hashlib.md5(content).hexdigest() or len(content) != info['size']:
            raise ValueError(f'Published checksum/size mismatch: {filename}')
        raw = pd.read_csv(path)
        for sensor in sorted(raw.id_sensor.unique()):
            hourly, stats = hourly_sensor(raw, sensor)
            hive = f'ufc_meliponini_s{sensor}' if filename == 'DadosMeliponas.csv' else ('ufc_apis_1' if '1Apis' in filename else 'ufc_apis_2')
            add(hourly, hive, 'UFC Zenodo 20399470', 'meliponini' if filename == 'DadosMeliponas.csv' else 'apis',
                {'filename': filename, 'sensor_id': int(sensor), 'sha256': hashlib.sha256(content).hexdigest(),
                 'published_checksum_verified': True, 'license': 'CC-BY-4.0', 'filters': stats})
    for entry in vec_metadata['datasetVersion']['files']:
        info = entry['dataFile']
        if not info.get('originalFileName'):
            continue
        path = raw_root / 'vecauce2021' / info['originalFileName']
        hourly, stats = load_hourly(path, info)
        add(hourly, stats['hive_id'], 'Vecauce Latvia 2021', 'apis', {'license': 'CC-BY-4.0', **stats})
    data = canonical_real_rows(frames)
    return data, {'sources': provenance, 'rows': len(data), 'hives': int(data.hive_id.nunique()),
                  'time_basis': 'source_clock_timezone_unknown',
                  'selection': 'All physical-valid occupied hourly bins in all checked-in raw source hives; no model-based or longest-block selection',
                  'aggregation': 'Joint-valid medians over [h,h+1), labeled at completed hour h+1; no imputation',
                  'event_policy': 'Adjacent-hour absolute weight change >1 kg; numerical alert, not a known intervention',
                  'snapshot_id': hashlib.sha256(data.to_csv(index=False).encode()).hexdigest()}


def partition_ranges(frame):
    """Per-hive chronological target partitions, explicitly retrospective transfer.

    Pooled training can include later calendar years than another hive's test;
    this is never represented as historical deployment performance.
    """
    times = frame.sort_values('timestamp').timestamp.reset_index(drop=True)
    edges = [0, int(len(times) * .45), int(len(times) * .65), int(len(times) * .82), len(times)]
    if any(b <= a for a, b in zip(edges, edges[1:])):
        raise ValueError('Insufficient rows for four chronological partitions')
    return {name: (times.iloc[a], times.iloc[b-1]) for name, a, b in
            zip(['train', 'validation', 'calibration', 'test'], edges, edges[1:])}


def build_windows(frame, context, horizon, start, end, stride=24, clean_targets=False, allow_partial_targets=False):
    """Retain future events in evaluation, reject only unavailable/event context.

    Every candidate origin and rejection is recorded. Every target hour must be
    inside its partition. Input history may precede that partition (available at
    the origin); target rows never cross its upper cutoff.
    """
    samples, rejected = [], []
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    for hive_id, group in frame.groupby('hive_id', sort=True):
        group = group.sort_values('timestamp').reset_index(drop=True)
        values = group[['weight_kg', 'temperature_c']].to_numpy(np.float32)
        timestamps = group.timestamp
        events = group.event.ne('normal').to_numpy()
        for i in range(context - 1, len(group) - 1, stride):
            origin = timestamps.iloc[i]
            expected_start, expected_end = origin + pd.Timedelta(hours=1), origin + pd.Timedelta(hours=horizon)
            if expected_start < start or expected_start > end:
                continue
            if allow_partial_targets:
                if not clean_targets:
                    raise ValueError('Partial targets are supported only for clean training')
                history = timestamps.iloc[i-context+1:i+1]
                reason = ('gap' if not history.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all()
                          else 'event_context' if events[i-context+1:i+1].any() else None)
                last = i
                if reason is None:
                    for j in range(i+1,min(i+horizon+1,len(group))):
                        if timestamps.iloc[j]>end or timestamps.iloc[j]-timestamps.iloc[j-1]!=pd.Timedelta(hours=1) or events[j]:
                            break
                        last=j
                    if last == i:
                        reason='no_available_clean_target'
                if reason:
                    rejected.append({'hive_id':str(hive_id),'origin':origin.isoformat(),'horizon_hours':horizon,'reason':reason})
                else:
                    samples.append(Window(str(hive_id),origin,expected_start,timestamps.iloc[last],
                        values[i-context+1:i+1].copy(),values[i+1:last+1,0].copy(),False))
                continue
            reason = None
            if expected_end > end:
                reason = 'target_crosses_cutoff'
            elif i + horizon >= len(group):
                reason = 'incomplete_target'
            else:
                span = timestamps.iloc[i-context+1:i+horizon+1]
                if not span.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all():
                    reason = 'gap'
                elif events[i-context+1:i+1].any():
                    reason = 'event_context'
                elif clean_targets and events[i+1:i+horizon+1].any():
                    reason = 'training_target_event'
            if reason:
                rejected.append({'hive_id': str(hive_id), 'origin': origin.isoformat(), 'horizon_hours': horizon, 'reason': reason})
                continue
            samples.append(Window(str(hive_id), origin, expected_start, expected_end,
                                  values[i-context+1:i+1].copy(), values[i+1:i+horizon+1, 0].copy(),
                                  bool(events[i+1:i+horizon+1].any())))
    return samples, rejected


def fit_scaler(contexts):
    x = np.asarray(contexts, dtype=float)
    relative = x[:, :, 0] - x[:, -1:, 0]
    return {'weight_scale': max(float(relative.std()), .01),
            'temperature_mean': float(x[:, :, 1].mean()),
            'temperature_scale': max(float(x[:, :, 1].std()), .01)}


def encode_context(contexts, scaler):
    x = np.asarray(contexts, dtype=np.float32).copy()
    x[:, :, 0] = (x[:, :, 0] - x[:, -1:, 0]) / scaler['weight_scale']
    x[:, :, 1] = (x[:, :, 1] - scaler['temperature_mean']) / scaler['temperature_scale']
    return x


def baseline_forecasts(context, horizon):
    weights = np.asarray(context)[:, 0]
    hours = np.arange(1, horizon+1)
    slope = (weights[-1] - weights[-min(24, len(weights))]) / max(min(24, len(weights))-1, 1)
    return {'persistence': np.full(horizon, weights[-1]),
            'seasonal_24h': np.resize(weights[-min(24, len(weights)):], horizon),
            'damped_trend': weights[-1] + slope * 24 * (1 - np.exp(-hours / 24))}


def tree_features(context, origin, horizons):
    """No future temperature or observed target-lag is accepted by this API."""
    x = np.asarray(context, dtype=float)
    w, t = x[:, 0], x[:, 1]
    stamp = pd.Timestamp(origin)
    features = [w[-1], t[-1], t.mean(), t.std(), np.sin(2*np.pi*stamp.hour/24), np.cos(2*np.pi*stamp.hour/24)]
    for lag in (1, 3, 6, 12, 23, 47, 71, 167):
        features.append(w[-1] - w[max(0, len(w)-1-lag)])
    for width in (6, 24, 72, 168):
        recent = w[-width:]
        features.extend([recent.mean()-w[-1], recent.std(), recent.max()-recent.min()])
    h = np.asarray(horizons, dtype=float)
    return np.column_stack([np.tile(features, (len(h), 1)), h / 168.])
