"""Leakage and boundary tests for the independent multi-horizon experiment."""
import numpy as np
import pandas as pd
import pytest
from app import horizon_data as hd


def frame(n=40, hive='real', start='2025-01-01'):
    return pd.DataFrame({'timestamp': pd.date_range(start, periods=n, freq='h', tz='UTC'),
                         'hive_id': hive, 'weight_kg': np.arange(n) / 10 + 40,
                         'temperature_c': np.arange(n) + 10., 'event': 'normal', 'source': 'real'})


def test_deduplication_never_counts_synthetic_or_duplicate_rows():
    real = frame()
    fake = frame(hive='BEE-DEMO'); fake['source'] = 'synthetic'
    actual = hd.canonical_real_rows([real, real.iloc[10:], fake])
    assert len(actual) == 40
    assert set(actual.hive_id) == {'real'}
    conflict = real.iloc[[0]].copy(); conflict['weight_kg'] = 200.
    with pytest.raises(ValueError, match='conflicting'):
        hd.canonical_real_rows([real, conflict])


def test_windows_purge_entire_target_span_and_do_not_cross_gaps_or_hives():
    data = frame()
    data.loc[14, 'event'] = 'feeding'
    data = data.drop(index=22)
    samples, rejected = hd.build_windows(data, context=4, horizon=5,
        start=pd.Timestamp('2025-01-01T10:00Z'), end=pd.Timestamp('2025-01-02T06:00Z'), stride=1)
    assert samples
    assert all(s.target_start >= pd.Timestamp('2025-01-01T10:00Z') for s in samples)
    assert all(s.target_end <= pd.Timestamp('2025-01-02T06:00Z') for s in samples)
    assert all(len(s.context) == 4 and len(s.target) == 5 for s in samples)
    assert any(s.operational for s in samples)  # Future event is retained for operational scores.
    assert any(r['reason'] == 'event_context' for r in rejected)
    assert any(r['reason'] == 'gap' for r in rejected)
    train, _ = hd.build_windows(data, 4, 5, data.timestamp.min(), data.timestamp.max(), 1, clean_targets=True)
    assert all(not s.operational for s in train)


def test_scaler_fits_only_provided_training_contexts():
    train = np.array([[[40., 10.], [41., 12.]], [[42., 14.], [44., 16.]]])
    scaler = hd.fit_scaler(train)
    assert scaler['temperature_mean'] == 13.
    before = dict(scaler)
    encoded = hd.encode_context(np.array([[[100., 1000.], [103., 2000.]]]), scaler)
    assert scaler == before
    assert encoded.shape == (1, 2, 2)
    assert encoded[0, -1, 0] == 0.
    assert encoded[0, 0, 0] < 0


def test_baselines_and_features_use_only_observed_origin_context():
    context = np.column_stack([np.arange(1, 25), np.ones(24) * 20])
    forecasts = hd.baseline_forecasts(context, 30)
    np.testing.assert_array_equal(forecasts['persistence'], [24.] * 30)
    np.testing.assert_array_equal(forecasts['seasonal_24h'][:26], list(range(1,25)) + [1,2])
    assert forecasts['damped_trend'][29] < 54.
    features = hd.tree_features(context, '2025-01-01T23:00:00+00:00', np.array([1,24,168]))
    assert features.shape[0] == 3
    assert np.isfinite(features).all()
    assert features[0, -1] == 1 / 168
    assert features[-1, -1] == 1.


def test_partition_cutoffs_have_disjoint_targets():
    data = frame(1000)
    partitions = hd.partition_ranges(data)
    assert list(partitions) == ['train', 'validation', 'calibration', 'test']
    assert partitions['train'][1] < partitions['validation'][0]
    assert partitions['validation'][1] < partitions['calibration'][0]
    assert partitions['calibration'][1] < partitions['test'][0]


def test_partial_training_targets_are_real_and_stop_at_event_gap_or_cutoff():
    data=frame(30)
    data.loc[20,'event']='feeding'
    samples,_=hd.build_windows(data,4,168,data.timestamp.iloc[0],data.timestamp.iloc[25],1,
                                clean_targets=True,allow_partial_targets=True)
    origin19=next(s for s in samples if s.origin==data.timestamp.iloc[18])
    assert len(origin19.target)==1
    assert origin19.target[0]==pytest.approx(41.9)
    assert origin19.target_end==data.timestamp.iloc[19]
    tail=next(s for s in samples if s.origin==data.timestamp.iloc[24])
    assert len(tail.target)==1
    assert tail.target_end==data.timestamp.iloc[25]
    assert all(s.target_end<=data.timestamp.iloc[25] for s in samples)
