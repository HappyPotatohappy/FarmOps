"""Past chart points use the current shared bundle and only preceding observations."""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from app.horizon_models import HorizonModelService
from app.store import Store
from test_horizon_models import manifest
from test_store import rows


class BatchFixture:
    def __init__(self):
        self.version = '2'
        self.calls = []
        self.fail = False

    def catalog(self):
        return {'registry_id': 'BeeOPS_Horizon_Weight', 'default_version': self.version,
                'versions': [{'registry_id': 'BeeOPS_Horizon_Weight', 'version': version,
                    'run_id': 'run-' + version, 'context_hours': 168, 'max_horizon_hours': 168,
                    'components': ['lstm', 'tirex2'], 'weights': {'lstm': .1, 'tirex2': .9}}
                    for version in ('2', '3')]}

    def active_version(self):
        return self.version

    def predict_batch(self, contexts, horizon_hours=1, version=None, origins=None):
        if self.fail:
            raise RuntimeError('batch artifact unavailable')
        self.calls.append((np.asarray(contexts).copy(), horizon_hours, version, list(origins)))
        return np.asarray(contexts)[:, -1, :1] + .03


def report_service(tmp_path, count=240):
    store = Store(tmp_path)
    if count:
        store.ingest(rows(count), 'fixture')
    return SimpleNamespace(store=store)


def history(provider):
    from app import horizon_history
    return horizon_history.HorizonHistoryReports(provider)


def test_history_aligns_one_hour_targets_without_reading_their_observations(tmp_path):
    service = report_service(tmp_path)
    provider = BatchFixture()
    reports = history(provider)
    with service.store.db() as db:
        before = list(db.iterdump())
    result = reports.report(service, 'BEE-01')
    assert result['status'] == 'ok'
    assert len(result['observations']) == 168
    assert result['observations'][0]['timestamp'] == '2025-01-04T00:00:00+00:00'
    assert len(result['predictions']) == result['metrics']['count'] == 48
    first = result['predictions'][0]
    assert first['origin_timestamp'] == '2025-01-08T23:00:00+00:00'
    assert first['target_timestamp'] == '2025-01-09T00:00:00+00:00'
    assert first['actual_weight_kg'] == pytest.approx(51.92)
    assert first['predicted_weight_kg'] == pytest.approx(51.94)
    assert result['metrics']['mae_kg'] == pytest.approx(.02)
    assert result['metrics']['evaluation'] == 'retrospective_rolling_origin'
    assert result['metrics']['is_live_performance'] is False
    contexts, horizon, version, origins = provider.calls[0]
    assert len(provider.calls) == 1
    assert contexts.shape == (48, 168, 2)
    assert contexts[0, 0, 0] == pytest.approx(50.24)
    assert contexts[0, -1, 0] == pytest.approx(51.91)
    assert contexts[-1, -1, 0] == pytest.approx(52.38)
    assert horizon == 1 and version == '2'
    assert origins[-1].isoformat() == '2025-01-10T22:00:00+00:00'
    with service.store.db() as db:
        assert list(db.iterdump()) == before


def test_history_cache_follows_active_version_and_observation_snapshot(tmp_path):
    service = report_service(tmp_path)
    provider = BatchFixture()
    reports = history(provider)
    first = reports.report(service, 'BEE-01')
    first['predictions'].clear()
    second = reports.report(service, 'BEE-01')
    assert len(second['predictions']) == 48
    assert len(provider.calls) == 1
    provider.version = '3'
    third = reports.report(service, 'BEE-01')
    assert third['model']['version'] == '3'
    assert third['history_id'] != second['history_id']
    service.store.ingest(rows(1, start=240), 'append')
    fourth = reports.report(service, 'BEE-01')
    assert fourth['snapshot_id'] != third['snapshot_id']
    assert fourth['history_id'] != third['history_id']
    assert len(provider.calls) == 3


def test_history_can_display_complete_observed_period_with_bounded_prediction_work(tmp_path):
    provider = BatchFixture()
    result = history(provider).report(report_service(tmp_path,count=609),'BEE-01',limit=720)
    assert len(result['observations']) == 609
    assert result['observations'][0]['timestamp'] == rows(1)[0]['timestamp']
    assert len(result['predictions']) == result['metrics']['count'] == 48
    assert provider.calls[0][0].shape == (48,168,2)


@pytest.mark.parametrize('count,expected', [(0, 0), (168, 0), (169, 1), (180, 12)])
def test_history_needs_a_complete_context_before_each_target(tmp_path, count, expected):
    provider = BatchFixture()
    result = history(provider).report(report_service(tmp_path, count), 'BEE-01')
    assert len(result['predictions']) == expected
    assert result['status'] == ('ok' if expected else 'insufficient_history')
    assert result['metrics']['count'] == expected
    if not expected:
        assert result['metrics']['mae_kg'] is None
        assert provider.calls == []


def test_history_rejects_malformed_outputs_without_caching_a_fake_line(tmp_path):
    service = report_service(tmp_path)
    provider = BatchFixture()
    reports = history(provider)
    original = provider.predict_batch
    for broken in (np.ones((48, 168)), np.full((48, 1), np.nan), np.zeros((48, 1))):
        provider.predict_batch = lambda *a, output=broken, **kw: output
        with pytest.raises(RuntimeError, match='prediction'):
            reports.report(service, 'BEE-01')
    provider.predict_batch = original
    assert reports.report(service, 'BEE-01')['status'] == 'ok'


def test_history_never_substitutes_a_different_ensemble_family(tmp_path):
    provider = BatchFixture()
    catalog = provider.catalog()
    catalog['versions'][0]['components'] = ['lstm', 'lightgbm']
    provider.catalog = lambda: deepcopy(catalog)
    result = history(provider).report(report_service(tmp_path), 'BEE-01')
    assert result['status'] == 'model_unavailable'
    assert result['predictions'] == []
    assert len(result['observations']) == 168
    assert not provider.calls


@pytest.mark.parametrize('lstm_weight', [.05, .4, .95])
def test_history_continues_with_retrained_ensemble_blend(tmp_path, lstm_weight):
    provider = BatchFixture()
    catalog = provider.catalog()
    catalog['versions'][0]['weights'] = {'lstm': lstm_weight, 'tirex2': 1-lstm_weight}
    provider.catalog = lambda: deepcopy(catalog)
    result = history(provider).report(report_service(tmp_path), 'BEE-01')
    assert result['status'] == 'ok'
    assert result['model']['weights'] == catalog['versions'][0]['weights']
    assert len(result['predictions']) == result['metrics']['count'] == 48


def test_batch_inference_combines_every_context_in_one_component_call(tmp_path):
    manifest(tmp_path, components=['lstm', 'tirex2'], weights={'lstm': .1, 'tirex2': .9},
             files={'lstm': 'lstm.keras', 'tirex2': 'tirex2'})
    provider = HorizonModelService(tmp_path)
    calls = []

    def lstm(values, training=False):
        calls.append(('lstm', values.shape))
        return np.full((len(values), 168), .2)

    class Tirex:
        def predict(self, contexts, horizon):
            calls.append(('tirex2', len(contexts), horizon))
            return contexts[:, -1, :1] + np.full((len(contexts), horizon), .4)

    provider._models['1'] = {'lstm': lstm, 'tirex2': Tirex()}
    contexts = np.empty((48, 24, 2), dtype=np.float32)
    contexts[:, :, 0] = np.arange(48)[:, None] + 40
    contexts[:, :, 1] = 20
    assert hasattr(provider, 'predict_batch'), 'Shared batch inference is not implemented'
    actual = provider.predict_batch(contexts, horizon_hours=1, version='1')
    assert actual.shape == (48, 1)
    assert actual[0, 0] == pytest.approx(40.38)
    assert actual[-1, 0] == pytest.approx(87.38)
    assert calls == [('lstm', (48, 24, 2)), ('tirex2', 48, 1)]


@pytest.mark.parametrize('contexts', [np.zeros((0, 24, 2)), np.ones((1, 23, 2)),
    np.full((1, 24, 2), np.nan), np.full((1, 24, 2), 400.)])
def test_batch_inference_validates_context_before_loading_components(tmp_path, contexts):
    manifest(tmp_path)
    provider = HorizonModelService(tmp_path)
    assert hasattr(provider, 'predict_batch'), 'Shared batch inference is not implemented'
    with pytest.raises(ValueError, match='context|Context'):
        provider.predict_batch(contexts, horizon_hours=1, version='1')


@pytest.mark.parametrize('default,expected', [('3', '3'), ('1', '2'), ('missing', '2')])
def test_active_version_uses_shared_family_and_falls_back_to_base_bundle(tmp_path, default, expected):
    base = manifest(tmp_path)
    versions = [{'version': '1', 'manifest': '1/manifest.json'}]
    for version in ('2', '3'):
        folder = tmp_path/version
        folder.mkdir()
        body = {**base, 'version': version, 'run_id': 'run-'+version,
                'components': ['lstm', 'tirex2'], 'weights': {'lstm': .1, 'tirex2': .9},
                'files': {'lstm': 'lstm.keras', 'tirex2': 'tirex2'}}
        (folder/'manifest.json').write_text(json.dumps(body))
        (folder/'lstm.keras').write_bytes(b'artifact')
        (folder/'tirex2').mkdir()
        for name in ('model.ckpt', 'model-config.yaml'):
            (folder/'tirex2'/name).write_bytes(b'artifact')
        versions.append({'version': version, 'manifest': version+'/manifest.json'})
    (tmp_path/'index.json').write_text(json.dumps({'registry_id': 'BeeOPS_Horizon_Weight',
        'default_version': default, 'versions': versions}))
    provider = HorizonModelService(tmp_path)
    assert hasattr(provider, 'active_version'), 'Shared family selection is not implemented'
    assert provider.active_version() == expected
    (tmp_path/'2/tirex2/model.ckpt').unlink()
    (tmp_path/'3/tirex2/model.ckpt').unlink()
    assert provider.active_version() is None
