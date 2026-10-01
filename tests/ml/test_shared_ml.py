"""Shared registry must train real bundles without crossing hive/time boundaries."""
from importlib import import_module, util

import numpy as np
import pandas as pd
import pytest


def shared_module():
    assert util.find_spec('app.shared_ml') is not None, 'Shared model lifecycle is not implemented'
    return import_module('app.shared_ml')


def hive(name, weight, start, n=240):
    t=np.arange(n)
    return pd.DataFrame({'timestamp':pd.date_range(start, periods=n, freq='h', tz='UTC'),
        'hive_id':name,'weight_kg':weight+.3*np.sin(t*np.pi/12)+.002*t,
        'temperature_c':20+6*np.sin(t*np.pi/12),'event':'normal'})


def frames():
    return {'light':hive('light',6,'2021-01-01'), 'heavy':hive('heavy',84,'2025-01-01')}


def test_pool_splits_hives_before_windows_and_fits_scaler_only_to_train_rows():
    module=shared_module(); data=frames()
    data['heavy'].loc[144:,'weight_kg']+=20
    pool=module.prepare_pool(data)
    assert pool.windows['train'].shape==(240,24,2)
    assert pool.windows['validation'].shape==(48,24,2)
    for name in ('train','validation','test'):
        x=pool.windows[name]
        assert np.ptp(x[:,:,0],axis=1).max()<1
        assert set(pool.hive_ids[name])=={'light','heavy'}
    expected=np.concatenate([data['heavy'].iloc[:144][['weight_kg','temperature_c']],
                             data['light'].iloc[:144][['weight_kg','temperature_c']]])
    np.testing.assert_allclose(pool.mean,expected.mean(axis=0),atol=1e-12)
    for name,year in [('light',2021),('heavy',2025)]:
        ranges=pool.per_hive_ranges[name]
        assert pd.Timestamp(ranges['train_end'])<pd.Timestamp(ranges['validation_start'])
        assert pd.Timestamp(ranges['validation_end'])<pd.Timestamp(ranges['test_start'])
        assert pd.Timestamp(ranges['monitoring_after']).year==year
        assert ranges['train_rows']==144 and ranges['validation_samples']==24


def test_pool_uses_first_eligible_clean_segment_and_rejects_wrong_hive_identity():
    module=shared_module(); data=hive('one',40,'2026-01-01',1000)
    data.loc[700:,'weight_kg']=100
    pool=module.prepare_pool({'one':data})
    assert pool.per_hive_ranges['one']['rows']==672
    assert pd.Timestamp(pool.per_hive_ranges['one']['test_end'])==data.timestamp.iloc[671]
    with pytest.raises(ValueError,match='hive'):
        module.prepare_pool({'wrong':data})


@pytest.fixture(scope='module')
def trained(tmp_path_factory):
    service=shared_module().SharedModelService(tmp_path_factory.mktemp('shared-registry'))
    data=frames()
    data['recent_upload']=hive('recent_upload',20,'2026-02-01',48)
    result=service.ensure_ready(data)
    assert result['ready'] is True
    return service


def test_shared_pair_has_distinct_trained_artifacts_and_hive_specific_evaluation(trained):
    service=trained
    entries={v['version']:v for v in service.versions()}
    assert set(entries)=={'1','2','3'}
    assert service.status()['version']=='1'
    assert service.status()['registry_id']=='BeeOPS_Common_Weight'
    assert service.status()['model_scope']=='shared'
    first,second=entries['1'],entries['2']
    assert first['deployable'] and second['deployable']
    assert first['run_id']!=second['run_id']
    assert first['model_sha256']!=second['model_sha256']
    assert first['params']['epochs_completed']=='12'
    assert second['params']['epochs_completed']=='8'
    assert second['parent_version']=='1'
    assert first['metrics']['selected_epoch']>=1 and second['metrics']['selected_epoch']>=1
    one=service.version_metadata('1'); two=service.version_metadata('2')
    assert one['per_hive_ranges']==two['per_hive_ranges']
    assert set(one['per_hive_metrics'])=={'light','heavy'}
    assert two['reused_observations_for_demo'] is True
    assert two['evaluation_mode']=='reused_training_observations_for_version_comparison'
    for name,data in frames().items():
        window=data.iloc[-24:][['weight_kg','temperature_c']].to_numpy()[None]
        p1,v1=service.predict(window,'1'); p2,v2=service.predict(window,'2')
        assert (v1,v2)==('1','2') and np.isfinite([p1,p2]).all()
        assert not np.array_equal(p1,p2)
        assert one['per_hive_metrics'][name]['test_samples']==24


def test_shared_quality_demo_is_naturally_rejected_and_cannot_be_activated(trained):
    service=trained
    third=next(v for v in service.versions() if v['version']=='3')
    assert third['metrics']['validation_mae']>.15
    assert third['metrics']['selected_epoch']>=1
    assert 'absolute_mae' in third['gate_reasons']
    assert not third['gate_passed'] and not third['deployable']
    assert third['promotion_policy']=='quality_gate_demo_v1'
    with pytest.raises(ValueError,match='gate'):
        service.rollback('3')
    assert service.status()['version']=='1'


def test_shared_ready_is_idempotent_restart_preserves_artifacts_and_global_activation(trained):
    service=trained
    windows=frames()['heavy'].iloc[-24:][['weight_kg','temperature_c']].to_numpy()[None]
    before={v:service.predict(windows,v)[0].copy() for v in ('1','2')}
    runs={v['version']:v['run_id'] for v in service.versions()}
    result=service.ensure_ready({})
    assert result['status']=='already_available' and result['versions']==['1','2','3']
    service.rollback('2')
    assert service.status()['version']=='2'
    restarted=type(service)(service.runtime)
    assert restarted.status()['version']=='2'
    assert {v['version']:v['run_id'] for v in restarted.versions()}==runs
    assert restarted.ensure_ready(frames())['status']=='already_available'
    for version in ('1','2'):
        np.testing.assert_array_equal(restarted.predict(windows,version)[0],before[version])
    restarted.rollback('1')


def test_new_pool_training_preserves_inherited_hive_cutoffs_and_rejects_consumed_rows(trained):
    service=trained
    service.rollback('1')
    original=service.version_metadata('1')
    with pytest.raises(ValueError,match='168'):
        service.train_candidate_pool(frames(),'operator_request','duplicate-pool')
    data=frames()
    data['light']=hive('light',6,'2021-02-01')
    result=service.train_candidate_pool(data,'operator_request','new-light-only')
    assert result['promoted'] and result['version']=='4'
    metadata=service.version_metadata('4')
    assert metadata['per_hive_ranges']['heavy']==original['per_hive_ranges']['heavy']
    assert pd.Timestamp(metadata['per_hive_ranges']['light']['monitoring_after'])>pd.Timestamp(original['per_hive_ranges']['light']['monitoring_after'])
    assert metadata['evaluated_hive_ids']==['light']
    assert set(metadata['per_hive_metrics'])=={'light'}
    consumed=service.status()['consumed_through_by_hive']
    assert set(consumed)=={'light','heavy'}
    assert pd.Timestamp(consumed['heavy']).year==2025
    with pytest.raises(ValueError,match='168'):
        service.train_candidate_pool(data,'operator_request','duplicate-new-light')
    service.rollback('1')
