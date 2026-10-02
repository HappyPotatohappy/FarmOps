"""Temperature changes must train the served ensemble and preserve provenance."""
from copy import deepcopy
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from app.store import Store


def observations(count=336, start=0, shift=0., hive='TEMP-TEST'):
    return [{'timestamp': t.isoformat(), 'hive_id': hive, 'weight_kg': 40+i*.002,
             'temperature_c': 20+shift+np.sin(i*2*np.pi/24), 'event': 'normal'}
            for i,t in zip(range(start,start+count), pd.date_range('2026-01-01', periods=start+count, freq='h', tz='UTC')[start:])]


class Provider:
    def __init__(self, path): self.artifact_root=path; self.version='2'
    def catalog(self):
        return {'registry_id':'BeeOPS_Horizon_Weight','default_version':self.version,
                'versions':[{'version':self.version,'run_id':'run-'+self.version,
                'components':['lstm','tirex2'],'weights':{'lstm':.1,'tirex2':.9}}]}
    def active_version(self):
        catalog=self.catalog()
        active=next(v for v in catalog['versions'] if v['version']==catalog['default_version'])
        if set(active['components'])!={'lstm','tirex2'}: raise RuntimeError('No active TiRex ensemble')
        return catalog['default_version']


def coordinator(tmp_path):
    from app.temperature_drift import TemperatureDriftCoordinator
    provider=Provider(tmp_path/'models')
    return TemperatureDriftCoordinator(tmp_path,provider),provider


@pytest.mark.parametrize('hours,shift,status',[(23,8.,'waiting'),(24,0.,'stable'),
                                             (24,8.,'drift'),(24,-8.,'drift')])
def test_temperature_detection_requires_full_new_day_and_detects_both_directions(hours,shift,status):
    from app.temperature_drift import reference_for, temperature_check
    baseline=observations()
    reference=reference_for(baseline[-168:],'2','test baseline')
    check=temperature_check(baseline+observations(hours,336,shift),reference)
    assert check['status']==status
    assert check['available_hours']==hours
    if status!='waiting':
        assert check['mean_shift_c']==pytest.approx(shift)


@pytest.mark.parametrize('hours',[1,24])
def test_single_temperature_spike_does_not_trigger_distribution_drift(hours):
    from app.temperature_drift import reference_for, temperature_check
    baseline=observations()
    current=observations(hours,336)
    current[-1]['temperature_c']=60.
    check=temperature_check(baseline+current,reference_for(baseline[-168:],'2','test baseline'))
    assert check['status']==('waiting' if hours<24 else 'stable')


@pytest.mark.parametrize('break_kind,available',[('event',23),('gap',1)])
def test_event_or_hourly_gap_restarts_the_continuous_temperature_window(break_kind,available):
    from app.temperature_drift import reference_for, temperature_check
    baseline=observations()
    current=observations(24,336,8.)
    if break_kind=='event':
        current[0]['event']='inspection'
    else:
        current[-1]['timestamp']=(pd.Timestamp(current[-1]['timestamp'])+pd.Timedelta(hours=1)).isoformat()
    check=temperature_check(baseline+current,reference_for(baseline[-168:],'2','test baseline'))
    assert check['status']=='waiting'
    assert check['available_hours']==available


def test_mean_shift_alone_is_insufficient_when_reference_variability_is_high():
    from app.temperature_drift import reference_for, temperature_check
    baseline=observations()
    for row in baseline:
        row['temperature_c']=20+10*(row['temperature_c']-20)
    check=temperature_check(baseline+observations(24,336,8.),
                            reference_for(baseline[-168:],'2','variable baseline'))
    assert check['mean_shift_c']==pytest.approx(8.)
    assert check['standardized_shift']<1.5
    assert check['status']=='stable'


def test_reference_persists_and_temperature_change_requires_new_observations(tmp_path):
    item,provider=coordinator(tmp_path)
    service=SimpleNamespace(store=Store(tmp_path/'hive'))
    service.store.ingest(observations(),'baseline')
    state=item.observe(service,'TEMP-TEST')
    reference=deepcopy(state['reference'])
    assert reference['count']==168
    assert state['check']['status']=='reference_initialized'
    assert item.observe(service,'TEMP-TEST')['job'] is None
    service.store.ingest(observations(24,336,shift=1.),'normal')
    checked=item.observe(service,'TEMP-TEST')
    assert checked['check']['status']=='stable'
    assert checked['check']['mean_shift_c']==pytest.approx(1.)
    assert checked['reference']==reference
    item.close()
    from app.temperature_drift import TemperatureDriftCoordinator
    restored=TemperatureDriftCoordinator(tmp_path,provider)
    assert restored.status('TEMP-TEST')['reference']==reference
    restored.close()


def test_detected_shift_queues_once_and_failure_keeps_reference(tmp_path,monkeypatch):
    item,_=coordinator(tmp_path)
    service=SimpleNamespace(store=Store(tmp_path/'hive'))
    service.store.ingest(observations(),'baseline')
    reference=item.observe(service,'TEMP-TEST')['reference']
    def fail(*args,**kwargs): raise RuntimeError('training unavailable')
    monkeypatch.setattr(item,'_train_bundle',fail)
    service.store.ingest(observations(168,336,8.),'warm observations')
    queued=item.observe(service,'TEMP-TEST')
    assert queued['check']['status']=='drift'
    assert queued['check']['mean_shift_c']==pytest.approx(8.)
    item.close()
    final=item.status('TEMP-TEST')
    assert final['job']['status']=='failed'
    assert final['reference']==reference
    assert final['active_model']['version']=='2'
    assert item.observe(service,'TEMP-TEST')['job']['job_id']==queued['job']['job_id']
    assert len(item.status('TEMP-TEST')['jobs'])==1


def test_new_reference_is_committed_only_after_successful_global_deployment(tmp_path,monkeypatch):
    item,provider=coordinator(tmp_path)
    service=SimpleNamespace(store=Store(tmp_path/'hive'))
    service.store.ingest(observations(),'baseline')
    old=item.observe(service,'TEMP-TEST')['reference']
    def success(job,rows,reference,progress):
        progress('train',{'epochs_completed':1})
        progress('register',{'mlflow_model_version':'1'})
        provider.version='3'
        progress('deploy',{'version':'3'})
        return {'parent_version':'2','new_version':'3','mlflow_model_version':'1','promoted':True}
    monkeypatch.setattr(item,'_train_bundle',success)
    service.store.ingest(observations(168,336,8.),'warm observations')
    item.observe(service,'TEMP-TEST'); item.close()
    state=item.status('TEMP-TEST')
    assert state['job']['status']=='completed'
    assert state['job']['new_version']=='3'
    assert state['reference']['reference_id']!=old['reference_id']
    assert state['reference']['mean_c']==pytest.approx(28.)
    assert state['reference']['count']==24
    assert state['check']['status']=='reference_updated'
    assert state['check']['reference_mean_c']==pytest.approx(28.)
    assert state['check']['mean_shift_c']==pytest.approx(0.)
    assert state['job']['check']['mean_shift_c']==pytest.approx(8.)
    assert state['active_model']['version']=='3'
    assert all(step['status']=='completed' for step in state['job']['steps'])


def test_drift_in_another_workspace_is_queued_while_shared_training_runs(tmp_path,monkeypatch):
    item,provider=coordinator(tmp_path)
    services={name:SimpleNamespace(store=Store(tmp_path/name)) for name in ('FIRST','SECOND')}
    for name,service in services.items():
        service.store.ingest(observations(hive=name),'baseline')
        item.observe(service,name)
    entered=threading.Event(); release=threading.Event()
    def train(job,rows,reference,progress):
        if job['workspace_id']=='FIRST': entered.set(); assert release.wait(10)
        version='3' if job['workspace_id']=='FIRST' else '4'
        provider.version=version
        for stage in ('train','register','deploy'): progress(stage)
        return {'new_version':version,'promoted':True}
    monkeypatch.setattr(item,'_train_bundle',train)
    services['FIRST'].store.ingest(observations(168,336,8.,'FIRST'),'warm')
    item.observe(services['FIRST'],'FIRST'); assert entered.wait(10)
    services['SECOND'].store.ingest(observations(168,336,8.,'SECOND'),'warm')
    try:
        queued=item.observe(services['SECOND'],'SECOND')
        assert queued['job'] is not None
        assert queued['job']['status']=='queued'
    finally:
        release.set(); item.close()
    assert item.status('SECOND')['job']['status']=='completed'
    assert item.status('SECOND')['active_model']['version']=='4'


def test_queued_same_workspace_rechecks_new_reference_before_retraining(tmp_path,monkeypatch):
    item,provider=coordinator(tmp_path)
    service=SimpleNamespace(store=Store(tmp_path/'hive'))
    service.store.ingest(observations(),'baseline'); item.observe(service,'TEMP-TEST')
    entered=threading.Event(); release=threading.Event()
    def train(job,rows,reference,progress):
        entered.set(); assert release.wait(10)
        provider.version=str(int(provider.version)+1)
        for stage in ('train','register','deploy'): progress(stage)
        return {'new_version':provider.version,'promoted':True}
    monkeypatch.setattr(item,'_train_bundle',train)
    service.store.ingest(observations(168,336,8.),'warm')
    item.observe(service,'TEMP-TEST'); assert entered.wait(10)
    service.store.ingest(observations(24,504,8.),'more warm')
    try:
        queued=item.observe(service,'TEMP-TEST')
        assert queued['job']['status']=='queued'
    finally:
        release.set(); item.close()
    state=item.status('TEMP-TEST')
    assert state['job']['status']=='skipped'
    assert state['active_model']['version']=='3'
    assert state['check']['status']=='stable'


def test_training_split_contains_new_observed_labels_and_no_validation_targets(tmp_path):
    from app.temperature_drift import prepare_training
    rows=observations()+observations(168,336,8.)
    data=prepare_training(rows,rows[335]['timestamp'],168,168)
    assert len(data['train'])>0 and len(data['validation'])>0
    for sample in data['train']:
        assert sample.target_start>pd.Timestamp(rows[335]['timestamp'])
        assert sample.target_end<=pd.Timestamp(data['train_end'])
        assert len(sample.target)>0
    assert min(s.target_start for s in data['validation'])>pd.Timestamp(data['train_end'])
    assert max(s.target_end for s in data['validation'])==pd.Timestamp(rows[-1]['timestamp'])
    assert data['new_observations']==168


def test_recommendation_follows_current_shared_tirex_bundle_and_rejects_other_family(tmp_path):
    from app.horizon_service import HorizonReports
    provider=Provider(tmp_path)
    provider.version='3'
    report=HorizonReports(provider)
    report.harvest=lambda service,workspace_id,horizon_hours,version: {'model':{'version':version}}
    result=report.recommendation(None,'hive')
    assert result['selection']['model_version']=='3'
    assert result['model']['version']=='3'
    provider.catalog=lambda: {'registry_id':'BeeOPS_Horizon_Weight','default_version':'1',
                            'versions':[{'version':'1','components':['lstm'],'weights':{'lstm':1.}}]}
    with pytest.raises(RuntimeError,match='TiRex'):
        report.recommendation(None,'hive')


def test_startup_removes_pending_bundle_left_by_killed_training_but_keeps_versions(tmp_path):
    from app.temperature_drift import TemperatureDriftCoordinator
    provider=Provider(tmp_path/'models')
    stale=tmp_path/'models/.pending-temperature-dead/tirex2'; stale.mkdir(parents=True)
    (stale/'model.ckpt').write_bytes(b'partial')
    kept=tmp_path/'models/3'; kept.mkdir()
    TemperatureDriftCoordinator(tmp_path,provider)
    assert not (tmp_path/'models/.pending-temperature-dead').exists()
    assert kept.is_dir()


@pytest.mark.parametrize('failure_stage',[None,'register','publish','report','manual'])
def test_real_lstm_training_registers_immutable_bundle_and_deploys_only_after_registration(tmp_path,monkeypatch,failure_stage):
    tf=pytest.importorskip('tensorflow')
    from mlflow.tracking import MlflowClient
    from app import tirex_adapter
    from app.horizon_models import HorizonModelService, train_lstm
    from app import temperature_drift
    from app.temperature_drift import TemperatureDriftCoordinator
    model_root=tmp_path/'models'; parent=model_root/'2'; parent.mkdir(parents=True)
    model,_=train_lstm(np.zeros((4,24,2),dtype=np.float32),np.zeros((4,168),dtype=np.float32),epochs=1)
    model.save(parent/'lstm.keras')
    parent_bytes=(parent/'lstm.keras').read_bytes()
    checkpoint=parent/'tirex2'; checkpoint.mkdir()
    (checkpoint/'model.ckpt').write_bytes(b'foundation-fixture')
    (checkpoint/'model-config.yaml').write_text('fixture: true')
    manifest={'registry_id':'BeeOPS_Horizon_Weight','version':'2','run_id':'parent',
              'context_hours':24,'max_horizon_hours':168,'components':['lstm','tirex2'],
              'weights':{'lstm':.3,'tirex2':.7},'files':{'lstm':'lstm.keras','tirex2':'tirex2'},
              'scaler':{'weight_scale':1.,'temperature_mean':20.,'temperature_scale':1.},
              'validation':{'status':'insufficient','calibrated':False},'training_hive_ids':[]}
    (parent/'manifest.json').write_text(json.dumps(manifest))
    (model_root/'index.json').write_text(json.dumps({'registry_id':'BeeOPS_Horizon_Weight',
                   'default_version':'2','versions':[{'version':'2','manifest':'2/manifest.json'}]}))
    class FrozenFoundation:
        def __init__(self,path):
            assert (path/'model.ckpt').read_bytes()==b'foundation-fixture'
        def predict(self,contexts,horizon):
            return np.asarray([np.full(horizon,x[-1,0]) for x in contexts])
    monkeypatch.setattr(tirex_adapter,'TirexAdapter',FrozenFoundation)
    if failure_stage=='register':
        def reject(*args,**kwargs): raise RuntimeError('registry unavailable')
        monkeypatch.setattr(MlflowClient,'create_model_version',reject)
    if failure_stage=='publish':
        original_write=temperature_drift.atomic_json
        def failed_publication(path,value):
            if Path(path)==model_root/'index.json': raise OSError('publication interrupted')
            return original_write(path,value)
        monkeypatch.setattr(temperature_drift,'atomic_json',failed_publication)
    if failure_stage=='report':
        original_tag=MlflowClient.set_tag
        def failed_reporting(client,run_id,key,value,*args,**kwargs):
            if key=='promoted':
                assert json.loads((model_root/'index.json').read_text())['default_version']=='3'
                raise RuntimeError('post-deployment metadata unavailable')
            return original_tag(client,run_id,key,value,*args,**kwargs)
        monkeypatch.setattr(MlflowClient,'set_tag',failed_reporting)
    provider=HorizonModelService(model_root)
    item=TemperatureDriftCoordinator(tmp_path/'runtime',provider)
    epoch_states=[]
    actual_batch_losses=[]
    original_batch=tf.keras.Model.train_on_batch
    def tracked_batch(model,*args,**kwargs):
        value=original_batch(model,*args,**kwargs)
        actual_batch_losses.append(float(np.asarray(value)))
        return value
    monkeypatch.setattr(tf.keras.Model,'train_on_batch',tracked_batch)
    original_save=item._save
    def capture_persisted_epoch():
        original_save()
        saved=json.loads(item.path.read_text())
        if not saved['jobs']:
            return
        saved_job=saved['jobs'][-1]
        progress=saved_job.get('training_progress')
        if progress is not None and (not epoch_states or progress['completed_epochs']!=epoch_states[-1][0]['completed_epochs']):
            epoch_states.append((progress,deepcopy(saved_job['steps'][4]),len(actual_batch_losses)))
    monkeypatch.setattr(item,'_save',capture_persisted_epoch)
    service=SimpleNamespace(store=Store(tmp_path/'hive'))
    service.store.ingest(observations(),'baseline')
    reference=item.observe(service,'TEMP-TEST')['reference']
    if failure_stage=='manual':
        item.request_training(service,'TEMP-TEST')
    else:
        service.store.ingest(observations(168,336,8.),'warm observations')
        item.observe(service,'TEMP-TEST')
    item.close()
    state=item.status('TEMP-TEST')
    assert [progress['completed_epochs'] for progress,_,_ in epoch_states]==list(range(9))
    for progress,train_step,batches in epoch_states:
        assert progress['total_epochs']==8
        assert progress['training_windows']>0
        assert progress['updated_at']
        assert train_step['status']=='running'
        batches_per_epoch=(progress['training_windows']+31)//32
        assert batches==progress['completed_epochs']*batches_per_epoch
        if batches:
            assert progress['last_loss']==actual_batch_losses[batches-1]
        else:
            assert progress['last_loss'] is None
    assert (parent/'lstm.keras').read_bytes()==parent_bytes
    if failure_stage in ('register','publish'):
        assert state['job']['status']=='failed'
        assert state['active_model']['version']=='2'
        assert state['reference']==reference
        if failure_stage=='register': assert not (model_root/'3').exists()
        else:
            assert (model_root/'3').is_dir()
            monkeypatch.setattr(temperature_drift,'atomic_json',original_write)
            retry=TemperatureDriftCoordinator(tmp_path/'runtime',provider)
            service.store.ingest(observations(24,504,8.),'more warm observations')
            retry.observe(service,'TEMP-TEST'); retry.close()
            assert retry.status('TEMP-TEST')['job']['status']=='completed'
            assert retry.status('TEMP-TEST')['active_model']['version']=='4'
    else:
        assert state['job']['status']=='completed',state['job'].get('error')
        assert state['active_model']['version']=='3'
        assert all(step['status'] in ('completed','skipped') for step in state['job']['steps'])
        if failure_stage=='report':
            assert state['job']['promoted'] is True
            assert state['job']['reporting_warnings']==['post-deployment metadata unavailable']
        result=json.loads((model_root/'3/manifest.json').read_text())
        assert result['temperature_training']['before_weights_sha256']!=result['temperature_training']['after_weights_sha256']
        assert (model_root/'3/tirex2/model.ckpt').read_bytes()==b'foundation-fixture'
        assert set(result['weights'])=={'lstm','tirex2'}
        assert all(.05<=weight<=.95 for weight in result['weights'].values())
        assert sum(result['weights'].values())==pytest.approx(1.)
        training=result['temperature_training']
        assert training['old_weights']=={'lstm':.3,'tirex2':.7}
        assert training['new_weights']==result['weights']
        assert training['trained_components']==['lstm','ensemble_weights']
        assert training['frozen_components']==['tirex2']
        assert training['tirex_before_sha256']==training['tirex_after_sha256']
        assert pd.Timestamp(training['training_end'])<pd.Timestamp(training['calibration_start'])
        assert pd.Timestamp(training['calibration_end'])<pd.Timestamp(training['validation_start'])
        calibration=json.loads((model_root/'3/ensemble_calibration.json').read_text())
        assert calibration['new_weights']==result['weights']
        assert max(row['target_timestamp'] for row in calibration['rows'])<state['job']['evaluation']['validation_start']
        for row in state['job']['evaluation']['rows']:
            assert row['previous_kg']==pytest.approx(.3*row['previous_lstm_kg']+.7*row['tirex2_kg'])
            assert row['new_kg']==pytest.approx(result['weights']['lstm']*row['lstm_kg']+result['weights']['tirex2']*row['tirex2_kg'])
        if failure_stage=='manual':
            assert training['reason']=='manual_retraining'
            assert training['training_start']==observations()[24]['timestamp']
            assert item.request_training(service,'TEMP-TEST')['status']=='already_evaluated'
        assert result['mlflow']['model_version']=='1'
        assert result['validation']['calibrated'] is False
        tracking=f'sqlite:///{tmp_path}/runtime/temperature_drift/mlflow.db'
        client=MlflowClient(tracking_uri=tracking,registry_uri=tracking)
        record=client.get_model_version_by_alias('BeeOPS_Horizon_Weight','champion')
        assert record.run_id==result['run_id']
        if failure_stage!='report':
            assert client.get_run(record.run_id).info.status=='FINISHED'
        assert provider.predict(observations(24,480,8.),168)['model']['version']=='3'
