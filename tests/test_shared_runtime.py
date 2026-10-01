"""One model identity and deployment alias must serve isolated hive ledgers."""
import numpy as np
import pytest
from fastapi.testclient import TestClient
from app.main import create_app
from app.store import Store
from test_store import rows


class SharedFixture:
    registry_id='BeeOPS_Common_Weight'
    def __init__(self): self.current='1'; self.preparations=[]
    def status(self):
        return {'ready':True,'version':self.current,'model_scope':'shared','registry_id':self.registry_id,
                'minimum_training_rows':168,'model_source':'shared-test'}
    def ensure_ready(self,frames):
        self.preparations.append(sorted(frames))
        return {'ready':True,'status':'already_available','versions':['1','2','3'],'version':self.current}
    def predict(self,windows,version=None):
        selected=version or self.current
        return windows[:,-1,0].astype(float)+int(selected)*.1,selected
    def version_metadata(self,version):
        return {'version':str(version),'run_id':'common-run-'+str(version),'model_scope':'shared',
                'registry_id':self.registry_id,'per_hive_ranges':{
                    'BEE-01':{'test_end':rows(1,start=40)[0]['timestamp'],'train_end':rows(1,start=20)[0]['timestamp']},
                    'OTHER':{'test_end':rows(1,start=30)[0]['timestamp'],'train_end':rows(1,start=10)[0]['timestamp']}}}
    def versions(self):
        return [{'version':v,'run_id':'common-run-'+v,'model_scope':'shared','registry_id':self.registry_id,
                 'champion':v==self.current,'deployable':v in ('1','2'),'gate_passed':v in ('1','2'),
                 'params':{},'metrics':{}} for v in ('1','2','3')]
    def rollback(self,version):
        if version=='3': raise ValueError('Dedicated gate demonstration is blocked')
        old=self.current; self.current=version
        return {'version':version,'previous_version':old,'model_scope':'shared','registry_id':self.registry_id}


def test_shared_alias_and_model_identity_across_hives_preserves_old_predictions(tmp_path):
    legacy=Store(tmp_path); legacy.ingest(rows(50),'legacy')
    previous=legacy.record_prediction(rows(24,start=26),99,'1')
    model=SharedFixture()
    with TestClient(create_app(tmp_path,shared_model=model)) as client:
        other,_=client.app.state.workspaces.ensure('OTHER')
        sample=rows(50)
        for row in sample: row['hive_id']='OTHER'; row['weight_kg']+=10
        other.store.ingest(sample,'other')
        for hive in ('BEE-01','OTHER'):
            scope={'workspace_id':hive}
            report=client.get('/forecast/report',params={**scope,'model_version':'1'}).json()
            assert report['model']['registry_id']=='BeeOPS_Common_Weight'
            assert report['model']['run_id']=='common-run-1'
            expected=rows(1,start=40 if hive=='BEE-01' else 30)[0]['timestamp']
            assert report['model']['monitoring_after']==expected
            payload=client.get('/demo/payload',params=scope).json()
            prediction=client.post('/predict',params={**scope,'model_version':'1'},json=payload).json()
            assert prediction['registry_id']=='BeeOPS_Common_Weight' and prediction['run_id']=='common-run-1'
            assert prediction['predicted_weight_kg']==pytest.approx((50.49 if hive=='BEE-01' else 60.49)+.1,abs=1e-5)
        assert legacy.predictions()[0]['prediction_id']==previous['prediction_id']
        assert legacy.predictions()[0]['predicted_weight_kg']==99
        switched=client.post('/models/rollback',params={'workspace_id':'OTHER'},json={'version':'2'})
        assert switched.status_code==200,switched.text
        for hive in ('BEE-01','OTHER'):
            health=client.get('/health',params={'workspace_id':hive}).json()
            assert health['version']=='2' and health['model_scope']=='shared'
            assert health['registry_id']=='BeeOPS_Common_Weight'
        rejected=client.post('/models/rollback',json={'version':'3'})
        assert rejected.status_code==422 and model.current=='2'
        catalog=client.get('/models/catalog').json()
        assert len(catalog)==2 and all(g['model_scope']=='shared' for g in catalog)
        assert client.get('/workspaces').json()['model_count_total']==3


def test_new_hive_uses_existing_common_models_without_local_training(tmp_path):
    model=SharedFixture()
    with TestClient(create_app(tmp_path,shared_model=model)) as client:
        service=client.app.state.service
        result=service.import_upload(rows(30),'new hive')
        assert result['status']=='completed'
        assert result['model_initialization']['status']=='already_ready'
        assert result['forecast']['prediction']['registry_id']=='BeeOPS_Common_Weight'
        assert result['forecast']['prediction']['run_id']=='common-run-1'
        assert len(service.model.versions())==3


def test_shared_metadata_for_unseen_hive_does_not_borrow_another_hives_cutoff(tmp_path):
    from app.shared_runtime import SharedModelCoordinator
    coordinator=SharedModelCoordinator(tmp_path,model=SharedFixture())
    data=rows(30)
    for row in data: row['hive_id']='UNSEEN'
    Store(tmp_path).ingest(data,'unseen')
    adapter=coordinator.for_runtime(tmp_path)
    metadata=adapter.version_metadata('1')
    assert metadata['monitoring_after'] is None and metadata['test_end'] is None
    assert metadata['run_id']=='common-run-1'


def test_consumed_training_cutoff_does_not_rewind_when_switching_common_version(tmp_path):
    from app.shared_runtime import SharedModelCoordinator
    model=SharedFixture()
    original=model.status
    consumed=rows(1,start=60)[0]['timestamp']
    model.status=lambda: {**original(),'consumed_through_by_hive':{'BEE-01':consumed}}
    Store(tmp_path).ingest(rows(80),'history')
    adapter=SharedModelCoordinator(tmp_path,model=model).for_runtime(tmp_path)
    assert adapter.status()['consumed_through']==consumed
    assert adapter.version_metadata('1')['monitoring_after']==rows(1,start=40)[0]['timestamp']


def test_shared_gate_demo_reuses_real_rejected_v3_without_changing_observations(tmp_path):
    model=SharedFixture()
    Store(tmp_path).ingest(rows(50),'history')
    with TestClient(create_app(tmp_path,shared_model=model)) as client:
        service=client.app.state.service
        before=service.store.observations()
        service._demo('gate_fail')
        assert service.demo['status']=='completed',service.demo
        assert service.demo['action']['version']=='3'
        assert service.demo['action']['reused_existing_candidate'] is True
        assert service.demo['action']['gate_passed'] is False
        assert service.store.observations()==before
        assert model.current=='1' and len(model.versions())==3


def test_legacy_training_and_drift_keys_do_not_suppress_shared_actions(tmp_path,monkeypatch):
    from app.store import digest
    Store(tmp_path).ingest(rows(260),'history')
    with TestClient(create_app(tmp_path,shared_model=SharedFixture())) as client:
        service=client.app.state.service
        snapshot=digest(service.clean_training_frame().to_dict('records'))
        service.store.set_meta('last_attempt_snapshot',snapshot)
        monkeypatch.setattr(service.executor,'submit',lambda *args:None)
        assert service.request_training()['status']=='queued'
        assert service.store.get_meta('shared-last_attempt_snapshot')==snapshot
        assert service.store.get_meta('last_attempt_snapshot')==snapshot
        service.store.set_meta('quality-alert-1',True)
        service.store.set_meta('demo_runtime',True)
        service.demo={'status':'running','scenario':'drift'}
        monkeypatch.setattr(service.store,'quality',lambda *args:{'status':'degraded','mae':1,'count':24})
        monkeypatch.setattr(service,'request_training',lambda *args:{'status':'queued'})
        service.check_quality()
        assert service.store.get_meta('shared-quality-alert-1') is True
        assert any(a['category']=='model' for a in service.store.alerts())


def test_shared_bootstrap_reports_the_initial_activation(tmp_path):
    import pandas as pd
    from app.shared_runtime import SharedModelCoordinator
    model=SharedFixture()
    model.ensure_ready=lambda frames:{'ready':True,'status':'created',
        'results':[{'version':'1','promoted':True},{'version':'2','promoted':False}]}
    adapter=SharedModelCoordinator(tmp_path,model=model).for_runtime(tmp_path)
    result=adapter.bootstrap(pd.DataFrame(rows(240)))
    assert result['version']=='1' and result['promoted'] is True
