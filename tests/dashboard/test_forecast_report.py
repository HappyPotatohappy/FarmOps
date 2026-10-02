"""Fresh version-specific inference must not reuse or mutate champion history."""
import copy
import numpy as np
import pytest
from fastapi.testclient import TestClient
from app.main import create_app
from app.service import Service
from test_store import rows


class VersionedForecast:
    def __init__(self):
        self.champion='1'; self.calls=[]; self.wrong_version=False; self.fail=False

    def status(self): return {'ready':True,'version':self.champion}
    def versions(self): return []

    def version_metadata(self,version):
        return {'version':version,'run_id':f'run-{version}',
                'monitoring_after':rows(1,start=25 if version=='1' else 40)[0]['timestamp']}

    def predict(self,windows,version=None):
        if self.fail: raise RuntimeError('artifact unavailable')
        self.calls.append((windows.copy(),version))
        return windows[:,-1,0].astype(float)+int(version)*.1,'999' if self.wrong_version else version

    def rollback(self,version):
        previous=self.champion; self.champion=version
        return {'version':version,'previous_version':previous,'rolled_back':True}


@pytest.fixture
def report_service(tmp_path):
    service=Service(tmp_path,VersionedForecast())
    service.store.ingest(rows(50),'fixture')
    yield service
    service.close()


def database_dump(service):
    with service.store.db() as db: return list(db.iterdump())


def test_exact_version_fresh_windows_target_alignment_and_no_ledger_mutation(report_service):
    service=report_service
    service.store.record_prediction(rows(24),99,'1')  # Deliberately stale stored prediction.
    before=database_dump(service)
    report=service.forecast_report('2')
    assert report['status']=='completed'
    assert report['workspace_id']==report['hive_id']=='BEE-01'
    assert report['horizon_hours']==1 and report['evaluation_mode']=='rolling_one_step'
    assert report['model']['version']=='2' and report['model']['run_id']=='run-2'
    assert report['model']['is_champion'] is False
    windows,version=service.model.calls[-1]
    assert version=='2' and windows.shape==(27,24,2)
    np.testing.assert_allclose(windows[0,:,0],[r['weight_kg'] for r in rows(24)])
    first=report['rows'][0]
    assert first['target_timestamp']==rows(1,start=24)[0]['timestamp']
    assert first['input_end']==rows(1,start=23)[0]['timestamp']
    assert first['actual_weight_kg']==50.24
    assert first['predicted_weight_kg']==pytest.approx(50.43,abs=1e-5)
    assert first['persistence_weight_kg']==50.23 and first['seasonal24_weight_kg']==50
    assert first['historical_replay'] is True and first['monitoring_eligible'] is False
    assert report['next_forecast']['target_timestamp']==rows(1,start=50)[0]['timestamp']
    assert report['next_forecast']['predicted_delta_kg']==pytest.approx(.2,abs=1e-5)
    assert report['next_forecast']['seasonal24_weight_kg']==50.26
    assert database_dump(service)==before


def test_metrics_use_exact_version_cutoff_and_exclude_abnormal_context(report_service):
    service=report_service
    one=service.forecast_report('1'); two=service.forecast_report('2')
    assert one['metrics']['post_cutoff_clean']['count']==24
    score=two['metrics']['post_cutoff_clean']
    assert score['count']==9 and score['excluded_count']==17
    assert score['model_mae_kg']==pytest.approx(.19,abs=1e-5)
    assert score['persistence_mae_kg']==pytest.approx(.01)
    assert score['seasonal24_mae_kg']==pytest.approx(.24)
    assert score['improvement_vs_persistence_kg']==pytest.approx(-.18,abs=1e-5)
    assert two['metrics']['all_window_descriptive']['count']==26
    service.store.ingest(rows(1,start=50,event='harvest'),'operation')
    service.store.ingest(rows(2,start=51),'operation')
    after=service.forecast_report('2')
    assert after['metrics']['post_cutoff_clean']['count']==9
    assert after['rows'][-3]['target_clean'] is False
    assert after['rows'][-1]['context_clean'] is False
    assert after['next_forecast']['context_clean'] is False


def test_same_snapshot_determinism_cache_isolation_and_append_refresh(report_service):
    service=report_service
    first=service.forecast_report('2',limit=7)
    assert len(first['rows'])==7 and service.model.calls[-1][0].shape==(8,24,2)
    original=copy.deepcopy(first)
    first['rows'][0]['predicted_weight_kg']=999
    second=service.forecast_report('2',limit=7)
    assert second==original and len(service.model.calls)==1
    service.store.ingest(rows(1,start=50),'append')
    third=service.forecast_report('2',limit=7)
    assert third['snapshot_id']!=original['snapshot_id']
    assert third['fingerprint']!=original['fingerprint']
    assert third['next_forecast']['target_timestamp']==rows(1,start=51)[0]['timestamp']
    assert third['next_forecast']['predicted_weight_kg']!=original['next_forecast']['predicted_weight_kg']
    assert len(service.model.calls)==2


def test_adapter_cannot_substitute_requested_model(report_service):
    report_service.model.wrong_version=True
    with pytest.raises(RuntimeError,match='different requested version'):
        report_service.forecast_report('2')


def test_unknown_cutoff_is_descriptive_only_and_missing_history_is_explicit(report_service,tmp_path):
    report_service.model.version_metadata=lambda version: {'version':version,'run_id':'legacy'}
    report=report_service.forecast_report('1')
    assert report['metrics']['post_cutoff_clean']['count']==0
    assert report['metrics']['post_cutoff_clean']['status']=='metadata_unavailable'
    assert report['metrics']['all_window_descriptive']['count']==26
    empty=Service(tmp_path/'empty',VersionedForecast())
    try:
        assert empty.forecast_report()['status']=='insufficient_data'
        assert not empty.model.calls
        empty.store.ingest(rows(24),'fixture')
        exact=empty.forecast_report()
        assert exact['rows']==[] and exact['next_forecast']['target_timestamp']==rows(1,start=24)[0]['timestamp']
    finally: empty.close()


def test_metadata_cannot_claim_another_version_or_hive(report_service):
    report_service.model.version_metadata=lambda version: {'version':'1','run_id':'wrong'}
    with pytest.raises(RuntimeError,match='metadata'):
        report_service.forecast_report('2')
    report_service.model.version_metadata=lambda version: {'version':version,'hive_id':'OTHER'}
    with pytest.raises(RuntimeError,match='hive'):
        report_service.forecast_report('2')


@pytest.mark.parametrize('bad_values',[np.array([50.]),np.full(27,np.nan),np.full(27,301.)])
def test_invalid_adapter_outputs_are_rejected(report_service,bad_values):
    report_service.model.predict=lambda windows,version: (bad_values,version)
    with pytest.raises(RuntimeError,match='prediction'):
        report_service.forecast_report('1')


def test_scoped_api_uses_selected_workspace_and_activation_recomputes(tmp_path):
    model=VersionedForecast()
    with TestClient(create_app(tmp_path,model=model,model_factory=lambda _:VersionedForecast())) as client:
        primary=client.app.state.service
        primary.store.ingest(rows(50),'primary')
        other,_=client.app.state.workspaces.ensure('OTHER')
        sample=rows(50)
        for row in sample: row['hive_id']='OTHER'; row['weight_kg']+=10
        other.store.ingest(sample,'other')
        report=client.get('/forecast/report',params={'workspace_id':'OTHER','model_version':'2','limit':8})
        assert report.status_code==200,report.text
        assert report.json()['hive_id']=='OTHER' and len(report.json()['rows'])==8
        assert report.json()['next_forecast']['predicted_weight_kg']>60
        assert not primary.model.calls
        response=client.post('/models/rollback',json={'version':'2'})
        assert response.status_code==200,response.text
        result=response.json()
        assert result['version']=='2' and result['activation']['previous_version']=='1'
        assert result['forecast_report']['model']['version']=='2'
        assert result['forecast_report']['model']['is_champion'] is True
        assert result['forecast_report']['next_forecast']['predicted_delta_kg']==pytest.approx(.2,abs=1e-5)
        assert primary.store.predictions()==[]


def test_successful_activation_stays_successful_when_report_or_logging_fails(tmp_path,monkeypatch):
    model=VersionedForecast()
    with TestClient(create_app(tmp_path,model=model)) as client:
        service=client.app.state.service; service.store.ingest(rows(50),'fixture')
        model.fail=True
        monkeypatch.setattr(service.store,'log',lambda *args,**kwargs: (_ for _ in ()).throw(OSError('log disk error')))
        response=client.post('/models/rollback',json={'version':'2'})
        assert response.status_code==200,response.text
        result=response.json()
        assert result['status']=='partial' and result['activation']['version']=='2'
        assert result['forecast_report'] is None and 'artifact unavailable' in result['report_error']
        assert model.champion=='2'


def test_prediction_button_can_use_v2_without_changing_deployed_model(tmp_path):
    with TestClient(create_app(tmp_path,model=VersionedForecast(),model_factory=lambda _:VersionedForecast())) as client:
        primary=client.app.state.service
        primary.store.ingest(rows(50),'primary')
        other,_=client.app.state.workspaces.ensure('OTHER')
        sample=rows(50)
        for row in sample: row['hive_id']='OTHER'; row['weight_kg']+=10
        other.store.ingest(sample,'other')
        outputs=[]
        for version in ('1','2'):
            response=client.post('/predict',params={'workspace_id':'OTHER','model_version':version},
                                 json={'sequence':sample[-24:]})
            assert response.status_code==200,response.text
            result=response.json(); outputs.append(result)
            assert result['model_version']==version
            assert result['hive_id']=='OTHER'
            report=client.get('/forecast/report',params={'workspace_id':'OTHER','model_version':version}).json()
            assert result['predicted_weight_kg']==pytest.approx(report['next_forecast']['predicted_weight_kg'])
            assert result['target_timestamp']==report['next_forecast']['target_timestamp']
        assert outputs[1]['predicted_weight_kg']-outputs[0]['predicted_weight_kg']==pytest.approx(.1,abs=1e-5)
        assert {p['model_version'] for p in other.store.predictions()}=={'1','2'}
        assert other.model.champion==primary.model.champion=='1'
        assert primary.store.predictions()==[] and not primary.model.calls


def test_prediction_cannot_record_another_version_as_the_selected_result(tmp_path):
    model=VersionedForecast(); model.wrong_version=True
    with TestClient(create_app(tmp_path,model=model)) as client:
        service=client.app.state.service; service.store.ingest(rows(24),'fixture')
        before=database_dump(service)
        response=client.post('/predict',params={'model_version':'2'},json={'sequence':rows(24)})
        assert response.status_code==503,response.text
        assert 'different requested version' in response.json()['detail']
        assert database_dump(service)==before
