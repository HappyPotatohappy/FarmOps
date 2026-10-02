"""Future forecasts keep one immutable observation/model identity across decisions."""
from copy import deepcopy
from datetime import datetime, timedelta
import pytest
from fastapi.testclient import TestClient
from app.main import create_app
from test_api import ForecastFixture
from test_store import rows


class HorizonFixture:
    def __init__(self): self.calls=[]; self.fail=False; self.wrong=False
    def catalog(self):
        return {'registry_id':'BeeOPS_Horizon_Weight','default_version':'1','versions':[
            {'version':v,'run_id':'horizon-'+v,'context_hours':24,'max_horizon_hours':168,
             'components':['lstm','lightgbm'],'weights':{'lstm':.25,'lightgbm':.75}}
            for v in ('1','2')]}
    def predict(self, data, horizon_hours, version=None):
        if self.fail: raise RuntimeError('broken horizon artifact')
        self.calls.append((deepcopy(data),horizon_hours,version))
        selected=version or '1'
        if selected not in ('1','2'):
            return {'status':'model_unavailable','model':{'version':selected},'trajectory':[],
                    'validation':{'status':'insufficient','calibrated':False},'reasons':['모델 없음']}
        model=deepcopy(self.catalog()['versions'][int(selected)-1])
        model['registry_id']='BeeOPS_Horizon_Weight'
        if self.wrong: model['version']='99'
        result={'status':'ok','model':model,'trajectory':[],
                'validation':{'status':'underperforming','calibrated':False,'noise_kg':.1},'reasons':[]}
        if len(data)<24:
            result.update(status='insufficient_history',reasons=['24시간 관측 필요'])
            return result
        origin=datetime.fromisoformat(data[-1]['timestamp'])
        result['trajectory']=[{'timestamp':(origin+timedelta(hours=h)).isoformat(),
                              'weight_kg':data[-1]['weight_kg']+.01*h,'lower_kg':None,'upper_kg':None}
                             for h in range(1,horizon_hours+1)]
        return result


@pytest.fixture
def horizon_client(tmp_path):
    provider=HorizonFixture()
    app=create_app(tmp_path,model=ForecastFixture(),model_factory=lambda _:ForecastFixture(),horizon_model=provider)
    with TestClient(app) as client: yield client,provider


def dump(store):
    with store.db() as db: return list(db.iterdump())


def test_same_origin_forecast_and_harvest_identity_without_legacy_writes(horizon_client):
    client,provider=horizon_client
    store=client.app.state.service.store
    store.ingest(rows(80),'fixture')
    before=dump(store)
    args={'horizon_hours':168,'model_version':'2'}
    response=client.get('/forecast/horizon',params=args)
    assert response.status_code==200,response.text
    forecast=response.json()
    assert response.headers['cache-control']=='no-store'
    assert forecast['as_of']==rows(80)[-1]['timestamp']
    assert forecast['historical'] is True
    assert len(forecast['trajectory'])==168
    assert forecast['trajectory'][0]['timestamp']==rows(1,start=80)[0]['timestamp']
    report=client.get('/harvest/report',params=args).json()
    for key in ('forecast_id','workspace_id','snapshot_id','as_of','horizon_hours','requested_horizon_hours'):
        assert report[key]==forecast[key]==report['forecast'][key]
    assert report['candidate_window'] is not None
    assert report['status']=='inspection_window'
    assert report['field_checks']=={'maturity':'unknown','reserves':'unknown'}
    assert len(provider.calls)==1
    assert dump(store)==before


@pytest.mark.parametrize('hours',[1,24,72,168])
def test_valid_forecast_recommends_without_accuracy_or_species_eligibility(horizon_client,hours):
    client,_=horizon_client
    client.app.state.service.store.ingest(rows(50),'fixture')
    report=client.get('/harvest/report',params={'horizon_hours':hours}).json()
    assert report['forecast']['validation']['status']=='underperforming'
    assert report['forecast']['validation']['calibrated'] is False
    assert report['status']=='inspection_window'
    window=report['candidate_window']
    assert report['forecast']['trajectory'][0]['timestamp']<=window['start']<=window['end']
    assert window['end']<=report['forecast']['trajectory'][-1]['timestamp']


def test_workspace_horizon_and_snapshot_isolation(horizon_client):
    client,provider=horizon_client
    primary=client.app.state.service.store
    primary.ingest(rows(50),'primary')
    other,_=client.app.state.workspaces.ensure('OTHER')
    sample=rows(50)
    for r in sample: r['hive_id']='OTHER'; r['weight_kg']+=10
    other.store.ingest(sample,'other')
    first=client.get('/forecast/horizon',params={'horizon_hours':24}).json()
    second=client.get('/forecast/horizon',params={'horizon_hours':24,'workspace_id':'OTHER'}).json()
    assert second['workspace_id']==second['hive_id']=='OTHER'
    assert second['trajectory'][0]['weight_kg']>60
    assert first['forecast_id']!=second['forecast_id']
    third=client.get('/forecast/horizon',params={'horizon_hours':72}).json()
    assert third['forecast_id']!=first['forecast_id']
    primary.ingest(rows(1,start=50),'append')
    fourth=client.get('/forecast/horizon',params={'horizon_hours':24}).json()
    assert fourth['snapshot_id']!=first['snapshot_id']
    assert fourth['forecast_id']!=first['forecast_id']
    assert client.get('/forecast/horizon',params={'workspace_id':'MISSING'}).status_code==404


def test_unavailable_invalid_horizon_and_inference_failure_are_explicit(horizon_client):
    client,provider=horizon_client
    empty=client.get('/harvest/report').json()
    assert empty['candidate_window'] is None
    assert empty['forecast']['status']=='insufficient_history'
    assert client.get('/forecast/horizon',params={'horizon_hours':48}).status_code==422
    assert client.get('/forecast/horizon',params={'model_version':'../../bad'}).status_code==422
    unknown=client.get('/forecast/horizon',params={'model_version':'99'}).json()
    assert unknown['status']=='model_unavailable' and unknown['trajectory']==[]
    client.app.state.service.store.ingest(rows(50),'fixture')
    provider.fail=True
    failed=client.get('/forecast/horizon')
    assert failed.status_code==503 and 'artifact' in failed.text


def test_provider_cannot_substitute_model_or_return_invalid_path(horizon_client):
    client,provider=horizon_client
    client.app.state.service.store.ingest(rows(50),'fixture')
    provider.wrong=True
    assert client.get('/forecast/horizon',params={'model_version':'2'}).status_code==503
    provider.wrong=False
    original=provider.predict
    def broken(*args,**kwargs):
        result=original(*args,**kwargs)
        result['trajectory'][0]['timestamp']=rows(1)[0]['timestamp']
        return result
    provider.predict=broken
    assert client.get('/forecast/horizon').status_code==503


def test_catalog_is_distinct_from_existing_models(horizon_client):
    client,provider=horizon_client
    result=client.get('/models/horizon')
    assert result.status_code==200
    assert result.json()['registry_id']=='BeeOPS_Horizon_Weight'
    assert client.get('/models').json()==[]


def test_nonfinite_metadata_is_reported_as_model_failure(horizon_client):
    client,provider=horizon_client
    client.app.state.service.store.ingest(rows(50),'fixture')
    original=provider.predict
    def broken(*args,**kwargs):
        result=original(*args,**kwargs)
        result['validation']['noise_kg']=float('nan')
        return result
    provider.predict=broken
    assert client.get('/forecast/horizon').status_code==503


@pytest.mark.parametrize('change',['missing_version','unlisted_version','list_envelope','list_model','list_point','invalid_status'])
def test_malformed_or_unlisted_provider_identity_is_a_503(horizon_client,change):
    client,provider=horizon_client
    client.app.state.service.store.ingest(rows(50),'fixture')
    original=provider.predict
    def broken(*args,**kwargs):
        result=original(*args,**kwargs)
        if change=='missing_version': result['model'].pop('version')
        elif change=='unlisted_version': result['model']['version']='9'; result['model']['run_id']='horizon-9'
        elif change=='list_envelope': return []
        elif change=='list_model': result['model']=[]
        elif change=='list_point': result['trajectory'][0]=[]
        elif change=='invalid_status': result['status']='made_up'
        return result
    provider.predict=broken
    query={'model_version':'9'} if change=='unlisted_version' else {}
    # Force a complete but unregistered successful forecast for version 9.
    if change=='unlisted_version':
        def unlisted(*args,**kwargs):
            result=original(args[0],args[1],version='1')
            result['model'].update(version='9',run_id='horizon-9')
            return result
        provider.predict=unlisted
    assert client.get('/forecast/horizon',params=query).status_code==503


def test_transient_unavailable_response_recovers_without_new_observation(horizon_client):
    client,provider=horizon_client
    client.app.state.service.store.ingest(rows(50),'fixture')
    original=provider.predict
    provider.predict=lambda *a,**kw:{'status':'model_unavailable','model':{'version':'1'},
        'trajectory':[],'validation':{'status':'insufficient','calibrated':False},'reasons':['artifact offline']}
    assert client.get('/forecast/horizon').json()['status']=='model_unavailable'
    provider.predict=original
    assert client.get('/forecast/horizon').json()['status']=='ok'


def test_public_species_requires_verified_recent_decision_context():
    import csv
    from pathlib import Path
    from app.horizon_service import species_for
    source=list(csv.DictReader((Path(__file__).resolve().parents[2]/'data/real_hive.csv').open()))
    assert species_for('ufc_apis_2',source[:96])=='apis'
    fake=deepcopy(source[:24])
    origin=datetime.fromisoformat(fake[-1]['timestamp'])
    for hour in range(1,73):
        fake.append({**fake[-1],'timestamp':(origin+timedelta(days=100,hours=hour)).isoformat()})
    assert species_for('ufc_apis_2',fake)=='unknown'


@pytest.mark.parametrize('boundary',['catalog','predict'])
def test_corrupt_artifact_is_server_unavailable_not_user_validation(horizon_client,boundary):
    client,provider=horizon_client
    client.app.state.service.store.ingest(rows(50),'fixture')
    def corrupt(*args,**kwargs): raise ValueError('invalid artifact metadata')
    setattr(provider,boundary,corrupt)
    path='/models/horizon' if boundary=='catalog' else '/forecast/horizon'
    assert client.get(path).status_code==503
