import json
import time
import numpy as np
import csv
import io
import pytest
from fastapi.testclient import TestClient
from app.main import create_app
from test_store import rows


class ForecastFixture:
    """Cheap deterministic inference boundary; real registry tested in test_ml."""
    def status(self): return {'ready':True,'version':'1','model_source':'fixture','minimum_training_rows':168,'consumed_through':'2025-01-01T23:00:00+00:00'}
    def predict(self,windows,version=None): return windows[:,-1,0]+.01,version or '1'
    def versions(self): return []


@pytest.fixture
def client(tmp_path):
    app=create_app(tmp_path,model=ForecastFixture())
    with TestClient(app,raise_server_exceptions=False) as c: yield c


def test_prediction_observation_join_through_actual_api(client):
    assert client.post('/observations',json={'observations':rows(24)}).status_code==200
    prediction=client.post('/predict',json={'sequence':rows(24)})
    assert prediction.status_code==200
    assert prediction.json()['actual_weight_kg'] is None
    client.post('/observations',json={'observations':rows(1,start=24)})
    assert client.get('/predictions').json()[0]['error_kg']==pytest.approx(0,abs=1e-5)


def test_invalid_length_nonfinite_and_mixed_hive_are_422(client):
    assert client.post('/predict',json={'sequence':rows(23)}).status_code==422
    data=rows(24); data[-1]['hive_id']='OTHER'
    assert client.post('/predict',json={'sequence':data}).status_code==422
    import json
    data=rows(24); data[0]['weight_kg']=float('inf')
    response=client.post('/predict',content=json.dumps({'sequence':data}),headers={'Content-Type':'application/json'})
    assert response.status_code==422


def test_bad_upload_does_not_destroy_existing_data(client):
    client.post('/observations',json={'observations':rows(24)})
    r=client.post('/data/upload',files={'file':('bad.csv',b'timestamp,weight_kg\nwrong,-5\n','text/csv')})
    assert r.status_code==422
    assert client.get('/data/status').json()['rows']==24


def test_batch_persists_actuals_and_does_not_double_count(client):
    r=client.post('/predict/batch-test',json={'observations':rows(50)})
    assert r.status_code==200
    assert r.json()['quality']['count']==24
    assert client.get('/data/status').json()['rows']==50
    client.post('/predict/batch-test',json={'observations':rows(50)})
    assert len(client.get('/predictions').json())==26


def test_cors_and_methods_do_not_mutate_on_get(client):
    assert client.get('/demo/run').status_code in (404,405)
    assert client.get('/health').json()['model_loaded'] is True


@pytest.mark.parametrize('hive_id,synthetic,kind',[
    ('BEE-DEMO',True,'synthetic'),('ufc_apis_2',False,'public_real'),('OTHER',False,'single_hive')])
def test_config_identifies_the_upload_environment(client,monkeypatch,hive_id,synthetic,kind):
    sample=rows(24)
    for row in sample: row['hive_id']=hive_id
    client.post('/observations',json={'observations':sample})
    client.app.state.service.store.set_meta('demo_runtime',synthetic)
    monkeypatch.setenv('BEEOPS_PUBLIC_DATA_URL','http://127.0.0.1:8012')
    config=client.get('/config').json()
    assert config['runtime_type']==kind
    assert config['hive_id']==hive_id
    assert config['public_data_dashboard_url']=='http://127.0.0.1:8012'


@pytest.mark.parametrize('incoming,configured,expected_url',[
    ('ufc_apis_2','http://127.0.0.1:8012','http://127.0.0.1:8012'),
    ('OTHER','http://127.0.0.1:8012',None),
    ('ufc_apis_2','',None)])
def test_csv_hive_mismatch_has_recovery_guidance_without_changing_data(client,monkeypatch,incoming,configured,expected_url):
    monkeypatch.setenv('BEEOPS_PUBLIC_DATA_URL',configured)
    client.post('/observations',json={'observations':rows(24)})
    before=client.get('/data/status').json()
    sample=rows(24)
    for row in sample: row['hive_id']=incoming
    csv_text=io.StringIO(); writer=csv.DictWriter(csv_text,fieldnames=list(sample[0]))
    writer.writeheader(); writer.writerows(sample)
    response=client.post('/data/upload',files={'file':('real.csv',csv_text.getvalue(),'text/csv')})
    assert response.status_code==422
    detail=response.json()['detail']
    assert isinstance(detail,dict),detail
    assert detail['code']=='hive_mismatch'
    assert detail['expected_hive_id']=='BEE-01'
    assert detail['received_hive_id']==incoming
    assert detail['upload_dashboard_url']==expected_url
    assert 'BEE-01' in detail['message'] and incoming in detail['message']
    assert client.get('/data/status').json()==before


def test_public_sample_csv_serves_only_catalogued_files(client):
    from app.public_data import sample_catalog
    sample_id=sample_catalog()['default_sample_id']
    response=client.get(f'/data/samples/{sample_id}.csv')
    assert response.status_code==200 and response.text.startswith('timestamp')
    assert client.get('/data/samples/unknown.csv').status_code==404
    assert client.get('/data/samples/..%2Fpublic_samples.csv').status_code==404


def test_public_sample_file_refuses_paths_outside_data_root(tmp_path,monkeypatch):
    from app import public_data
    data=tmp_path/'data'; data.mkdir(); (tmp_path/'secret.csv').write_text('x')
    (data/'public_samples.json').write_text(json.dumps({'default_sample_id':'escape',
        'samples':[{'id':'escape','file':'../secret.csv'},{'id':'missing','file':'missing.csv'}]}))
    monkeypatch.setattr(public_data,'DATA_ROOT',data)
    for sample_id in ('escape','missing'):
        with pytest.raises(RuntimeError): public_data.sample_file(sample_id)
    assert public_data.sample_catalog()['samples'][0]=={'id':'escape'}


@pytest.mark.parametrize('filename',['..%2F..%2Fapp%2Fmain.py','control%2F..%2F01_baseline.csv','other.csv'])
def test_temperature_practice_csv_is_allow_listed(client,filename):
    assert client.get('/data/simulations/temperature/'+filename).status_code==404
    assert client.get('/data/simulations/temperature/01_baseline.csv').status_code==200


def await_demo(service,attempts=200):
    for _ in range(attempts):
        if service.demo.get('status')!='running': return service.demo
        time.sleep(.02)
    raise AssertionError('Demo did not finish')


def test_demo_run_requires_synthetic_runtime(client):
    client.post('/observations',json={'observations':rows(24)})
    assert client.post('/demo/run',json={'scenario':'normal'}).status_code==422
    assert client.post('/demo/run',json={'scenario':'unknown'}).status_code==422


@pytest.mark.parametrize('scenario,event',[('colony','colony_alert'),('sensor','sensor_fault')])
def test_demo_run_records_flagged_synthetic_event(client,scenario,event):
    client.post('/observations',json={'observations':rows(24)})
    service=client.app.state.service
    service.store.set_meta('demo_runtime',True)
    response=client.post('/demo/run',json={'scenario':scenario})
    assert response.status_code==202 and response.json()['status']=='running'
    demo=await_demo(service)
    assert demo['status']=='completed',demo
    observations=service.store.observations()
    assert len(observations)==24+48
    assert observations[-1]['event']==event
    assert all(row['hive_id']=='BEE-01' for row in observations)


def test_demo_run_rejects_concurrent_request(client):
    client.post('/observations',json={'observations':rows(24)})
    service=client.app.state.service
    service.store.set_meta('demo_runtime',True)
    service.demo={'status':'running','scenario':'normal'}
    assert client.post('/demo/run',json={'scenario':'normal'}).status_code==409
def test_unregistered_model_version_is_404_not_500(tmp_path):
    from app.service import UnknownModelVersion
    class Registry(ForecastFixture):
        def predict(self,windows,version=None):
            if version=='99': raise UnknownModelVersion('Model version 99 is not registered')
            return super().predict(windows,version)
    with TestClient(create_app(tmp_path,model=Registry()),raise_server_exceptions=False) as c:
        c.post('/observations',json={'observations':rows(24)})
        assert c.post('/predict',params={'model_version':'99'},json={'sequence':rows(24)}).status_code==404
        assert c.get('/forecast/report',params={'model_version':'99'}).status_code==404


def test_csv_upload_ingests_off_the_event_loop(client,monkeypatch):
    import asyncio
    from app.service import Service
    original=Service.ingest; on_loop=[]
    def recording(self,*args,**kwargs):
        try: asyncio.get_running_loop(); on_loop.append(True)
        except RuntimeError: on_loop.append(False)
        return original(self,*args,**kwargs)
    monkeypatch.setattr(Service,'ingest',recording)
    sample=rows(24); csv_text=io.StringIO(); writer=csv.DictWriter(csv_text,fieldnames=list(sample[0]))
    writer.writeheader(); writer.writerows(sample)
    response=client.post('/data/upload',files={'file':('real.csv',csv_text.getvalue(),'text/csv')})
    assert response.status_code==200,response.text
    assert on_loop==[False]
