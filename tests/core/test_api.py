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
