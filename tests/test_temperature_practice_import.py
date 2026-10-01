"""Built-in heatwave import must recover its real baseline, then queue one job."""
import csv
import io
from pathlib import Path
import time

from fastapi.testclient import TestClient
import pytest

from app.contracts import validate_series
from app.main import create_app
from app.store import Store
from test_api import ForecastFixture
from test_imports import csv_bytes, wait_job
from test_temperature_drift import coordinator

ROOT=Path(__file__).resolve().parents[1]/'data/simulations/temperature_practice'
HIVE='TEMP-DRIFT-PRACTICE'


def rows(filename):
    return validate_series(list(csv.DictReader(io.StringIO((ROOT/filename).read_text()))))


@pytest.fixture
def practice(tmp_path,monkeypatch):
    monitor,provider=coordinator(tmp_path/'monitor')
    def train(job,observations,reference,progress):
        # Replace only expensive model fitting; real persistence/queue/deployment state runs.
        assert len(observations)==504
        assert reference['end']=='2026-07-14T23:00:00+00:00'
        for stage in ('train','register','deploy'): progress(stage)
        parent=provider.version
        provider.version=str(int(parent)+1)
        return {'new_version':provider.version,'parent_version':parent,'promoted':True}
    monkeypatch.setattr(monitor,'_train_bundle',train)
    app=create_app(tmp_path/'api',model=ForecastFixture(),model_factory=lambda _:ForecastFixture(),temperature_monitor=monitor)
    with TestClient(app,raise_server_exceptions=False) as client:
        yield client,monitor


def preview(client):
    response=client.post('/data/simulations/temperature/preview',json={'filename':'02_heatwave.csv'})
    assert response.status_code==200,response.text
    return response.json()


def commit(client,value):
    response=client.post('/imports/commit',json={'preview_id':value['preview_id']})
    assert response.status_code==202,response.text
    return wait_job(client.app.state.imports,response.json()['job_id'])


def terminal(monitor):
    deadline=time.monotonic()+5
    while time.monotonic()<deadline:
        job=monitor.status(HIVE)['job']
        if job and job['status'] not in ('queued','running'): return job
        time.sleep(.01)
    raise AssertionError('retraining did not finish')


def test_heatwave_preview_is_read_only_and_commit_trains_using_included_baseline(practice):
    client,monitor=practice
    before=client.get('/workspaces').json()
    result=preview(client)
    assert result['can_commit'] and result['row_count']==504 and result['insert_count']==504
    assert result['temperature_practice']['phase']=='heatwave'
    assert result['temperature_practice']['baseline_rows']==336
    assert client.get('/workspaces').json()==before
    assert monitor.status(HIVE)['reference'] is None
    assert monitor.status(HIVE)['jobs']==[]
    job=commit(client,result)
    assert job['result']['ingest']['inserted']==504
    assert job['result']['retraining_requested'] is True
    trained=terminal(monitor)
    assert trained['status']=='completed'
    assert job['result']['retraining_job_id']==trained['job_id']
    assert trained['check']['status']=='drift'
    assert trained['check']['mean_shift_c']>7


def test_heat_only_existing_import_is_recovered_then_new_preview_retrains_again(practice):
    client,monitor=practice
    generic=client.post('/imports/preview',files={'file':('02_heatwave.csv',(ROOT/'02_heatwave.csv').read_bytes(),'text/csv')}).json()
    assert 'temperature_practice' not in generic and generic['row_count']==168
    first=commit(client,generic)
    assert first['result'].get('retraining_requested',False) is False
    assert monitor.status(HIVE)['job'] is None
    assert monitor.status(HIVE)['reference']['mean_c']>31
    original=client.app.state.workspaces.get(HIVE).store.observations()
    result=preview(client)
    assert result['can_commit'] and result['insert_count']==336 and result['duplicate_count']==168
    assert client.app.state.workspaces.get(HIVE).store.observations()==original
    recovered=commit(client,result)
    assert recovered['result']['ingest']['inserted']==336
    assert recovered['result']['ingest']['duplicate']==168
    assert recovered['result']['retraining_requested'] is True
    assert terminal(monitor)['status']=='completed'
    store=client.app.state.workspaces.get(HIVE).store
    assert store.observations()[336:]==original
    with store.db() as db:
        assert {row[0] for row in db.execute('SELECT source FROM observations WHERE timestamp>=?',(original[0]['timestamp'],))}=={'사용자 CSV: 02_heatwave.csv'}
    repeated=commit(client,preview(client))
    assert repeated['result']['retraining_requested'] is True
    assert repeated['result']['retraining_status']=='queued'
    assert repeated['result']['retraining_job_id']!=recovered['result']['retraining_job_id']
    assert repeated['result']['ingest']['inserted']==0
    assert terminal(monitor)['status']=='completed'
    assert terminal(monitor)['new_version']=='4'
    assert len(monitor.status(HIVE)['jobs'])==2


def test_generic_upload_named_heatwave_cannot_recover_missing_prefix(practice):
    client,monitor=practice
    hot=rows('02_heatwave.csv')
    selected,_=client.app.state.workspaces.ensure(HIVE)
    selected.store.ingest(hot,'original')
    result=client.post('/imports/preview',files={'file':('02_heatwave.csv',csv_bytes(rows('01_baseline.csv')+hot),'text/csv')}).json()
    assert not result['can_commit'] and 'Backdated' in str(result['errors'])
    assert 'temperature_practice' not in result
    assert selected.store.observations()==hot


def test_trusted_preview_conflicts_fail_without_rewriting_arbitrary_user_values(practice):
    client,monitor=practice
    hot=rows('02_heatwave.csv');hot[2]['weight_kg']+=.1
    selected,_=client.app.state.workspaces.ensure(HIVE)
    selected.store.ingest(hot,'user data')
    result=preview(client)
    assert not result['can_commit'] and result['conflict_count']==1
    assert client.post('/imports/commit',json={'preview_id':result['preview_id']}).status_code==422
    assert selected.store.observations()==hot and monitor.status(HIVE)['reference'] is None


def test_changed_ledger_after_preview_is_revalidated_atomically(practice):
    client,monitor=practice
    selected,_=client.app.state.workspaces.ensure(HIVE)
    hot=rows('02_heatwave.csv');selected.store.ingest(hot,'original')
    result=preview(client)
    changed=rows('03_followup.csv')[:1];changed[0]['temperature_c']+=1
    selected.store.ingest(changed,'user addition')
    before=selected.store.observations()
    job=commit(client,result)
    assert job['status']=='failed'
    assert selected.store.observations()==before
    assert monitor.status(HIVE)['job'] is None


@pytest.mark.parametrize('filename',['../02_heatwave.csv','unknown.csv','control/02_normal.csv'])
def test_practice_preview_accepts_only_supported_built_in_phases(practice,filename):
    client,_=practice
    assert client.post('/data/simulations/temperature/preview',json={'filename':filename}).status_code==422


def test_explicit_store_recovery_flag_does_not_accept_arbitrary_prefix(tmp_path):
    store=Store(tmp_path/'hive');hot=rows('02_heatwave.csv');store.ingest(hot,'original')
    combined=rows('01_baseline.csv')+hot;combined[1]['weight_kg']+=.1
    with pytest.raises(ValueError,match='built-in|canonical|practice'):
        store.ingest(combined,'attempt',temperature_practice=True)
    assert store.observations()==hot


def test_baseline_first_practice_still_queues_only_after_heatwave(practice):
    client,monitor=practice
    baseline=client.post('/data/simulations/temperature/preview',json={'filename':'01_baseline.csv'}).json()
    assert baseline['can_commit'] and baseline['row_count']==336
    initial=commit(client,baseline)
    assert initial['result']['retraining_requested'] is False
    assert monitor.status(HIVE)['job'] is None
    assert monitor.status(HIVE)['reference']['end']=='2026-07-14T23:00:00+00:00'
    result=preview(client)
    assert result['insert_count']==168 and result['duplicate_count']==336
    trained=commit(client,result)
    assert trained['result']['retraining_requested'] is True
    assert terminal(monitor)['status']=='completed'


def test_repeated_heatwave_during_running_job_preserves_reference_and_job(practice,monkeypatch):
    import threading
    client,monitor=practice
    entered=threading.Event();release=threading.Event()
    original=monitor._train_bundle
    def paused(*args):
        entered.set()
        assert release.wait(10)
        return original(*args)
    monkeypatch.setattr(monitor,'_train_bundle',paused)
    try:
        first=commit(client,preview(client))
        assert entered.wait(5)
        state=monitor.status(HIVE)
        assert state['job']['status']=='running'
        repeat=commit(client,preview(client))
        after=monitor.status(HIVE)
        assert repeat['result']['retraining_requested'] is False
        assert repeat['result']['retraining_job_id']==first['result']['retraining_job_id']
        assert repeat['result']['retraining_status']=='already_running'
        assert after['reference']==state['reference']
        assert after['job']['job_id']==first['result']['retraining_job_id']
        assert len(after['jobs'])==1
    finally:
        release.set()
    assert terminal(monitor)['status']=='completed'


def test_same_preview_commit_is_idempotent_even_after_training_completed(practice):
    client,monitor=practice
    pending=preview(client)
    first=commit(client,pending)
    assert terminal(monitor)['status']=='completed'
    again=commit(client,pending)
    assert again['job_id']==first['job_id']
    assert again['result']['retraining_job_id']==first['result']['retraining_job_id']
    assert again['result']['retraining_request_id']==first['job_id']
    assert len(monitor.status(HIVE)['jobs'])==1


def test_explicit_replay_job_records_request_and_no_new_data_claim(practice):
    client,monitor=practice
    first=commit(client,preview(client));terminal(monitor)
    second=commit(client,preview(client));job=terminal(monitor)
    assert job['reason']=='explicit_temperature_practice'
    assert job['practice_replay']['request_id']==second['job_id']
    assert job['practice_replay']['inserted_observations']==0
    assert job['practice_replay']['reused_observations']==504
    assert job['practice_replay']['data_kind']=='synthetic'
    assert second['result']['retraining_job_id']==job['job_id']
    assert first['result']['retraining_job_id']!=job['job_id']


def test_generic_reupload_after_practice_completion_does_not_request_replay(practice):
    client,monitor=practice
    first=commit(client,preview(client));terminal(monitor)
    generic=client.post('/imports/preview',files={'file':('02_heatwave.csv',(ROOT/'02_heatwave.csv').read_bytes(),'text/csv')}).json()
    result=commit(client,generic)
    assert result['result']['retraining_requested'] is False
    assert result['result']['retraining_job_id'] is None
    assert len(monitor.status(HIVE)['jobs'])==1
    assert monitor.status(HIVE)['job']['job_id']==first['result']['retraining_job_id']


def test_explicit_practice_cutoff_reuses_baseline_even_after_consumed_targets():
    from app.temperature_drift import training_cutoff,prepare_training,reference_for
    observed=rows('01_baseline.csv')+rows('02_heatwave.csv')
    reference=reference_for(observed[168:336],'4','fixed practice baseline')
    parent={'context_hours':168,'temperature_training':{'consumed_through_by_hive':{HIVE:'2026-07-24T23:00:00+00:00'}}}
    cutoff=training_cutoff(observed,parent,HIVE,reference,practice_replay=True)
    assert cutoff=='2026-07-14T23:00:00+00:00'
    prepared=prepare_training(observed,cutoff,168,168)
    assert len(prepared['train'])==120 and len(prepared['calibration'])==24 and len(prepared['validation'])==24
    assert training_cutoff(observed,parent,HIVE,reference)=='2026-07-24T23:00:00+00:00'
    observed[0]['temperature_c']+=1
    with pytest.raises(ValueError,match='canonical|practice'):
        training_cutoff(observed,parent,HIVE,reference,practice_replay=True)


def test_exact_retraining_job_endpoint_survives_recent_list_window(practice):
    from copy import deepcopy
    client,monitor=practice
    imported=commit(client,preview(client));first=terminal(monitor)
    with monitor.lock:
        for index in range(11):
            old=deepcopy(first);old['job_id']=f'later-{index}'
            monitor.state['jobs'].append(old)
    response=client.get('/monitoring/retraining/jobs/'+first['job_id'])
    assert response.status_code==200,response.text
    assert response.json()['job_id']==first['job_id']
    assert response.json()['status']=='completed'
    assert client.get('/monitoring/retraining/jobs/missing').status_code==404
    assert client.get('/monitoring/retraining').status_code==200


def test_replay_keeps_tolerated_stored_numeric_roundoff_in_reference():
    from app.temperature_drift import training_cutoff,reference_for
    observed=rows('01_baseline.csv')+rows('02_heatwave.csv')
    observed[170]['weight_kg']+=1e-12
    reference=reference_for(observed[168:336],'4','fixed practice baseline')
    assert training_cutoff(observed,{'context_hours':168},HIVE,reference,practice_replay=True)=='2026-07-14T23:00:00+00:00'


def test_replay_completion_preserves_a_newer_operational_reference(practice):
    from copy import deepcopy
    from app.temperature_drift import reference_for
    client,monitor=practice
    commit(client,preview(client));terminal(monitor)
    service=client.app.state.workspaces.get(HIVE)
    followup=rows('03_followup.csv');service.store.ingest(followup,'already observed followup')
    later=reference_for(followup[-24:],'3','post_deployment_recent_24_normal_hours')
    with monitor.lock:
        monitor.state['workspaces'][HIVE]['reference']=later
        monitor.state['workspaces'][HIVE]['check']={'status':'waiting','available_hours':0}
        monitor._save()
    before=deepcopy(monitor.status(HIVE))
    replay=commit(client,preview(client));job=terminal(monitor)
    assert replay['result']['retraining_requested'] is True
    assert job['status']=='completed'
    assert monitor.status(HIVE)['reference']==before['reference']
    assert monitor.status(HIVE)['check']==before['check']
    assert service.store.observations()[-1]['timestamp']=='2026-07-24T23:00:00+00:00'
