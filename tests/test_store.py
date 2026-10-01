from datetime import datetime, timedelta, timezone
import pytest
from pydantic import ValidationError
from app.contracts import Observation, PredictRequest, validate_series
from app.store import Store


def rows(n=48, start=0, event='normal'):
    return [dict(timestamp=(datetime(2025,1,1,tzinfo=timezone.utc)+timedelta(hours=i)).isoformat(),
                 hive_id='BEE-01', weight_kg=50+i*.01, temperature_c=20, event=event)
            for i in range(start,start+n)]


@pytest.mark.parametrize('value',[float('nan'),float('inf'),-1,0,301])
def test_unsafe_weight_rejected(value):
    row=rows(1)[0]; row['weight_kg']=value
    with pytest.raises(ValidationError): Observation(**row)


def test_window_rejects_gaps_mixed_hives_and_naive_time():
    sample=rows(24); sample[5]['timestamp']=sample[4]['timestamp']
    with pytest.raises(ValueError): validate_series(sample, minimum=24)
    sample=rows(24); sample[5]['hive_id']='BEE-02'
    with pytest.raises(ValueError): PredictRequest(sequence=sample)
    sample=rows(1)[0]; sample['timestamp']='2025-01-01T00:00:00'
    with pytest.raises(ValidationError): Observation(**sample)


def test_delayed_actual_join_idempotency_and_restart(tmp_path):
    store=Store(tmp_path)
    store.ingest(rows(24), source='test')
    p=store.record_prediction(rows(24),50.2,'1')
    assert p['actual_weight_kg'] is None
    store.ingest(rows(1,start=24),source='test')
    again=store.record_prediction(rows(24),50.2,'1')
    assert again['prediction_id']==p['prediction_id']
    store.ingest(rows(1,start=24),source='test')
    history=Store(tmp_path).predictions()
    assert len(history)==1 and history[0]['actual_weight_kg']==50.24
    assert history[0]['error_kg']==pytest.approx(.04)


def test_conflicting_actual_is_atomic_and_does_not_erase_original(tmp_path):
    store=Store(tmp_path); store.ingest(rows(24),source='test')
    changed=rows(25); changed[0]['weight_kg']=90
    with pytest.raises(ValueError): store.ingest(changed,source='test')
    assert store.data_status()['rows']==24
    assert store.observations()[0]['weight_kg']==50


def test_event_and_sudden_change_are_excluded_from_quality(tmp_path):
    store=Store(tmp_path); store.ingest(rows(24),source='test')
    store.record_prediction(rows(24),50.2,'1')
    event=rows(1,start=24,event='harvest'); event[0]['weight_kg']=45
    store.ingest(event,source='test')
    assert store.quality('1')['count']==0
    assert store.alerts()[0]['category']=='operation'


def test_unlabelled_jump_is_quarantined(tmp_path):
    store=Store(tmp_path); store.ingest(rows(24),source='test')
    store.record_prediction(rows(24),50.2,'1')
    jump=rows(1,start=24); jump[0]['weight_kg']=47
    store.ingest(jump,source='test')
    assert store.quality('1')['count']==0
    assert store.observations()[-1]['event']=='colony_alert'


def test_quality_window_uses_only_requested_version(tmp_path):
    store=Store(tmp_path); store.ingest(rows(50),source='test')
    for i in range(24,49): store.record_prediction(rows(24,start=i-24),49,'1')
    store.record_prediction(rows(24,start=25),50,'2')
    q=store.quality('1'); assert q['count']==24 and q['status']=='degraded'
    assert store.quality('2')['count']==1


def test_prediction_requires_stored_matching_context(tmp_path):
    store=Store(tmp_path)
    with pytest.raises(ValueError): store.record_prediction(rows(24),50,'1')
    store.ingest(rows(24),source='test')
    changed=rows(24); changed[0]['weight_kg']=51
    with pytest.raises(ValueError): store.record_prediction(changed,50,'1')


def test_quarantined_observations_can_be_replayed_idempotently(tmp_path):
    store=Store(tmp_path)
    sample=rows(25); sample[-1]['weight_kg']=45
    store.ingest(sample,source='test')
    replay=store.observations()
    assert store.ingest(replay,source='replay')['inserted']==0


def test_csv_float_roundtrip_preserves_original_without_false_conflict(tmp_path):
    store=Store(tmp_path)
    sample=rows(24); sample[0]['weight_kg']=50.04528427000001
    store.ingest(sample,source='pandas')
    replay=rows(24); replay[0]['weight_kg']=50.04528427
    assert store.ingest(replay,source='csv')['duplicate']==24
    assert store.observations()[0]['weight_kg']==sample[0]['weight_kg']
    prediction=store.record_prediction(replay,50,'1')
    assert prediction['context_valid']==1
    replay[0]['weight_kg']+=0.000001
    with pytest.raises(ValueError): store.ingest(replay,source='changed')


def test_ingest_preview_matches_atomic_commit_without_writing(tmp_path):
    store=Store(tmp_path); store.ingest(rows(24),source='original')
    assert store.preview_ingest(rows(30))=={'inserted':6,'duplicate':24,'rows':30}
    assert store.data_status()['rows']==24
    bad=rows(30); bad[2]['weight_kg']+=1
    with pytest.raises(ValueError): store.preview_ingest(bad)
    assert store.data_status()['rows']==24
    result=store.ingest(rows(30),source='appended')
    assert result['inserted']==6 and result['duplicate']==24


def test_version_cutoff_excludes_legacy_predictions_from_monitoring(tmp_path):
    store=Store(tmp_path); store.ingest(rows(60),source='fixture')
    for i in range(24,60): store.record_prediction(rows(24,start=i-24),49,'1')
    assert store.quality('1')['count']==24
    store.set_model_cutoff('1',rows(1,start=50)[0]['timestamp'])
    assert store.quality('1')['count']==9
    assert sum(p['monitoring_eligible'] for p in store.predictions())==9
    assert Store(tmp_path).quality('1')['count']==9


def test_shared_v1_predictions_and_cutoffs_do_not_reuse_or_overwrite_legacy_v1(tmp_path):
    legacy=Store(tmp_path); legacy.ingest(rows(30),'original')
    old=legacy.record_prediction(rows(24),10,'1')
    legacy.set_model_cutoff('1',rows(1,start=29)[0]['timestamp'])
    snapshot=legacy.data_status()['snapshot_id']
    shared=Store(tmp_path,prediction_namespace='shared')
    shared.set_model_cutoff('1',rows(1,start=23)[0]['timestamp'])
    fresh=shared.record_prediction(rows(24),50.2,'1')
    assert fresh['prediction_id']!=old['prediction_id']
    assert fresh['predicted_weight_kg']==50.2
    assert shared.quality('1')['count']==1 and legacy.quality('1')['count']==0
    assert legacy.predictions()[0]['predicted_weight_kg']==10
    assert shared.data_status()['snapshot_id']==snapshot
    reopened=Store(tmp_path,prediction_namespace='shared')
    assert reopened.record_prediction(rows(24),50.2,'1')['prediction_id']==fresh['prediction_id']
    assert Store(tmp_path).predictions()[0]['prediction_id']==old['prediction_id']
