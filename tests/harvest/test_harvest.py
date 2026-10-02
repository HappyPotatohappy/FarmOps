"""Available forecasts produce recommendation periods without eligibility gates."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import pytest
from app.harvest import recommend_harvest

AS_OF = datetime(2025, 1, 4, tzinfo=timezone.utc)


def observations(days=(48.,49.,50.)):
    return [{'timestamp':(AS_OF-timedelta(hours=len(days)*24-i-1)).isoformat(),
        'hive_id':'apis-test','weight_kg':weight,'temperature_c':25.,'event':'normal'}
        for i,weight in enumerate(weight for day in days for weight in [day]*24)]


def forecast(days=(51.,52.,52.,51.,50.,49.,48.),hours=None):
    values=[weight for day in days for weight in [day]*24]
    if hours is not None: values=values[:hours]
    return {'status':'ok','workspace_id':'apis-test','hive_id':'apis-test',
        'snapshot_id':'snapshot-exact','forecast_id':'forecast-exact',
        'as_of':AS_OF.isoformat(),'generated_at':'2026-10-01T00:00:00+00:00',
        'horizon_hours':len(values),'requested_horizon_hours':len(values),
        'time_basis':'source_clock_unknown','historical':True,
        'model':{'registry_id':'BeeOPS_Horizon_Weight','version':'2','run_id':'run-exact','context_hours':72},
        'validation':{'status':'underperforming','calibrated':False,'noise_kg':None},'reasons':[],
        'trajectory':[{'timestamp':(AS_OF+timedelta(hours=i+1)).isoformat(),
            'weight_kg':weight,'lower_kg':None,'upper_kg':None} for i,weight in enumerate(values)]}


def assert_recommended(result,start=None,end=None):
    assert result['status']=='inspection_window'
    assert result['candidate_window'] is not None
    if start is not None: assert result['candidate_window']=={'start':start,'end':end}
    assert result['actionability']=='requires_field_check'
    assert result['reasons']
    assert result['field_checks']=={'maturity':'unknown','reserves':'unknown'}


def assert_unavailable(result):
    assert result['status']=='forecast_unavailable'
    assert result['candidate_window'] is None
    assert result['actionability']=='none'
    assert result['reasons']


def test_peak_period_preserves_exact_historical_identity_without_mutation():
    predicted,observed=forecast(),observations(); original=deepcopy((predicted,observed))
    result=recommend_harvest(predicted,observed,'apis')
    assert_recommended(result,'2025-01-05T01:00:00+00:00','2025-01-07T00:00:00+00:00')
    for key in ('workspace_id','hive_id','snapshot_id','forecast_id','as_of','generated_at',
        'horizon_hours','requested_horizon_hours','time_basis','historical','model','validation'):
        assert result[key]==predicted[key]
    assert result['evidence']['recent_growth_kg']==pytest.approx(2.)
    assert result['evidence']['forecast_peak_kg']==pytest.approx(52.)
    assert (predicted,observed)==original
    result['model']['version']='changed'
    assert predicted['model']['version']=='2'


@pytest.mark.parametrize('days,start,end',[
    ((51.,52.,53.,54.,55.,56.,57.),'2025-01-10T01:00:00+00:00','2025-01-11T00:00:00+00:00'),
    ((57.,56.,55.,54.,53.,52.,51.),'2025-01-04T01:00:00+00:00','2025-01-05T00:00:00+00:00'),
    ((51.,)*7,'2025-01-04T01:00:00+00:00','2025-01-07T00:00:00+00:00'),
    ((51.,52.,52.,52.,52.,52.,52.),'2025-01-05T01:00:00+00:00','2025-01-08T00:00:00+00:00')])
def test_boundary_flat_and_plateau_paths_still_produce_recommendations(days,start,end):
    assert_recommended(recommend_harvest(forecast(days),[],'unknown'),start,end)


@pytest.mark.parametrize('days,start,end',[
    ((50.,59.1,59.5,60.,59.9,59.8,50.),'2025-01-07T01:00:00+00:00','2025-01-10T00:00:00+00:00'),
    ((50.,59.5,59.5,60.,59.5,59.5,50.),'2025-01-05T01:00:00+00:00','2025-01-08T00:00:00+00:00'),
    ((50.,60.,59.,59.9,59.9,59.9,50.),'2025-01-05T01:00:00+00:00','2025-01-08T00:00:00+00:00')])
def test_broad_peak_uses_best_mean_connected_window_containing_peak_with_early_ties(days,start,end):
    predicted=forecast(days)
    result=recommend_harvest(predicted,[],'unknown')
    assert_recommended(result,start,end)
    assert result['horizon_hours']==result['requested_horizon_hours']==168
    assert len(predicted['trajectory'])==168
    assert len(result['evidence']['daily_forecast'])==7


def test_narrow_peak_is_not_expanded_into_lower_weight_days():
    result=recommend_harvest(forecast((50.,51.,52.,60.,52.,51.,50.)),[],'unknown')
    assert_recommended(result,'2025-01-07T01:00:00+00:00','2025-01-08T00:00:00+00:00')


@pytest.mark.parametrize('hours',[72,73,168])
def test_flat_window_has_exactly_72_inclusive_hourly_points_without_changing_horizon(hours):
    predicted=forecast((51.,)*7,hours=hours)
    result=recommend_harvest(predicted,[],'unknown')
    assert_recommended(result,'2025-01-04T01:00:00+00:00','2025-01-07T00:00:00+00:00')
    window=result['candidate_window']
    selected=[point for point in predicted['trajectory'] if window['start']<=point['timestamp']<=window['end']]
    assert len(selected)==72
    assert datetime.fromisoformat(window['end'])-datetime.fromisoformat(window['start'])==timedelta(hours=71)
    assert result['horizon_hours']==result['requested_horizon_hours']==hours
    assert len(predicted['trajectory'])==hours


def test_capped_recommendations_keep_hives_and_forecast_identity_separate_without_mutation():
    first=forecast((51.,)*7)
    second=forecast((61.,)*7)
    second.update(workspace_id='other-workspace',hive_id='other-hive',snapshot_id='other-snapshot',
                  forecast_id='other-forecast')
    originals=deepcopy((first,second))
    first_result=recommend_harvest(first,[],'apis')
    second_result=recommend_harvest(second,[],'meliponini')
    for predicted,result in ((first,first_result),(second,second_result)):
        for key in ('workspace_id','hive_id','snapshot_id','forecast_id','model','validation'):
            assert result[key]==predicted[key]
    assert first_result['evidence']['forecast_peak_kg']==51.
    assert second_result['evidence']['forecast_peak_kg']==61.
    first_result['model']['version']='changed'
    first_result['validation']['status']='changed'
    first_result['evidence']['daily_forecast'][0]['weight_kg']=1.
    assert second_result['model']['version']=='2'
    assert second_result['evidence']['daily_forecast'][0]['weight_kg']==61.
    assert (first,second)==originals


@pytest.mark.parametrize('hours,start,end',[
    (1,'2025-01-04T01:00:00+00:00','2025-01-04T01:00:00+00:00'),
    (24,'2025-01-04T01:00:00+00:00','2025-01-05T00:00:00+00:00'),
    (72,'2025-01-05T01:00:00+00:00','2025-01-07T00:00:00+00:00'),
    (168,'2025-01-05T01:00:00+00:00','2025-01-07T00:00:00+00:00')])
def test_every_supported_horizon_has_a_recommendation(hours,start,end):
    assert_recommended(recommend_harvest(forecast(hours=hours),[],'unknown'),start,end)


def test_near_peak_days_use_range_tolerance_and_remain_contiguous():
    result=recommend_harvest(forecast((50.,55.,59.,60.,59.1,57.,50.)),[],'apis')
    assert_recommended(result,'2025-01-06T01:00:00+00:00','2025-01-09T00:00:00+00:00')
    assert result['evidence']['tolerance_kg']==pytest.approx(1.)
    separated=recommend_harvest(forecast((51.,52.,50.,52.,51.,50.,49.)),[],'apis')
    assert_recommended(separated,'2025-01-05T01:00:00+00:00','2025-01-06T00:00:00+00:00')


def test_small_forecast_range_uses_absolute_near_peak_tolerance():
    result=recommend_harvest(forecast((50.,50.02,50.04)),[],'unknown')
    assert_recommended(result,'2025-01-04T01:00:00+00:00','2025-01-07T00:00:00+00:00')
    assert result['evidence']['tolerance_kg']==pytest.approx(.05)


def test_hourly_spike_does_not_replace_daily_median_peak():
    predicted=forecast((50.,51.,50.));predicted['trajectory'][0]['weight_kg']=100.
    result=recommend_harvest(predicted,[],'unknown')
    assert_recommended(result,'2025-01-05T01:00:00+00:00','2025-01-06T00:00:00+00:00')
    assert result['evidence']['daily_forecast'][0]['weight_kg']==50.


@pytest.mark.parametrize('species',['apis','meliponini','unknown','synthetic',''])
def test_species_does_not_gate_actual_forecast_recommendation(species):
    assert_recommended(recommend_harvest(forecast(),[],species))


@pytest.mark.parametrize('validation',[None,{}, {'status':'insufficient','calibrated':False},
    {'status':'underperforming','calibrated':False,'noise_kg':100.},
    {'status':'passed','calibrated':True,'noise_kg':.001}])
def test_validation_is_descriptive_and_never_an_eligibility_gate(validation):
    predicted=forecast()
    if validation is None: del predicted['validation']
    else: predicted['validation']=validation
    assert_recommended(recommend_harvest(predicted,[],'unknown'))


@pytest.mark.parametrize('event',['harvest','feeding','inspection','sensor_fault','colony_alert'])
def test_events_do_not_gate_an_available_forecast(event):
    observed=observations();observed[-5]['event']=event
    assert_recommended(recommend_harvest(forecast(),observed,'apis'))


@pytest.mark.parametrize('observed',[[],[{}],[{'weight_kg':float('nan')}],None])
def test_missing_history_does_not_invent_growth_or_block_a_forecast(observed):
    result=recommend_harvest(forecast(),observed,'unknown')
    assert_recommended(result)
    assert result['evidence']['recent_growth_kg'] is None


def test_declining_observed_history_does_not_block_forecast_peak():
    result=recommend_harvest(forecast(),observations((52.,51.,50.)),'apis')
    assert_recommended(result)
    assert result['evidence']['recent_growth_kg']==pytest.approx(-2.)


@pytest.mark.parametrize('key',['trajectory','as_of','model','forecast_id','horizon_hours'])
def test_missing_forecast_integrity_fields_are_honest_unavailable_states(key):
    predicted=forecast();del predicted[key]
    assert_unavailable(recommend_harvest(predicted,[],'unknown'))


@pytest.mark.parametrize('field,value',[
    ('weight_kg',float('nan')),('weight_kg',float('inf')),('weight_kg',10**400),
    ('weight_kg',-1),('weight_kg',301),('weight_kg',True),
    ('timestamp','2025-01-04T02:00:00+00:00'),('timestamp','2025-01-04T01:30:00+00:00'),
    ('timestamp','2025-01-04T01:00:00')])
def test_invalid_forecast_points_cannot_create_a_recommendation(field,value):
    predicted=forecast();predicted['trajectory'][0][field]=value
    assert_unavailable(recommend_harvest(predicted,[],'unknown'))


def test_optional_intervals_are_not_required_but_malformed_intervals_are_rejected():
    predicted=forecast()
    for row in predicted['trajectory']: row.pop('lower_kg');row.pop('upper_kg')
    assert_recommended(recommend_harvest(predicted,[],'unknown'))
    predicted['trajectory'][0].update(lower_kg=100.,upper_kg=101.)
    assert_unavailable(recommend_harvest(predicted,[],'unknown'))


@pytest.mark.parametrize('status',['model_unavailable','insufficient_history','unsupported_horizon','event_hold'])
def test_unavailable_provider_does_not_fabricate_a_recommendation(status):
    predicted=forecast();predicted.update(status=status,trajectory=[])
    result=recommend_harvest(predicted,[],'unknown')
    assert_unavailable(result)
    assert result['forecast_id']=='forecast-exact'


def test_incomplete_trajectory_is_not_presented_as_full_forecast():
    predicted=forecast();predicted['trajectory'].pop()
    assert_unavailable(recommend_harvest(predicted,[],'unknown'))
