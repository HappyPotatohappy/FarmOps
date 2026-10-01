from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timedelta
from pathlib import Path
from copy import deepcopy
import json
import threading
import time
import numpy as np
import pandas as pd
from .contracts import validate_series
from .store import Store,digest,utcnow
from .scenarios import generate


class BusyError(Exception): pass


class Service:
    def __init__(self,runtime,model):
        self.store=Store(runtime,prediction_namespace=getattr(model,'prediction_namespace',None)); self.model=model
        self._model_meta_prefix='shared-' if getattr(model,'model_scope',None)=='shared' else ''
        self._training_key=self._model_meta_prefix+'training'
        self.lock=threading.RLock(); self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='beeops-worker')
        self.training=self.store.get_meta(self._training_key,{'status':'idle'})
        if self.training.get('status') in ('running','queued'):
            self.training={'status':'interrupted','reason':'Server restarted; previous job not resumed'}
            self.store.set_meta(self._training_key,self.training)
        self.demo={'status':'idle'}; self.last_completed=0
        self._forecast_lock=threading.RLock(); self._forecast_cache={}
        if callable(getattr(model,'version_metadata',None)) and callable(getattr(model,'versions',None)):
            for item in model.versions(): self._cutoff_for(item['version'])

    def close(self): self.executor.shutdown(wait=True)

    def _cutoff_for(self,version):
        lookup=getattr(self.model,'version_metadata',None)
        if not callable(lookup): return False,None  # Explicit lightweight fixture/legacy adapter fallback.
        try:
            metadata=lookup(str(version)); cutoff=metadata.get('monitoring_after',metadata.get('test_end'))
        except Exception:
            cutoff=None  # Actual MLflow adapters without metadata never earn operational scores.
        self.store.set_model_cutoff(str(version),cutoff)
        return True,cutoff

    def predict(self,rows,version=None):
        rows=validate_series(rows,24,24)
        configured=self.store.data_status()['hive_id']
        if configured and configured!=rows[0]['hive_id']: raise ValueError('hive_id does not match this service')
        x=np.array([[[r['weight_kg'],r['temperature_c']] for r in rows]],dtype='float32')
        values,used=self.model.predict(x,version=version)
        if version is not None and str(used)!=str(version):
            raise RuntimeError('Model adapter returned a different requested version')
        self._cutoff_for(used)
        result=self.store.record_prediction(rows,float(values[0]),used)
        if getattr(self.model,'model_scope',None)=='shared':
            metadata=self.model.version_metadata(used)
            result.update(model_scope='shared',registry_id=metadata['registry_id'],run_id=metadata['run_id'])
        return result

    def forecast_report(self,version=None,limit=96):
        """Replay one immutable ledger snapshot with an exact model; never store scores.

        Every historical target uses only its preceding 24 observations. This is
        rolling one-step replay, not a 96-hour forecast from one starting point.
        Only clean targets after this version's final test can earn operational
        metrics. Cached inference is invalidated by any ledger/version change.
        """
        if isinstance(limit,bool) or not isinstance(limit,int) or not 1<=limit<=96:
            raise ValueError('Forecast report limit must be between 1 and 96')
        state=self.model.status()
        selected=str(version) if version is not None else (str(state['version']) if state.get('version') is not None else None)
        if selected is not None and (not selected.isdecimal() or int(selected)<1):
            raise ValueError('Model version must be a positive integer')
        with self.store.lock:
            data=self.store.data_status(); observed=self.store.observations(limit+24)
            binding=self.store.get_meta('workspace_binding')
        hive=data['hive_id']; metadata={}; metadata_error=None; cutoff=None
        lookup=getattr(self.model,'version_metadata',None)
        if selected is not None and callable(lookup):
            try: metadata=lookup(selected)
            except Exception as exc: metadata_error=str(exc)
            if not isinstance(metadata,dict): raise RuntimeError('Invalid model metadata response')
            if metadata.get('version') is not None and str(metadata['version'])!=selected:
                raise RuntimeError('Model metadata returned a different requested version')
            if metadata.get('hive_id') and hive and metadata['hive_id']!=hive:
                raise RuntimeError('Model metadata hive does not match this workspace')
            candidate=metadata.get('monitoring_after') or metadata.get('test_end')
            if candidate:
                try:
                    timestamp=pd.Timestamp(candidate)
                    if timestamp.tzinfo is None or pd.isna(timestamp): raise ValueError('Cutoff must be timezone aware')
                    cutoff=timestamp.isoformat()
                except (TypeError,ValueError): metadata_error='Invalid final-test cutoff metadata'
        if observed:
            observed=validate_series(observed,minimum=1,maximum=120)
            if observed[0]['hive_id']!=hive or (binding and binding!=hive):
                raise RuntimeError('Observation hive does not match this workspace')
        model={'version':selected,'run_id':metadata.get('run_id'),
               'is_champion':selected is not None and selected==str(state.get('version')),
               'monitoring_after':cutoff,'trained_until':metadata.get('train_end'),
               'metadata_available':bool(cutoff)}
        if metadata.get('model_scope')=='shared':
            model.update(model_scope='shared',registry_id=metadata['registry_id'])
        if metadata_error: model['metadata_error']=metadata_error
        fingerprint=digest({'snapshot_id':data['snapshot_id'],'model':model,'limit':limit,'hive_id':hive})

        def score(items,kind):
            result={'kind':kind,'count':len(items),'excluded_count':len(history)-len(items),
                    'status':'available' if items else 'insufficient_data',
                    'minimum_count':24,'sample_sufficient':len(items)>=24,
                    'model_mae_kg':None,'persistence_mae_kg':None,'seasonal24_mae_kg':None,
                    'improvement_vs_persistence_kg':None,'improvement_vs_seasonal24_kg':None}
            if items:
                actual=np.array([row['actual_weight_kg'] for row in items])
                for name,field in [('model','predicted_weight_kg'),('persistence','persistence_weight_kg'),('seasonal24','seasonal24_weight_kg')]:
                    result[f'{name}_mae_kg']=float(np.mean(np.abs(actual-np.array([row[field] for row in items]))))
                for name in ('persistence','seasonal24'):
                    result[f'improvement_vs_{name}_kg']=result[f'{name}_mae_kg']-result['model_mae_kg']
            if kind=='post_cutoff_clean' and cutoff is None: result['status']='metadata_unavailable'
            return result

        with self._forecast_lock:
            cached=self._forecast_cache.get(fingerprint)
            if cached is not None: return deepcopy(cached)
            history=[]
            report={'status':'completed','workspace_id':binding or hive or 'default','hive_id':hive,
                    'snapshot_id':data['snapshot_id'],'fingerprint':fingerprint,'generated_at':utcnow(),
                    'horizon_hours':1,'evaluation_mode':'rolling_one_step','model':model,
                    'data':{key:data[key] for key in ('snapshot_id','start','end','rows')},
                    'rows':history,'next_forecast':None}
            if len(observed)<24:
                report.update(status='insufficient_data',reason='최근 연속 관측 24개가 필요합니다.')
            elif selected is None:
                report.update(status='model_unavailable',reason='사용 가능한 모델 버전이 없습니다.')
            else:
                contexts=[observed[i-24:i] for i in range(24,len(observed))]+[observed[-24:]]
                windows=np.array([[[r['weight_kg'],r['temperature_c']] for r in window] for window in contexts],dtype='float32')
                values,used=self.model.predict(windows,version=selected)
                if str(used)!=selected: raise RuntimeError('Model adapter returned a different requested version')
                values=np.asarray(values,dtype=float)
                if values.shape!=(len(contexts),) or not np.isfinite(values).all() or (values<=0).any() or (values>300).any():
                    raise RuntimeError('Model adapter returned invalid prediction values')
                for index,(window,value) in enumerate(zip(contexts,values)):
                    last=window[-1]
                    target=(datetime.fromisoformat(last['timestamp'])+timedelta(hours=1)).isoformat()
                    clean=all(row['event']=='normal' for row in window)
                    prediction={'target_timestamp':target,'input_start':window[0]['timestamp'],
                                'input_end':last['timestamp'],'model_version':selected,'predicted_weight_kg':float(value),
                                'persistence_weight_kg':last['weight_kg'],'seasonal24_weight_kg':window[0]['weight_kg'],
                                'context_clean':clean}
                    if index==len(contexts)-1:
                        report['next_forecast']={**prediction,'actual_weight_kg':None,
                            'last_observed_weight_kg':last['weight_kg'],'predicted_delta_kg':float(value)-last['weight_kg']}
                    else:
                        actual=observed[index+24]
                        if target!=actual['timestamp']: raise RuntimeError('Prediction target does not align with the observed hour')
                        after=cutoff is not None and pd.Timestamp(target)>pd.Timestamp(cutoff)
                        history.append({**prediction,'actual_weight_kg':actual['weight_kg'],
                            'target_clean':actual['event']=='normal','historical_replay':True,
                            'period':'post_cutoff' if after else 'training_or_evaluation' if cutoff else 'unknown',
                            'monitoring_eligible':bool(after and clean and actual['event']=='normal')})
            report['metrics']={'post_cutoff_clean':score([row for row in history if row['monitoring_eligible']],'post_cutoff_clean'),
                               'all_window_descriptive':score(history,'all_window_descriptive')}
            if len(self._forecast_cache)>=8: self._forecast_cache.pop(next(iter(self._forecast_cache)))
            self._forecast_cache[fingerprint]=deepcopy(report)
            return report

    def ingest(self,rows,source='사용자 관측 업로드'):
        result=self.store.ingest(rows,source)
        try:
            self.check_quality()
        except Exception as exc:
            result.update(status='partial',warnings=[f'Quality monitoring: {exc}'])
        return result

    def batch(self,rows,source='사용자 과거 관측 재생',auto=True):
        rows=validate_series(rows,minimum=25,maximum=10000)
        self.store.ingest(rows,source)
        state=self.model.status()
        if not state.get('ready'): raise RuntimeError('Model service is not ready')
        version=state['version']
        known,cutoff=self._cutoff_for(version)
        eligible=lambda i: not known or (cutoff is not None and pd.Timestamp(rows[i]['timestamp'])>pd.Timestamp(cutoff))
        indices=[i for i in range(24,len(rows)) if eligible(i)]
        predictions=[]
        for start in range(0,len(indices),512):
            selected=indices[start:start+512]
            windows=np.array([[[r['weight_kg'],r['temperature_c']] for r in rows[i-24:i]] for i in selected],dtype='float32')
            values,used=self.model.predict(windows,version=version)
            if str(used)!=str(version): raise RuntimeError('Model adapter returned a different requested version')
            predictions.extend(self.store.record_prediction(rows[i-24:i],float(v),version) for i,v in zip(selected,values))
        quality=self.store.quality(version)
        action=self.check_quality() if auto else {'status':'not_requested'}
        return {'predictions':predictions,'quality':quality,'model_version':version,'action':action,
                'excluded_training_targets':len(rows)-24-len(predictions),'monitoring_after':cutoff,
                'snapshot_id':self.store.data_status()['snapshot_id']}

    def _analyze_ledger(self,rows,source):
        aggregate={'status':'completed','prediction_count':0,'excluded_training_targets':0,
                   'model_version':None,'monitoring_after':None,'quality':None}
        for start in range(24,len(rows),9976):
            chunk=rows[start-24:min(start+9976,len(rows))]
            analyzed=self.batch(chunk,source=source,auto=False)
            aggregate['prediction_count']+=len(analyzed['predictions'])
            aggregate['excluded_training_targets']+=analyzed['excluded_training_targets']
            for key in ('model_version','monitoring_after','quality'): aggregate[key]=analyzed[key]
        return aggregate

    def _bootstrap_frame(self,ignore_consumed=False):
        state=self.model.status(); cutoff=state.get('consumed_through')
        if ignore_consumed or state.get('can_reinitialize_for_demo'):
            cutoff=None  # A recorded policy migration creates a new trained version.
        minimum=state.get('minimum_training_rows',168); blocks=[]; clean=[]
        for row in self.store.observations():
            if cutoff and pd.Timestamp(row['timestamp'])<=pd.Timestamp(cutoff): continue
            if row['event']=='normal': clean.append(row)
            else:
                if clean: blocks.append(clean)
                clean=[]
        if clean: blocks.append(clean)
        selected=next((block[:672] for block in blocks if len(block)>=minimum),max(blocks,key=len,default=[]))
        frame=pd.DataFrame(selected)
        if selected: frame.attrs['source']=' | '.join(self.store.sources(selected[0]['timestamp'],selected[-1]['timestamp']))
        return frame

    def import_upload(self,rows,source,progress=None,*,temperature_practice=False):
        """Commit once; downstream failures remain visible without undoing data."""
        result={'status':'completed','ingest':None,'model_initialization':{},
                'analysis':{'status':'unavailable','prediction_count':0,'excluded_training_targets':0,
                            'model_version':None,'monitoring_after':None,'quality':None},
                'forecast':{'status':'unavailable','prediction':None},'warnings':[]}
        def report(stage,percent,details=None):
            if progress:
                try: progress(stage,percent,details)
                except Exception as exc: result['warnings'].append(f'Progress reporting: {exc}')
        def persist_training():
            try: self.store.set_meta(self._training_key,self.training)
            except Exception as exc: result['warnings'].append(f'Training reporting: {exc}')
        with self.lock:
            if self.training.get('status') in ('running','queued') or self.demo.get('status')=='running':
                raise BusyError('A model or demo job is active in this workspace; retry the import after it finishes')
            report('saving_data',10)
            result['ingest']=self.store.ingest(rows,source,temperature_practice=temperature_practice)
            report('data_saved',30,{'ingest':result['ingest']})
            state=self.model.status()
            initialization={'status':'already_ready','ready':bool(state.get('ready')),'version':state.get('version')}
            if not state.get('ready'):
                frame=self._bootstrap_frame(); minimum=state.get('minimum_training_rows',168)
                initialization.update(available_rows=len(frame),required_rows=minimum)
                if len(frame)<minimum:
                    initialization['status']='insufficient_data'
                    result['warnings'].append(f'모델 준비에는 새로운 연속 정상 관측 {minimum}개가 필요합니다. 현재 {len(frame)}개입니다.')
                else:
                    self.training={'status':'running','reason':'initial_upload','rows':len(frame)}
                    persist_training()
                    report('model_initialization',40,{'model_initialization':initialization})
                    try:
                        trained=self.model.bootstrap(frame)
                        state=self.model.status()
                        initialization.update(trained,candidate_version=trained.get('version'),
                                              ready=bool(state.get('ready')),version=state.get('version'))
                        initialization['status']='initialized' if state.get('ready') else 'quality_rejected'
                        if not state.get('ready'): result['status']='partial'
                        self.training={**trained,'status':'completed','reason':'initial_upload'}
                    except Exception as exc:
                        initialization.update(status='failed',ready=False,error=str(exc))
                        result['status']='partial'; result['warnings'].append(str(exc))
                        self.training={'status':'failed','reason':'initial_upload','error':str(exc)}
                    finally:
                        self.last_completed=time.monotonic()
                        persist_training()
            result['model_initialization']=initialization
            prepare_pair=getattr(self.model,'ensure_comparison_version',None)
            if initialization['ready'] and callable(prepare_pair):
                try:
                    report('model_versions',60)
                    result['model_versions']=prepare_pair(self._bootstrap_frame(ignore_consumed=True))
                except Exception as exc:
                    result['model_versions']={'status':'failed','error':str(exc)}
                    result['status']='partial'
                    result['warnings'].append(f'v1/v2 모델 준비: {exc}')
            report('model_prepared',65,{'model_initialization':initialization})
            if initialization['ready']:
                all_rows=self.store.observations()
                try:
                    report('analyzing',70)
                    if len(all_rows)>24:
                        result['analysis']=self._analyze_ledger(all_rows,source)
                except Exception as exc:
                    result['analysis']['status']='failed'; result['analysis']['error']=str(exc)
                    result['warnings'].append(str(exc)); result['status']='partial'
                report('analysis_complete',85,{'analysis':result['analysis']})
                try:
                    report('forecasting',90)
                    if len(all_rows)>=24:
                        result['forecast']={'status':'completed','prediction':self.predict(all_rows[-24:])}
                    else: result['forecast']['reason']='최근 연속 관측 24개가 필요합니다.'
                except Exception as exc:
                    result['forecast'].update(status='failed',reason=str(exc))
                    result['warnings'].append(str(exc)); result['status']='partial'
                try:
                    result['analysis']['action']=self.check_quality()
                except Exception as exc:
                    result['warnings'].append(f'Quality monitoring: {exc}'); result['status']='partial'
            else:
                result['forecast']['reason']='아직 학습이 완료된 예측 모델이 없습니다.'
            report('finished',100,{'analysis':result['analysis'],'forecast':result['forecast'],
                                   'warnings':result['warnings'],'status':result['status']})
            return result

    def clean_training_frame(self):
        status=self.model.status(); cutoff=status.get('consumed_through') or status.get('trained_through')
        clean=[]
        for r in self.store.observations():
            if cutoff and datetime.fromisoformat(r['timestamp'])<=pd.Timestamp(cutoff).to_pydatetime(): continue
            if r['event']!='normal': clean=[]
            else: clean.append(r)
        clean=clean[-672:]
        frame=pd.DataFrame(clean)
        if clean: frame.attrs['source']=' | '.join(self.store.sources(clean[0]['timestamp'],clean[-1]['timestamp']))
        return frame

    def check_quality(self):
        state=self.model.status()
        q=self.store.quality(state.get('version'))
        # Ordinary CSV/prediction workflows demonstrate inference. Error
        # measurements never block them or launch unsolicited retraining.
        # Keep the dedicated synthetic drift exercise operational.
        drift_demo=(self.store.get_meta('demo_runtime',False)
                    and self.demo.get('status')=='running'
                    and self.demo.get('scenario')=='drift')
        if not drift_demo:
            return {**q,'monitoring_mode':'record_only'}
        if q['status']!='degraded': return q
        key=f'{self._model_meta_prefix}quality-alert-{state.get("version")}'
        if not self.store.get_meta(key):
            self.store.alert('model','WARN','예측 품질 저하 — 작업·센서·벌통 상태 확인 후 정상 관측으로 재학습',q)
            self.store.log('WARN','quality drift detected',**q); self.store.set_meta(key,True)
        try: return self.request_training('quality_drift')
        except (ValueError,BusyError) as exc: return {**q,'training_deferred':str(exc)}

    def request_training(self,reason='operator_request'):
        with self.lock:
            if self.training.get('status') in ('queued','running'): raise BusyError('A training job is already active')
            if time.monotonic()-self.last_completed<60: raise BusyError('Training cooldown: 60 seconds')
            frame=self.clean_training_frame() if self.model.status().get('ready') else self._bootstrap_frame()
            minimum=self.model.status().get('minimum_training_rows',168)
            if len(frame)<minimum: raise ValueError(f'Need at least {minimum} new contiguous normal observations; found {len(frame)}')
            records=frame.to_dict('records'); snapshot_id=digest(records)
            attempt_key=self._model_meta_prefix+'last_attempt_snapshot'
            if self.store.get_meta(attempt_key)==snapshot_id: raise ValueError('This snapshot was already evaluated')
            folder=self.store.path/'snapshots'; folder.mkdir(exist_ok=True)
            (folder/f'{snapshot_id}.json').write_text(json.dumps({'source':frame.attrs.get('source'),'rows':records},ensure_ascii=False,indent=2))
            self.store.set_meta(attempt_key,snapshot_id)
            self.training={'status':'queued','reason':reason,'snapshot_id':snapshot_id,'rows':len(frame),'requested_at':utcnow()}
            self.store.set_meta(self._training_key,self.training)
            self.executor.submit(self._train,frame,reason,snapshot_id)
            return self.training.copy()

    def _train(self,frame,reason,snapshot_id):
        warnings=[]
        def report(fn,*args,**kwargs):
            try: fn(*args,**kwargs)
            except Exception as exc: warnings.append(str(exc))
        with self.lock: self.training={**self.training,'status':'running'}
        report(self.store.set_meta,self._training_key,self.training)
        report(self.store.log,'INFO','retrain triggered',snapshot_id=snapshot_id,rows=len(frame))
        try:
            status=getattr(self.model,'status',None)
            initial=callable(status) and not status().get('ready')
            result=self.model.bootstrap(frame) if initial else self.model.train_candidate(frame,reason=reason,snapshot_id=snapshot_id)
            prepare_pair=getattr(self.model,'ensure_comparison_version',None)
            pair_failed=False
            if initial and status().get('ready') and callable(prepare_pair):
                try: result['model_versions']=prepare_pair(frame)
                except Exception as exc:
                    pair_failed=True
                    result['model_versions']={'status':'failed','error':str(exc)}
                    warnings.append(f'v1/v2 모델 준비: {exc}')
            if result.get('version'): report(self._cutoff_for,result['version'])
            state={**self.training,**result,'status':'partial' if pair_failed else 'completed','completed_at':utcnow()}
            report(self.store.log,'OK' if result.get('promoted') else 'WARN',
                           'candidate promoted' if result.get('promoted') else 'gate failed; existing champion retained',**result)
            report(self.store.alert,'model','OK' if result.get('promoted') else 'WARN',
                             '새 모델 교체 완료' if result.get('promoted') else '후보 게이트 실패 — 기존 모델 유지',result)
        except Exception as exc:
            state={**self.training,'status':'failed','error':str(exc),'completed_at':utcnow()}
            report(self.store.log,'ERROR','retraining failed',error=str(exc))
        with self.lock:
            state['reporting_warnings']=warnings
            self.training=state; self.last_completed=time.monotonic()
            report(self.store.set_meta,self._training_key,state)

    def run_demo(self,scenario):
        with self.lock:
            if self.demo.get('status')=='running' or self.training.get('status') in ('running','queued'):
                raise BusyError('A demo or training job is already active')
            if not self.store.get_meta('demo_runtime',False): raise ValueError('Synthetic demos require the dedicated synthetic runtime')
            self.demo={'status':'running','scenario':scenario,'started_at':utcnow()}
            self.executor.submit(self._demo,scenario)
            return self.demo.copy()

    def _demo(self,scenario):
        try:
            if scenario=='gate_fail' and getattr(self.model,'model_scope',None)=='shared':
                candidate=next(v for v in self.model.versions() if v['version']=='3')
                if candidate['gate_passed']:
                    raise ValueError('Shared v3 is not a rejected quality demonstration')
                action={'status':'completed','version':'3','gate_passed':False,'promoted':False,
                        'reused_existing_candidate':True,'mae':candidate['metrics'].get('mae'),
                        'gate_reasons':candidate.get('gate_reasons',[])}
                self.demo={'status':'completed','scenario':scenario,'completed_at':utcnow(),
                           'quality':self.store.quality(self.model.status().get('version')),
                           'before_version':self.model.status().get('version'),'action':action,
                           'snapshot_id':self.model.version_metadata('3').get('snapshot_id')}
                self.store.set_meta('last_demo',self.demo)
                self.store.log('WARN','shared v3 quality gate demonstration',**action)
                return
            past=self.store.observations(24)
            count=336 if scenario in ('drift','gate_fail') else 48
            new=generate(count,start=(datetime.fromisoformat(past[-1]['timestamp'])+timedelta(hours=1)),
                         last_weight=past[-1]['weight_kg'],scenario=scenario,hive_id=past[-1]['hive_id'])
            result=self.batch(past+new,source=f'합성 교육 시나리오: {scenario}',auto=scenario in ('normal','drift'))
            if scenario=='gate_fail':
                # Same public training path and unchanged gate; unpredictable data should be rejected by quality.
                result['action']=self.request_training('synthetic_unpredictable_gate_test')
            self.demo={'status':'completed','scenario':scenario,'completed_at':utcnow(),'quality':result['quality'],
                       'before_version':result['model_version'],'action':result['action'],'snapshot_id':result['snapshot_id']}
            self.store.set_meta('last_demo',self.demo)
        except Exception as exc:
            self.demo={'status':'failed','scenario':scenario,'error':str(exc)}
            self.store.log('ERROR','demo failed',error=str(exc))

    def health(self):
        state=self.model.status(); version=state.get('version')
        if version is not None: self._cutoff_for(version)
        return {'status':'ok' if state.get('ready') else 'not_ready','model_loaded':state.get('ready',False),
                'model_scope':state.get('model_scope','hive'),'registry_id':state.get('registry_id'),
                'version':version,'model_source':state.get('model_source','mlflow'),'loading_mode':'eager',
                'retraining':self.training,'demo':self.demo,'quality':self.store.quality(version),
                'data':self.store.data_status(),'model':state}
