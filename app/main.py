import csv
import io
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit
from fastapi import FastAPI,File,HTTPException,UploadFile,Query,Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse,JSONResponse,PlainTextResponse,Response
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel,Field
from fastapi.staticfiles import StaticFiles
from .contracts import PredictRequest,ObservationsRequest,RollbackRequest,DemoRequest,validate_series
from .service import Service,BusyError,UnknownModelVersion
from .harvest_preferences import HarvestPreferences,read_preferences,save_preferences
from .store import utcnow


class ImportCommitRequest(BaseModel):
    preview_id: str = Field(min_length=1,max_length=128)


class TemperaturePracticePreviewRequest(BaseModel):
    filename: str = Field(min_length=1,max_length=80)


def create_app(runtime=None,model=None,model_factory=None,shared_model=None,horizon_model=None,temperature_monitor=None,dashboard_dataset=None):
    enable_temperature = model is None
    dataset=dashboard_dataset if dashboard_dataset is not None else os.environ.get('BEEOPS_DASHBOARD_DATASET','')
    runtime=Path(runtime or os.environ.get('BEEOPS_RUNTIME',Path(__file__).resolve().parents[1]/'runtime'))
    @asynccontextmanager
    async def lifespan(app):
        nonlocal model,model_factory
        if model is None:
            from .shared_runtime import SharedModelCoordinator
            coordinator=SharedModelCoordinator(runtime,model=shared_model)
            app.state.shared_models=coordinator
            model=coordinator.for_runtime(runtime)
            model_factory=coordinator.for_runtime
        from .workspaces import WorkspaceManager
        from .imports import ImportService
        app.state.service=Service(runtime,model)
        app.state.workspaces=WorkspaceManager(runtime,app.state.service,model_factory=model_factory)
        from .horizon_service import HorizonReports
        provider=horizon_model
        if provider is None:
            from .horizon_models import HorizonModelService
            artifact_root=Path(os.environ.get('BEEOPS_HORIZON_MODELS',
                Path(__file__).resolve().parents[1]/'data'/'horizon_models'))
            provider=HorizonModelService(artifact_root)
        app.state.horizon=HorizonReports(provider)
        from .horizon_history import HorizonHistoryReports
        app.state.history=HorizonHistoryReports(provider)
        from .trend_cases import TrendCaseReports
        app.state.trend_cases=TrendCaseReports(provider)
        app.state.dashboard_data=None
        if dataset=='observed':
            from .dashboard_data import DashboardData
            app.state.dashboard_data=DashboardData(app.state.workspaces,app.state.trend_cases)
            app.state.dashboard_data.seed()
        monitor=temperature_monitor
        if monitor is None and enable_temperature and hasattr(provider,'artifact_root'):
            from .temperature_drift import TemperatureDriftCoordinator
            monitor=TemperatureDriftCoordinator(runtime,provider)
        app.state.temperature=monitor
        def on_observations(selected,workspace_id,practice=None):
            if app.state.dashboard_data: app.state.dashboard_data.register(workspace_id)
            if monitor and practice and practice['phase']=='heatwave':
                return monitor.observe_practice(selected,workspace_id,request_id=practice['request_id'],
                    inserted_observations=practice['inserted_observations'])
            return monitor.observe(selected,workspace_id) if monitor else None
        app.state.imports=ImportService(app.state.workspaces,runtime,
            on_observations=on_observations if monitor or app.state.dashboard_data else None)
        if monitor and app.state.dashboard_data:
            for item in app.state.trend_cases.catalog()['cases']:
                monitor.observe(app.state.workspaces.get(item['hive_id']),item['hive_id'])
        try:
            yield
        finally:
            app.state.imports.close()
            if app.state.temperature: app.state.temperature.close()
            app.state.workspaces.close()

    app=FastAPI(title='BeeOPS · 벌통 예측 운영',version='2.4.0',lifespan=lifespan)
    metrics=[]; lock=threading.Lock()

    @app.exception_handler(RequestValidationError)
    async def validation(request,exc):
        return JSONResponse(status_code=422,content={'detail':[
            {k:e[k] for k in ('type','loc','msg')} for e in exc.errors()]})

    @app.exception_handler(ValueError)
    async def bad_value(request,exc): return JSONResponse(status_code=422,content={'detail':str(exc)})

    @app.exception_handler(UnknownModelVersion)
    async def unknown_version(request,exc): return JSONResponse(status_code=404,content={'detail':str(exc)})

    @app.exception_handler(BusyError)
    async def busy(request,exc): return JSONResponse(status_code=409,content={'detail':str(exc)})

    @app.middleware('http')
    async def measure(request,call_next):
        start=time.perf_counter(); code=500
        try:
            response=await call_next(request); code=response.status_code
            if request.url.path in ('/','/index.html','/app.js','/styles.css'):
                response.headers['Cache-Control']='no-store'
            return response
        finally:
            with lock:
                metrics.append({'path':request.url.path,'ms':(time.perf_counter()-start)*1000,'status':code})
                if len(metrics)>10000: del metrics[:1000]

    def workspace_key(workspace_id=None):
        dashboard=app.state.dashboard_data
        return workspace_id or (dashboard.default_workspace_id if dashboard else app.state.workspaces.default_workspace_id)

    def visible_workspaces():
        items=app.state.workspaces.catalog()
        return app.state.dashboard_data.catalog(items) if app.state.dashboard_data else items

    def service(workspace_id=None):
        try: return app.state.workspaces.get(workspace_key(workspace_id))
        except KeyError as exc: raise HTTPException(404,'작업공간을 찾을 수 없습니다.') from exc
        except (RuntimeError,FileNotFoundError) as exc:
            raise HTTPException(503,detail={'code':'workspace_unavailable',
                'message':'이 벌통의 모델 저장소를 불러오지 못했습니다. 모델 관리 화면의 오류를 확인해 주세요.',
                'workspace_id':workspace_id}) from exc

    @app.get('/workspaces')
    def workspaces():
        items=visible_workspaces()
        shared=getattr(app.state,'shared_models',None)
        return {'default_workspace_id':workspace_key(),
                'workspaces':items,'model_count_total':len(shared.model.versions()) if shared else sum(x.get('model_count',0) for x in items),
                'model_scope':'shared' if shared else 'hive'}

    @app.get('/models/catalog')
    def model_catalog():
        items=[]
        for workspace in visible_workspaces():
            entry={'workspace_id':workspace['workspace_id'],'name':workspace['name'],'versions':[],
                   'error':workspace.get('error')}
            if getattr(app.state,'shared_models',None):
                entry.update(model_scope='shared',registry_id='BeeOPS_Common_Weight')
            try: entry['versions']=service(workspace['workspace_id']).model.versions()
            except Exception as exc: entry['error']=str(exc)
            items.append(entry)
        return items

    @app.post('/imports/preview')
    async def preview_import(file:UploadFile=File(...)):
        content=await file.read(5*1024*1024+1)
        if len(content)>5*1024*1024: raise HTTPException(413,'CSV 파일은 5 MiB 이하여야 합니다.')
        return await run_in_threadpool(app.state.imports.preview,content,file.filename or 'data.csv')

    @app.post('/imports/commit',status_code=202)
    def commit_import(payload:ImportCommitRequest):
        try: return app.state.imports.commit(payload.preview_id)
        except KeyError as exc: raise HTTPException(404,'미리보기를 찾을 수 없습니다. 파일을 다시 선택해 주세요.') from exc

    @app.get('/imports/jobs')
    def import_jobs(workspace_id:str|None=None): return app.state.imports.list_jobs(workspace_id)

    @app.get('/imports/jobs/{job_id}')
    def import_job(job_id:str):
        try: return app.state.imports.get_job(job_id)
        except KeyError as exc: raise HTTPException(404,'업로드 작업을 찾을 수 없습니다.') from exc

    @app.get('/data/sample.csv')
    def sample_csv():
        if app.state.dashboard_data:
            catalog=app.state.trend_cases.catalog()
            path=app.state.trend_cases.file(catalog['default_case_id'])
            return FileResponse(path,media_type='text/csv',filename=path.name)
        return FileResponse(Path(__file__).resolve().parents[1]/'data/real_hive.csv',
                            media_type='text/csv',filename='BeeOPS_public_ufc_apis_2.csv')

    @app.get('/data/samples')
    def public_sample_catalog():
        if app.state.dashboard_data:
            catalog=app.state.trend_cases.catalog()
            return {'default_sample_id':catalog['default_case_id'],'samples':[
                {**case,'description':case['label'],'usage_note':'관측 기록을 CSV로 내려받거나 추가할 수 있습니다.'}
                for case in catalog['cases']]}
        from .public_data import sample_catalog
        return sample_catalog()

    @app.get('/data/samples/{sample_id}.csv')
    def public_sample_csv(sample_id:str):
        from .public_data import sample_file
        try: path=sample_file(sample_id)
        except KeyError as exc: raise HTTPException(404,'등록된 공개 CSV가 아닙니다.') from exc
        return FileResponse(path,media_type='text/csv',filename=f'BeeOPS_{sample_id}.csv')

    @app.get('/data/rows')
    def data_rows(workspace_id:str|None=None,page:int=Query(default=1,ge=1),
                  page_size:int=Query(default=25,ge=1,le=100)):
        store=service(workspace_id).store
        with store.db() as conn:
            total=conn.execute('SELECT COUNT(*) FROM observations').fetchone()[0]
            rows=conn.execute('SELECT timestamp,hive_id,weight_kg,temperature_c,event FROM observations ORDER BY timestamp DESC LIMIT ? OFFSET ?',
                              (page_size,(page-1)*page_size)).fetchall()
        return {'items':[dict(row) for row in rows],'total':total,'page':page,'page_size':page_size}

    @app.get('/data/trend-cases')
    def trend_case_catalog():
        try: return app.state.trend_cases.catalog()
        except (OSError,ValueError) as exc: raise HTTPException(503,'관측 사례를 불러오지 못했습니다.') from exc

    @app.get('/data/trend-cases.zip')
    def trend_case_zip():
        path=app.state.trend_cases.root.with_suffix('.zip')
        if not path.is_file(): raise HTTPException(404,'관측 사례 파일이 없습니다.')
        return FileResponse(path,media_type='application/zip',filename='BeeOPS_observation_cases.zip')

    @app.get('/data/trend-cases/{case_id}.csv')
    def trend_case_csv(case_id:str):
        try: path=app.state.trend_cases.file(case_id)
        except KeyError as exc: raise HTTPException(404,'관측 사례를 찾을 수 없습니다.') from exc
        except ValueError as exc: raise HTTPException(503,'관측 사례 파일이 손상되었습니다.') from exc
        return FileResponse(path,media_type='text/csv',filename=path.name)

    @app.get('/analysis/trend-cases/{case_id}')
    def trend_case_report(case_id:str,response:Response):
        response.headers['Cache-Control']='no-store'
        try: return app.state.trend_cases.report(case_id)
        except KeyError as exc: raise HTTPException(404,'관측 사례를 찾을 수 없습니다.') from exc
        except (OSError,RuntimeError,ValueError) as exc: raise HTTPException(503,'관측 사례 예측을 계산하지 못했습니다.') from exc

    @app.get('/analysis/trend-cases/{case_id}/forecast.csv')
    def trend_case_forecast_csv(case_id:str):
        try: content=app.state.trend_cases.forecast_csv(case_id)
        except KeyError as exc: raise HTTPException(404,'관측 사례를 찾을 수 없습니다.') from exc
        except (OSError,RuntimeError,ValueError) as exc: raise HTTPException(503,'관측 사례 예측을 계산하지 못했습니다.') from exc
        return Response(content,media_type='text/csv; charset=utf-8',headers={
            'Cache-Control':'no-store','Content-Disposition':f'attachment; filename="{case_id}_forecast.csv"'})

    @app.get('/data/export')
    def export_data(workspace_id:str|None=None):
        store=service(workspace_id).store
        output=io.StringIO(); writer=csv.DictWriter(output,fieldnames=['timestamp','hive_id','weight_kg','temperature_c','event'])
        writer.writeheader(); writer.writerows(store.observations())
        hive=store.data_status()['hive_id'] or 'observations'
        return Response(output.getvalue(),media_type='text/csv; charset=utf-8',
                        headers={'Content-Disposition':f'attachment; filename="BeeOPS_{hive}.csv"'})

    def public_data_dashboard_url():
        value=os.environ.get('BEEOPS_PUBLIC_DATA_URL','').strip()
        parsed=urlsplit(value)
        return value if parsed.scheme in ('http','https') and parsed.hostname else None

    @app.get('/health')
    def health(workspace_id:str|None=None): return service(workspace_id).health()

    @app.get('/config')
    def config(workspace_id:str|None=None):
        s=service(workspace_id); data=s.store.data_status()
        observed_hives={item['hive_id'] for item in app.state.trend_cases.catalog()['cases']} if app.state.dashboard_data else set()
        kind='synthetic' if s.store.get_meta('demo_runtime',False) else 'public_real' if data['hive_id']=='ufc_apis_2' or data['hive_id'] in observed_hives else 'single_hive'
        return {'seq_len':24,'window':24,'threshold_kg':.15,'jump_threshold_kg':1,'horizon_hours':1,
                'features':['weight_kg','temperature_c'],'data_source':data['source'],
                'runtime_type':kind,'hive_id':data['hive_id'],
                'public_data_dashboard_url':public_data_dashboard_url()}

    @app.get('/data/status')
    def data_status(workspace_id:str|None=None): return service(workspace_id).store.data_status()

    @app.post('/data/upload')
    async def upload(file:UploadFile=File(...),workspace_id:str|None=None):
        content=await file.read(5*1024*1024+1)
        if len(content)>5*1024*1024: raise ValueError('CSV maximum size is 5 MiB')
        # Validation, SQLite writes and temperature monitoring block; keep them off the event loop.
        return await run_in_threadpool(ingest_upload,content,file.filename,workspace_id)

    def ingest_upload(content,filename,workspace_id):
        try:
            reader=csv.DictReader(io.StringIO(content.decode('utf-8-sig')))
            expected={'timestamp','hive_id','weight_kg','temperature_c','event'}
            if set(reader.fieldnames or [])!=expected: raise ValueError('CSV columns must be '+','.join(sorted(expected)))
            rows=validate_series(list(reader),minimum=24)
        except UnicodeError as exc: raise ValueError('CSV must be UTF-8') from exc
        current=service(workspace_id).store.data_status()['hive_id']; incoming=rows[0]['hive_id']
        if current and current!=incoming:
            target=public_data_dashboard_url() if incoming=='ufc_apis_2' else None
            message=f'현재 화면은 {current} 벌통용이며, CSV에는 {incoming} 벌통이 들어 있습니다. '
            message+=('공개 실측 환경을 열어 같은 CSV를 다시 선택해 주세요.' if target else '이 CSV의 벌통 전용 환경에서 업로드해 주세요.')
            raise HTTPException(422,detail={'code':'hive_mismatch','message':message,
                                'expected_hive_id':current,'received_hive_id':incoming,
                                'upload_dashboard_url':target})
        selected=service(workspace_id)
        result=selected.ingest(rows,source=f'사용자 CSV: {Path(filename or "data.csv").name}')
        if app.state.imports.on_observations:
            try:
                result['temperature_monitoring']=app.state.imports.on_observations(selected,result['hive_id'])
            except Exception as exc:
                result.setdefault('warnings',[]).append(f'온도 모니터링 실패: {exc}')
                result['status']='partial'
        return result

    @app.post('/predict')
    def predict(payload:PredictRequest,model_version:str|None=Query(default=None,pattern=r'^[1-9][0-9]*$'),workspace_id:str|None=None):
        try: return service(workspace_id).predict(payload.sequence,model_version)
        except (FileNotFoundError,RuntimeError) as exc: raise HTTPException(503,str(exc)) from exc

    @app.post('/observations')
    def observations(payload:ObservationsRequest,workspace_id:str|None=None):
        selected=service(workspace_id)
        result=selected.ingest(payload.observations)
        if app.state.imports.on_observations:
            try:
                result['temperature_monitoring']=app.state.imports.on_observations(selected,result['hive_id'])
            except Exception as exc:
                result.setdefault('warnings',[]).append(f'온도 모니터링 실패: {exc}')
                result['status']='partial'
        return result

    @app.post('/predict/batch-test')
    def batch(payload:ObservationsRequest,workspace_id:str|None=None):
        try: return service(workspace_id).batch(payload.observations)
        except (FileNotFoundError,RuntimeError) as exc: raise HTTPException(503,str(exc)) from exc

    @app.get('/predictions')
    def predictions(limit:int=Query(default=100,ge=1,le=1000),workspace_id:str|None=None): return service(workspace_id).store.predictions(limit)

    @app.get('/forecast/report')
    def forecast_report(response:Response,model_version:str|None=Query(default=None,pattern=r'^[1-9][0-9]*$'),
                        limit:int=Query(default=96,ge=1,le=96),workspace_id:str|None=None):
        response.headers['Cache-Control']='no-store'
        try: result=service(workspace_id).forecast_report(model_version,limit)
        except (FileNotFoundError,RuntimeError) as exc: raise HTTPException(503,str(exc)) from exc
        result['workspace_id']=workspace_key(workspace_id)
        return result

    @app.get('/forecast/history')
    def forecast_history(response:Response,workspace_id:str|None=None,limit:int=Query(default=168,ge=24,le=720)):
        response.headers['Cache-Control']='no-store'
        selected=service(workspace_id)
        try:
            return app.state.history.report(selected,workspace_key(workspace_id),limit=limit)
        except (FileNotFoundError,RuntimeError) as exc: raise HTTPException(503,str(exc)) from exc

    @app.get('/monitoring/retraining')
    def retraining_status(response:Response):
        response.headers['Cache-Control']='no-store'
        lookup=getattr(app.state.temperature,'retraining_status',None)
        if not callable(lookup):
            result={'enabled':False,'active_model':None,'running_job':None,'queued_jobs':[],
                    'latest_job':None,'jobs':[],'server_time':utcnow()}
        else:
            result=dict(lookup())
        result['legacy_jobs']=app.state.workspaces.training_jobs()
        return result

    @app.get('/monitoring/retraining/jobs/{job_id}')
    def retraining_job(job_id:str,response:Response):
        response.headers['Cache-Control']='no-store'
        lookup=getattr(app.state.temperature,'retraining_job',None)
        if not callable(lookup): raise HTTPException(404,'재학습 작업을 찾을 수 없습니다.')
        try: return lookup(job_id)
        except KeyError as exc: raise HTTPException(404,'재학습 작업을 찾을 수 없습니다.') from exc

    @app.get('/monitoring/temperature')
    def temperature_status(response:Response,workspace_id:str|None=None):
        response.headers['Cache-Control']='no-store'
        service(workspace_id)
        key=workspace_key(workspace_id)
        if app.state.temperature is None:
            return {'enabled':False,'workspace_id':key,'check':{'status':'unavailable'},'job':None}
        return app.state.temperature.status(key)

    @app.post('/monitoring/temperature/check')
    def check_temperature(workspace_id:str|None=None):
        selected=service(workspace_id)
        if app.state.temperature is None: raise HTTPException(503,'온도 모니터링이 준비되지 않았습니다.')
        return app.state.temperature.observe(selected,workspace_key(workspace_id))

    @app.get('/data/simulations/temperature-drift.zip')
    def temperature_zip():
        path=Path(__file__).resolve().parents[1]/'data/simulations/temperature_practice.zip'
        if not path.is_file(): raise HTTPException(404,'시뮬레이션 파일이 없습니다.')
        return FileResponse(path,media_type='application/zip',filename='BeeOPS_temperature_drift.zip')

    @app.post('/data/simulations/temperature/preview')
    def preview_temperature_practice(payload:TemperaturePracticePreviewRequest):
        return app.state.imports.preview_temperature_practice(payload.filename)

    @app.get('/data/simulations/temperature/{filename:path}')
    def temperature_csv(filename:str):
        allowed={'01_baseline.csv','02_heatwave.csv','03_followup.csv','control/01_baseline.csv','control/02_normal.csv'}
        if filename not in allowed: raise HTTPException(404,'알 수 없는 시뮬레이션 파일입니다.')
        path=Path(__file__).resolve().parents[1]/'data/simulations/temperature_practice'/filename
        if not path.is_file(): raise HTTPException(404,'시뮬레이션 파일이 없습니다.')
        return FileResponse(path,media_type='text/csv',filename=path.name)

    @app.get('/observations')
    def observation_list(limit:int=Query(default=100,ge=1,le=3000),workspace_id:str|None=None): return service(workspace_id).store.observations(limit)

    @app.get('/models/horizon')
    def horizon_catalog(response:Response):
        response.headers['Cache-Control']='no-store'
        try: return app.state.horizon.catalog()
        except (FileNotFoundError,RuntimeError) as exc: raise HTTPException(503,str(exc)) from exc

    @app.get('/forecast/horizon')
    def horizon_forecast(response:Response,horizon_hours:int=Query(default=168,ge=1,le=168),
                         model_version:str|None=Query(default=None,pattern=r'^[1-9][0-9]*$'),workspace_id:str|None=None):
        response.headers['Cache-Control']='no-store'
        selected=service(workspace_id)
        try: return app.state.horizon.forecast(selected,workspace_key(workspace_id),
                                               horizon_hours,model_version)
        except (FileNotFoundError,RuntimeError) as exc: raise HTTPException(503,str(exc)) from exc

    @app.get('/harvest/preferences')
    def harvest_preferences(response:Response,workspace_id:str|None=None):
        response.headers['Cache-Control']='no-store'
        selected=service(workspace_id)
        return read_preferences(selected.store,workspace_key(workspace_id))

    @app.put('/harvest/preferences')
    def update_harvest_preferences(payload:HarvestPreferences,response:Response,workspace_id:str|None=None):
        response.headers['Cache-Control']='no-store'
        selected=service(workspace_id)
        return save_preferences(selected.store,workspace_key(workspace_id),payload)

    @app.get('/harvest/report')
    def harvest_report(response:Response,horizon_hours:int=Query(default=168,ge=1,le=168),
                       model_version:str|None=Query(default=None,pattern=r'^[1-9][0-9]*$'),workspace_id:str|None=None):
        response.headers['Cache-Control']='no-store'
        selected=service(workspace_id)
        try:
            result=app.state.horizon.harvest(selected,workspace_key(workspace_id),
                                            horizon_hours,model_version)
            from .harvest_context import build_harvest_context
            settings=read_preferences(selected.store,result['workspace_id'])
            result['preferences']=settings['preferences']
            result['preferences_updated_at']=settings['updated_at']
            result['planning']=build_harvest_context(result,settings['preferences'])
            return result
        except (FileNotFoundError,RuntimeError) as exc: raise HTTPException(503,str(exc)) from exc

    @app.get('/harvest/recommendation')
    def harvest_recommendation(response:Response,workspace_id:str|None=None):
        response.headers['Cache-Control']='no-store'
        selected=service(workspace_id)
        try:
            result=app.state.horizon.recommendation(
                selected,workspace_key(workspace_id))
            from .harvest_context import build_harvest_context
            settings=read_preferences(selected.store,result['workspace_id'])
            result['preferences']=settings['preferences']
            result['preferences_updated_at']=settings['updated_at']
            result['planning']=build_harvest_context(result,settings['preferences'])
            return result
        except (FileNotFoundError,RuntimeError) as exc: raise HTTPException(503,str(exc)) from exc

    @app.get('/alerts')
    def alerts(workspace_id:str|None=None): return service(workspace_id).store.alerts()

    @app.get('/models')
    def models(workspace_id:str|None=None): return service(workspace_id).model.versions()

    @app.post('/models/retrain',status_code=202)
    def retrain(workspace_id:str|None=None):
        selected=service(workspace_id)
        trainer=getattr(app.state.temperature,'request_training',None)
        if not callable(trainer):
            raise HTTPException(503,'앙상블 재학습이 준비되지 않았습니다.')
        try:
            return trainer(selected,workspace_key(workspace_id))
        except (FileNotFoundError,RuntimeError) as exc:
            raise HTTPException(503,str(exc)) from exc

    @app.post('/models/rollback')
    def rollback(payload:RollbackRequest,workspace_id:str|None=None):
        s=service(workspace_id)
        with s.lock:
            if s.training.get('status') in ('queued','running'): raise BusyError('Cannot roll back during training')
            try: result=s.model.rollback(payload.version)
            except (FileNotFoundError,RuntimeError) as exc: raise HTTPException(404,str(exc)) from exc
            response={**result,'activation':result,'status':'completed','forecast_report':None,'warnings':[]}
            # Activation is already committed. Reporting errors must not imply
            # the alias swap failed, or encourage an accidental second swap.
            try: s.store.log('INFO','operator rollback',**result)
            except Exception as exc:
                response['warnings'].append(f'Activation logging: {exc}'); response['status']='partial'
            try:
                response['forecast_report']=s.forecast_report(version=result['version'])
                response['forecast_report']['workspace_id']=workspace_key(workspace_id)
            except Exception as exc:
                response.update(status='partial',report_error=str(exc))
            return response

    @app.get('/demo/payload')
    def demo_payload(workspace_id:str|None=None): return {'sequence':service(workspace_id).store.observations(24)}

    @app.post('/demo/run',status_code=202)
    def demo(payload:DemoRequest,workspace_id:str|None=None): return service(workspace_id).run_demo(payload.scenario)

    @app.get('/metrics/summary')
    def metric_summary():
        with lock: data=list(metrics)
        values=sorted(r['ms'] for r in data); predictions=sorted(r['ms'] for r in data if r['path']=='/predict')
        return {'requests':len(data),'average_latency_ms':sum(values)/len(values) if values else 0,
                'p95_latency_ms':values[min(len(values)-1,int(.95*len(values)))] if values else 0,
                'predict_p95_latency_ms':predictions[min(len(predictions)-1,int(.95*len(predictions)))] if predictions else 0,
                'error_rate':sum(r['status']>=500 for r in data)/len(data) if data else 0,
                'validation_errors':sum(r['status']==422 for r in data)}

    @app.get('/logs')
    def logs(): return ['aiops.log']

    @app.get('/logs/aiops.log',response_class=PlainTextResponse)
    def log_text(workspace_id:str|None=None):
        path=service(workspace_id).store.path/'aiops.log'
        return '\n'.join(path.read_text().splitlines()[-150:]) if path.exists() else '아직 운영 로그가 없습니다.'

    @app.get('/proposal')
    def proposal():
        path=Path(__file__).resolve().parents[1]/'docs/BeeOPS_조별기획서.html'
        if not path.exists(): raise HTTPException(404,'Proposal document not generated yet')
        return FileResponse(path)

    @app.get('/model-guide')
    def model_guide():
        path=Path(__file__).resolve().parents[1]/'docs/BeeOPS_모델_상세설명.html'
        if not path.exists(): raise HTTPException(404,'Model guide not generated yet')
        return FileResponse(path)

    @app.get('/proposal.pdf')
    def proposal_pdf():
        path=Path(__file__).resolve().parents[1]/'output/pdf/BeeOPS_조별기획서.pdf'
        if not path.exists(): raise HTTPException(404,'Proposal PDF not generated yet')
        return FileResponse(path,media_type='application/pdf',filename=path.name)

    @app.get('/model-guide.pdf')
    def model_guide_pdf():
        path=Path(__file__).resolve().parents[1]/'output/pdf/BeeOPS_모델_상세설명.pdf'
        if not path.exists(): raise HTTPException(404,'Model guide PDF not generated yet')
        return FileResponse(path,media_type='application/pdf',filename=path.name)

    static=Path(__file__).parent/'static'
    if static.is_dir(): app.mount('/',StaticFiles(directory=static,html=True),name='dashboard')
    return app


app=create_app()
