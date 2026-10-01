/* Global retraining activity at the real frontend render / HTTP boundaries. */
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
const source=fs.readFileSync(path.join(__dirname,'../app/static/app.js'),'utf8').split('init();setInterval')[0];
const html=fs.readFileSync(path.join(__dirname,'../app/static/index.html'),'utf8');
const clone=value=>JSON.parse(JSON.stringify(value));
const clock='2026-10-01T08:30:00+00:00';
const model={registry_id:'BeeOPS_Horizon_Weight',version:'3',run_id:'trained-3',context_hours:168,max_horizon_hours:168};
const panelIds=['retraining-panel','retraining-title','retraining-badge','retraining-context','retraining-progress','retraining-progress-label','retraining-steps','retraining-error','retraining-details','retraining-toggle','retraining-result','retraining-updated'];
function job(status='running',patch={}){
 const finished=['completed','failed','skipped','interrupted'].includes(status);
 return {job_id:'job-running',workspace_id:'OTHER-HIVE',status,stage:status==='queued'?'trigger':status==='completed'?'deploy':'train',
  requested_at:clock,started_at:status==='queued'?null:clock,completed_at:finished?clock:null,
  parent_version:'3',registry_id:'BeeOPS_Horizon_Weight',model_scope:'shared',reason:'temperature_distribution_drift',
  check:{status:'drift',reference_mean_c:24,current_mean_c:32,mean_shift_c:8},
  training_progress:{completed_epochs:status==='completed'?8:status==='queued'?0:3,total_epochs:8,training_windows:120,last_loss:status==='queued'?null:.02,updated_at:clock},
  steps:['collect','monitor','detect','trigger','train','register','deploy'].map((key,index)=>({key,status:status==='completed'?'completed':index<4?'completed':index===4&&status!=='queued'?status==='running'?'running':status:'pending',...(index<4?{completed_at:clock}:{})})),
  ...(status==='completed'?{new_version:'4',run_id:'trained-4',mlflow_model_version:'4',evaluation:{validation_samples:24,previous_ensemble_mae_kg:.4,new_ensemble_mae_kg:.2,validation_start:clock,validation_end:clock}}:{}),...patch};
}
function activity(active=job(),patch={}){
 return {enabled:true,active_model:clone(model),running_job:active?.status==='running'?active:null,
  queued_jobs:active?.status==='queued'?[active]:[],latest_job:active,jobs:active?[active]:[],server_time:clock,...patch};
}
function fixture(handler=()=>activity(),storage=new Map()){
 const nodes=new Map(),calls=[],events=new Map(),timers=new Map();let now=Date.parse(clock),timerId=0;
 class FixtureDate extends Date{constructor(...args){super(...(args.length?args:[now]));}static now(){return now;}}
 const later=(fn,delay)=>{const id=++timerId;timers.set(id,{fn,at:now+delay});return id;};
 function advance(ms){now+=ms;for(const [id,timer] of [...timers])if(timer.at<=now){timers.delete(id);timer.fn();}}
 function node(id){
  if(!nodes.has(id)){
   const attrs=new Map(),classes=new Set();let content='',text='';
   const element={id,hidden:false,disabled:false,open:false,value:'',dataset:{},style:{},clientWidth:850,clientHeight:290,className:'',
    classList:{add(...values){values.forEach(value=>classes.add(value));},remove(...values){values.forEach(value=>classes.delete(value));},toggle(value,force){if(force===false||force===undefined&&classes.has(value)){classes.delete(value);return false;}classes.add(value);return true;},contains(value){return classes.has(value);}},
    lastElementChild:{textContent:''},addEventListener(name,callback){events.set(id+':'+name,callback);},
    setAttribute(name,value){attrs.set(name,String(value));if(name.startsWith('data-'))this.dataset[name.slice(5).replace(/-([a-z])/g,(_,letter)=>letter.toUpperCase())]=String(value);},
    getAttribute(name){return attrs.get(name)??null;},removeAttribute(name){attrs.delete(name);},append(){},scrollIntoView(){},focus(){}};
   Object.defineProperties(element,{innerHTML:{get(){return content;},set(value){content=String(value);text=content.replace(/<[^>]*>/g,'');for(const match of content.matchAll(/\bid="([^"]+)"/g))node(match[1]);}},textContent:{get(){return text;},set(value){text=String(value);content='';}}});nodes.set(id,element);
  }return nodes.get(id);
 }
 for(const match of html.matchAll(/\bid="([^"]+)"/g))node(match[1]);panelIds.forEach(node);
 const context={console,URL,AbortController,FormData,File,Blob,Map,Set,Date:FixtureDate,Number,Math,JSON,Promise,setTimeout:later,clearTimeout:id=>timers.delete(id),
  location:{origin:'http://beeops.test',hash:'#overview'},history:{replaceState(){}},localStorage:{getItem(key){return storage.get(key)||null;},setItem(key,value){storage.set(key,value);},removeItem(key){storage.delete(key);}},
  window:{addEventListener(name,callback){events.set('window:'+name,callback);},scrollTo(){}},
  document:{hidden:false,getElementById:id=>nodes.get(id)||null,querySelectorAll(){return [];},createElement:type=>node('created-'+type),addEventListener(){}},
  fetch:async(url,options)=>{calls.push({url,options});const payload=await handler(url,options);return {ok:true,text:async()=>JSON.stringify(payload)};}};
 vm.createContext(context);vm.runInContext(source,context);const state=vm.runInContext('state',context);
 Object.assign(state,{workspaceId:'SELECTED-HIVE',workspaces:[{workspace_id:'SELECTED-HIVE',name:'선택한 벌통'},{workspace_id:'OTHER-HIVE',name:'다른 벌통'}],horizonCatalog:{versions:[clone(model)]},horizonCatalogLoaded:true,horizonVersion:'3'});
 return {node,calls,context,state,events,advance,run:code=>vm.runInContext(code,context),render:data=>{context.payload=clone(data);return vm.runInContext('renderRetrainingActivity(payload)',context);},text:()=>panelIds.map(id=>node(id).textContent).join('\n')};
}
const flush=()=>new Promise(resolve=>setTimeout(resolve,0));

test('running training shows actual completed epochs and stays visible for another selected hive',async()=>{
 const f=fixture();await f.run('loadRetrainingActivity()');
 assert.equal(f.node('retraining-panel').hidden,false);assert.equal(f.node('retraining-panel').dataset.status,'running');assert.match(f.node('retraining-title').textContent,/재학습 중/);
 assert.equal(f.node('retraining-progress').value,3);assert.equal(f.node('retraining-progress').max,8);assert.match(f.node('retraining-progress-label').textContent,/3\s*\/\s*8/);assert.doesNotMatch(f.node('retraining-progress-label').textContent,/%/);
 assert.match(f.node('retraining-steps').textContent,/학습/);assert.match(f.node('retraining-context').textContent,/다른 벌통|OTHER-HIVE/);
 assert.equal(f.state.workspaceId,'SELECTED-HIVE');assert.equal(f.state.retrainingOpen,true);assert.equal(f.node('retraining-details').hidden,false);
});

test('global polling is an unscoped GET and a hive switch does not discard its pending activity',async()=>{
 let finish;const f=fixture(()=>new Promise(resolve=>{finish=resolve;}));const pending=f.run('loadRetrainingActivity()');
 f.run("refreshWorkspace=async()=>{};selectWorkspace('THIRD-HIVE')");
 assert.equal(f.calls[0].options.signal.aborted,false);
 finish(activity());await pending;
 assert.equal(f.calls.length,1);assert.equal(f.calls[0].url,'/monitoring/retraining');assert.equal(f.calls[0].options.method,'GET');
 assert.equal(f.state.retrainingActivity.running_job.job_id,'job-running');assert.equal(f.node('retraining-panel').dataset.status,'running');
});

test('a running job remains primary even when a newer queued job is latest',()=>{
 const running=job(),queued=job('queued',{job_id:'job-queued',workspace_id:'WAITING-HIVE'});
 const f=fixture();f.render(activity(running,{queued_jobs:[queued],latest_job:queued,jobs:[queued,running]}));
 assert.equal(f.node('retraining-panel').dataset.status,'running');assert.match(f.node('retraining-title').textContent,/재학습 중/);
 assert.match(f.node('retraining-progress-label').textContent,/3\s*\/\s*8/);assert.match(f.text(),/대기/);
});

test('a queued-only activity is waiting and does not claim training or deployment completion',()=>{
 const f=fixture();f.render(activity(job('queued')));assert.equal(f.node('retraining-panel').dataset.status,'queued');assert.match(f.node('retraining-title').textContent,/대기/);assert.doesNotMatch(f.node('retraining-title').textContent,/재학습 중|완료/);assert.doesNotMatch(f.node('retraining-result').textContent,/적용 완료|배포 완료/);
});

test('a live completed activity shows its new shared model result briefly',()=>{
 const f=fixture();f.render(activity(job()));f.render(activity(job('completed'),{active_model:{...model,version:'4',run_id:'trained-4'}}));
 assert.equal(f.node('retraining-panel').hidden,false);assert.equal(f.node('retraining-panel').dataset.status,'completed');assert.match(f.node('retraining-title').textContent,/완료/);assert.match(f.node('retraining-result').textContent,/4/);assert.match(f.node('retraining-result').textContent,/적용|공통/);
});

for(const [status,title] of [['failed','실패'],['interrupted','중단'],['skipped','건너뜀']])test(`${status} live activity shows the real outcome and never claims a model was deployed`,()=>{
 const f=fixture();f.render(activity(job()));f.render(activity(job(status,{error:status==='failed'?'training failed':null,reason:status==='skipped'?'no_temperature_drift_after_previous_deployment':'temperature_distribution_drift'})));
 assert.equal(f.node('retraining-panel').hidden,false);assert.equal(f.node('retraining-panel').dataset.status,status);assert.match(f.node('retraining-title').textContent,new RegExp(title));assert.doesNotMatch(f.node('retraining-result').textContent,/적용 완료|배포 완료|새 모델.*적용/);
 assert.match(f.node('retraining-result').textContent,/유지|기존/);if(status==='failed')assert.match(f.text(),/training failed/);
});

test('no-job or disabled monitoring hides the inactive global panel',()=>{
 const f=fixture();f.render(activity(null));assert.equal(f.node('retraining-panel').hidden,true);
 f.render(activity(job()));assert.equal(f.node('retraining-panel').hidden,false);
 f.render(activity(null,{enabled:false}));assert.equal(f.node('retraining-panel').hidden,true);
});

test('manual collapse survives polling of the same job and a new job opens details again',()=>{
 const f=fixture();f.render(activity(job()));assert.equal(f.state.retrainingOpen,true);
 f.run('toggleRetrainingDetails()');assert.equal(f.state.retrainingOpen,false);assert.equal(f.node('retraining-details').hidden,true);
 f.render(activity(job('running',{training_progress:{completed_epochs:4,total_epochs:8,training_windows:120,last_loss:.01,updated_at:clock}})));
 assert.equal(f.state.retrainingOpen,false);assert.equal(f.node('retraining-details').hidden,true);
 f.render(activity(job('running',{job_id:'next-job'})));assert.equal(f.state.retrainingOpen,true);assert.equal(f.node('retraining-details').hidden,false);
});

test('a network failure preserves the last known job, marks it stale, and recovers on the next poll',async()=>{
 let fail=false;const f=fixture(()=>{if(fail)throw Error('Status connection unavailable');return activity();});
 await f.run('loadRetrainingActivity()');fail=true;await f.run('loadRetrainingActivity()');
 assert.equal(f.state.retrainingActivity.running_job.job_id,'job-running');assert.equal(f.node('retraining-panel').hidden,false);
 assert.equal(f.node('retraining-panel').dataset.status,'stale');assert.match(f.node('retraining-badge').textContent,/확인 필요/);assert.match(f.node('retraining-error').textContent,/Status connection unavailable/);
 assert.match(f.node('retraining-progress-label').textContent,/3\s*\/\s*8/);
 fail=false;await f.run('loadRetrainingActivity()');assert.equal(f.node('retraining-panel').dataset.status,'running');assert.equal(f.node('retraining-error').hidden,true);
});

test('overlapping polls share one request and start a new request only after it settles',async()=>{
 const resolvers=[];const f=fixture(()=>new Promise(resolve=>resolvers.push(resolve)));
 const first=f.run('loadRetrainingActivity()'),second=f.run('loadRetrainingActivity()');await flush();assert.equal(f.calls.length,1);
 resolvers[0](activity());await Promise.all([first,second]);
 const third=f.run('loadRetrainingActivity()');await flush();assert.equal(f.calls.length,2);resolvers[1](activity(job('completed')));await third;
 assert.equal(f.node('retraining-panel').dataset.status,'completed');
});

test('a superseded poll cannot overwrite newer global activity',async()=>{
 let finish;const f=fixture(()=>new Promise(resolve=>{finish=resolve;}));const old=f.run('loadRetrainingActivity()');
 f.state.retrainingSequence++;f.render(activity(job('running',{job_id:'newer-job',training_progress:{completed_epochs:6,total_epochs:8,training_windows:120,last_loss:.01,updated_at:clock}})));
 finish(activity());await old;assert.equal(f.state.retrainingActivity.running_job.job_id,'newer-job');assert.match(f.node('retraining-progress-label').textContent,/6\s*\/\s*8/);
});

test('backend error text is displayed as text without creating injected HTML',()=>{
 const f=fixture(),message='<img src=x onerror=alert(1)>';f.render(activity(job()));f.render(activity(job('failed',{error:message})));
 assert(f.text().includes(message)||f.text().includes('&lt;img'));
 for(const id of panelIds)assert.doesNotMatch(f.node(id).innerHTML,/<img\b|<script\b/);
});

test('a changed active shared version refreshes the model catalog once across repeated polls',async()=>{
 let version='3';const f=fixture(()=>activity(job(version==='3'?'running':'completed'),{active_model:{...model,version,run_id:'trained-'+version}}));
 f.context.refreshes=0;f.run('loadHorizonCatalog=async()=>{refreshes++;};');
 await f.run('loadRetrainingActivity()');assert.equal(f.context.refreshes,0);
 version='4';await f.run('loadRetrainingActivity()');assert.equal(f.context.refreshes,1);
 await f.run('loadRetrainingActivity()');assert.equal(f.context.refreshes,1);assert.equal(f.state.retrainingActivity.active_model.version,'4');
});

test('a later overlapping catalog failure retries the remembered active model on the next global poll',async()=>{
 const next={...model,version:'4',run_id:'trained-4'},catalog={registry_id:'BeeOPS_Horizon_Weight',active_version:'4',versions:[next]};
 let catalogCalls=0,finishFirst,failSecond;
 const f=fixture(url=>{
  if(url==='/monitoring/retraining')return activity(job('completed'),{active_model:next});
  if(url==='/models/horizon'){
   catalogCalls++;
   if(catalogCalls===1)return new Promise(resolve=>{finishFirst=resolve;});
   if(catalogCalls===2)return new Promise((resolve,reject)=>{failSecond=reject;});
   return catalog;
  }
  return [];
 });
 const globalPoll=f.run('loadRetrainingActivity()');await flush();
 const overlappingCatalog=f.run('loadHorizonCatalog()');
 finishFirst(catalog);await globalPoll;
 failSecond(new Error('temporary catalog failure'));await overlappingCatalog;
 assert.equal(f.state.horizonCatalogLoaded,false);
 await f.run('loadRetrainingActivity()');
 assert.equal(catalogCalls,3);assert.equal(f.state.horizonCatalogLoaded,true);assert.equal(f.state.horizonVersion,'4');
});

test('completed deployment displays reporting warnings without reverting to running or failed status',()=>{
 const f=fixture(),warnings=['Reference persistence: disk full','Runtime cache reload: unavailable'];
 f.render(activity(job()));f.render(activity(job('completed',{reporting_warnings:warnings}),{active_model:{...model,version:'4',run_id:'trained-4'}}));
 assert.equal(f.node('retraining-panel').dataset.status,'completed');assert.match(f.node('retraining-title').textContent,/완료/);
 assert.equal(f.node('retraining-error').hidden,false);
 for(const warning of warnings)assert(f.node('retraining-error').textContent.includes(warning));
 assert.match(f.node('retraining-result').textContent,/4.*적용 완료/);
 assert.equal(f.node('retraining-progress').value,8);assert.equal(f.node('retraining-progress').max,8);assert.match(f.node('retraining-progress-label').textContent,/8\s*\/\s*8/);
});


for(const status of ['completed','failed','skipped','interrupted'])test(`initial ${status} history never replays a notification`,()=>{
 const f=fixture();f.render(activity(job(status)));assert.equal(f.node('retraining-panel').hidden,true);
 f.advance(5000);f.render(activity(job(status)));assert.equal(f.node('retraining-panel').hidden,true);
});

test('legacy training never enters the main ensemble notification',()=>{
 const legacy={job_id:'legacy:other:snapshot',workspace_id:'other',source:'legacy_lstm',status:'running',steps:[],requested_at:clock};
 const f=fixture();f.render(activity(job('completed'),{legacy_jobs:[legacy]}));assert.equal(f.node('retraining-panel').hidden,true);
 f.render(activity(job(),{legacy_jobs:[legacy]}));assert.match(f.node('retraining-title').textContent,/앙상블/);assert.doesNotMatch(f.text(),/기존 LSTM/);
});

test('a failed initial poll never creates a fake training notification',async()=>{
 const f=fixture(()=>{throw Error('offline');});await f.run('loadRetrainingActivity()');
 assert.equal(f.node('retraining-panel').hidden,true);assert.equal(f.state.retrainingActivity,null);
});

test('a live terminal notification hides after 15 seconds without replay on later polls',()=>{
 const f=fixture();f.render(activity(job()));f.render(activity(job('completed')));
 f.advance(14000);f.render(activity(job('completed')));assert.equal(f.node('retraining-panel').hidden,false);
 f.advance(1001);assert.equal(f.node('retraining-panel').hidden,true);
 f.render(activity(job('completed')));assert.equal(f.node('retraining-panel').hidden,true);
 assert.equal(f.state.retrainingActivity.latest_job.status,'completed');
});

test('a fresh job completed between polls appears once while old history stays hidden',()=>{
 const f=fixture();f.render(activity(job('completed')));f.advance(5000);
 const fresh=job('completed',{job_id:'fresh',requested_at:'2026-10-01T08:30:02Z',completed_at:'2026-10-01T08:30:04Z'});
 f.render(activity(fresh));assert.equal(f.node('retraining-panel').hidden,false);
 f.advance(15001);f.render(activity(fresh));assert.equal(f.node('retraining-panel').hidden,true);
});

test('a new active job survives an earlier completion notification timer',()=>{
 const f=fixture();f.render(activity(job()));f.render(activity(job('completed')));f.advance(10000);
 f.render(activity(job('running',{job_id:'next'})));f.advance(6000);
 assert.equal(f.node('retraining-panel').hidden,false);assert.equal(f.node('retraining-panel').dataset.status,'running');
});

test('manual ensemble request polls real monitoring and never reports queued acceptance as completion',async()=>{
 const f=fixture(url=>url.startsWith('/models/retrain')?{status:'queued',job:job('queued')}:activity(job('queued')));
 f.state.health={model_loaded:true,data:{rows:336},retraining:{status:'completed'}};
 f.run('refreshWorkspace=async()=>{};');await f.run('runOperation()');
 assert.equal(f.calls[0].url,'/models/retrain?workspace_id=SELECTED-HIVE');assert.equal(f.calls[0].options.method,'POST');
 assert(f.calls.some(call=>call.url==='/monitoring/retraining'));
 assert.match(f.node('operation-result').textContent,/접수|대기/);assert.doesNotMatch(f.node('operation-result').textContent,/학습 완료|적용 완료|배포 완료/);
 assert.equal(f.node('retrain-button').textContent,'앙상블 재학습');assert.equal(f.node('retrain-button').disabled,true);
});

test('an already evaluated manual snapshot does not invent a new completion',async()=>{
 const f=fixture(url=>url.startsWith('/models/retrain')?{status:'already_evaluated',job:job('completed')}:activity(job('completed')));
 f.state.health={model_loaded:true,data:{rows:336},retraining:{status:'completed'}};
 f.run('refreshWorkspace=async()=>{};');await f.run('runOperation()');
 assert.match(f.node('operation-result').textContent,/현재 데이터.*기록/);assert.equal(f.node('retraining-panel').hidden,true);
});

test('ensemble controls ignore legacy activity and block actual common training',()=>{
 const f=fixture();f.state.health={model_loaded:true,data:{rows:336},retraining:{status:'running'}};
 f.render(activity(null));f.run('updateControls()');assert.equal(f.node('retrain-button').disabled,false);
 f.render(activity(job()));f.run('updateControls()');assert.equal(f.node('retrain-button').disabled,true);
});

test('model summary and catalog describe actual learned blend weights',()=>{
 const f=fixture();f.state.horizonCatalog.versions[0].weights={tirex2:.65,lstm:.35};
 f.run('renderAutomaticSummary();renderHorizonModels();');
 assert.match(f.node('stat-model-note').textContent,/65(?:\.0)?%.*35(?:\.0)?%/);
 assert.doesNotMatch(html,/TiRex-2 90% \+ LSTM 10%/);
 assert.doesNotMatch(f.node('model-operation-description').textContent,/고정한/);
});

test('historical result copy claims blend training only when its recorded components include it',()=>{
 const f=fixture();f.context.historical=job('completed');
 let result=f.run('renderTemperatureResult(historical)');assert.match(result,/LSTM/);assert.doesNotMatch(result,/LSTM과 결합 가중치를 학습/);
 f.context.historical.training={trained_components:['lstm','ensemble_weights'],new_weights:{tirex2:.65,lstm:.35}};
 result=f.run('renderTemperatureResult(historical)');assert.match(result,/LSTM과 결합 가중치를 학습/);assert.match(result,/65(?:\.0)?%.*35(?:\.0)?%/);
});

for(const [phase,label] of [['ensemble_calibration','앙상블 결합 비중 학습 중'],['held_out_evaluation','예측 오차 확인 중']])test(`completed LSTM epochs keep ${phase} visibly in progress`,()=>{
 const f=fixture();f.render(activity(job('running',{training_progress:{completed_epochs:8,total_epochs:8,training_windows:120,phase}})));
 assert.equal(f.node('retraining-panel').dataset.status,'running');assert.match(f.node('retraining-title').textContent,/재학습 중/);
 assert(f.node('retraining-progress-label').textContent.includes(label));assert.match(f.node('retraining-progress-label').textContent,/LSTM.*8\s*\/\s*8/);
 assert.equal(f.node('retraining-badge').textContent,label);assert.doesNotMatch(f.node('retraining-result').textContent,/적용 완료/);
});

function importFixture({requested=true, outcome='completed', monitoringJob=true}={}) {
 const trained=job(outcome,{job_id:'heatwave-ensemble',workspace_id:'OTHER-HIVE'});
 const imported={job_id:'heatwave-import',workspace_id:'OTHER-HIVE',filename:'02_heatwave.csv',status:'completed',stage:'completed',progress:100,
  result:{ingest:{inserted:336,duplicate:168,rows:504},retraining_requested:requested,retraining_job_id:requested?trained.job_id:null,temperature_monitoring:{job:monitoringJob?trained:null}}};
 const f=fixture(url=>url==='/imports/jobs/heatwave-import'?imported:activity(monitoringJob?trained:null));
 f.run("loadWorkspaces=async()=>{};loadCatalog=()=>{};selectWorkspace=id=>{state.workspaceId=id;};setView=view=>{state.view=view;};refreshWorkspace=()=>{};");
 Object.assign(f.state,{activeJob:{job_id:'heatwave-import',status:'running'},jobSequence:1,navigationRevision:1,view:'data'});
 f.state.importNavigation.set('heatwave-import',{revision:1,workspace:'OTHER-HIVE'});
 return {f,trained,imported};
}

test('built-in heatwave preview uses the server-validated scenario without committing data',async()=>{
 const preview={preview_id:'heatwave-preview',filename:'02_heatwave.csv',can_commit:true,workspace_id:'OTHER-HIVE',hive_id:'OTHER-HIVE',row_count:504,insert_count:336,duplicate_count:168,conflict_count:0,sample:[],errors:[],warnings:[],temperature_practice:{phase:'heatwave',includes_baseline:true,baseline_rows:336}};
 const f=fixture(()=>preview);
 await f.run("previewTemperatureFile('02_heatwave.csv')");
 assert.equal(f.state.preview?.preview_id,'heatwave-preview');
 assert.equal(f.calls.length,1);assert.equal(f.calls[0].url,'/data/simulations/temperature/preview');assert.equal(f.calls[0].options.method,'POST');
 assert.deepEqual(JSON.parse(f.calls[0].options.body),{filename:'02_heatwave.csv'});
 assert.match(f.node('commit-button').textContent,/재학습/);
 assert.match(f.node('commit-explanation').textContent,/정상|기준/);
});

test('import-linked real completion is shown on the dashboard even before the first global poll',async()=>{
 const {f}=importFixture();
 await f.run('pollActiveJob()');await flush();
 assert.equal(f.state.view,'overview');assert.equal(f.state.workspaceId,'OTHER-HIVE');
 assert.equal(f.node('retraining-panel').hidden,false);
 assert.equal(f.node('retraining-panel').dataset.status,'completed');assert.match(f.node('retraining-title').textContent,/재학습 완료/);
 assert(f.calls.some(c=>c.url==='/monitoring/retraining'));
 f.advance(15001);await f.run('pollActiveJob()');await f.run('loadRetrainingActivity()');assert.equal(f.node('retraining-panel').hidden,true);
});

for(const monitoringJob of [false,true])test(`import without a new training job does not announce completion (historical job ${monitoringJob})`,async()=>{
 const {f,trained}=importFixture({requested:false,monitoringJob});
 f.render(activity(monitoringJob?trained:null));await f.run('pollActiveJob()');await flush();
 assert.equal(f.state.view,'overview');assert.equal(f.node('retraining-panel').hidden,true);
});

test('failed import-linked training is reported as failed, never as completed',async()=>{
 const {f}=importFixture({outcome:'failed'});await f.run('pollActiveJob()');await flush();
 assert.equal(f.node('retraining-panel').hidden,false);assert.equal(f.node('retraining-panel').dataset.status,'failed');
 assert.match(f.node('retraining-title').textContent,/실패/);assert.doesNotMatch(f.node('retraining-title').textContent,/완료/);
});

test('completion is retained when another training job is still active',()=>{
 const f=fixture(),a=job('running',{job_id:'first'}),b=job('queued',{job_id:'second'});
 f.render(activity(a,{queued_jobs:[b],jobs:[a,b]}));
 const doneA=job('completed',{job_id:'first'}),runningB=job('running',{job_id:'second'});
 f.render(activity(runningB,{jobs:[runningB,doneA]}));assert.equal(f.node('retraining-panel').dataset.status,'running');
 const doneB=job('completed',{job_id:'second',completed_at:'2026-10-01T08:30:01Z'});
 f.render(activity(doneB,{jobs:[doneB,doneA]}));
 const shown=new Set([f.state.retrainingJobId]);
 f.advance(15001);f.render(activity(doneB,{jobs:[doneB,doneA]}));if(!f.node('retraining-panel').hidden)shown.add(f.state.retrainingJobId);
 assert.deepEqual([...shown].sort(),['first','second']);
 f.advance(15001);f.render(activity(doneB,{jobs:[doneB,doneA]}));assert.equal(f.node('retraining-panel').hidden,true);
});


test('an observed pending job survives reload but a delivered completion does not replay',()=>{
 const storage=new Map(),first=fixture(undefined,storage);first.render(activity(job()));
 const reloaded=fixture(undefined,storage);reloaded.render(activity(job('completed')));
 assert.equal(reloaded.node('retraining-panel').hidden,false);assert.match(reloaded.node('retraining-title').textContent,/완료/);
 const again=fixture(undefined,storage);again.render(activity(job('completed')));assert.equal(again.node('retraining-panel').hidden,true);
});

test('a completion received in a hidden tab waits until the user returns',()=>{
 const f=fixture();f.render(activity(job()));f.context.document.hidden=true;f.render(activity(job('completed')));
 f.advance(20000);f.context.document.hidden=false;f.render(activity(job('completed')));
 assert.equal(f.node('retraining-panel').hidden,false);assert.match(f.node('retraining-title').textContent,/완료/);
 f.advance(15001);f.render(activity(job('completed')));assert.equal(f.node('retraining-panel').hidden,true);
});

test('reload after saving an import still correlates completion before the first training poll',async()=>{
 const {f,trained}=importFixture();
 f.state.importNavigation.clear();
 f.state.retrainingPendingImports=new Set(['heatwave-import']);
 f.render(activity(trained));assert.equal(f.node('retraining-panel').hidden,true);
 await f.run('pollActiveJob()');await flush();
 assert.equal(f.node('retraining-panel').hidden,false);assert.match(f.node('retraining-title').textContent,/완료/);
 assert.equal(f.state.retrainingPendingImports.size,0);
});

test('a late import result cannot replay a completion already displayed by global polling',async()=>{
 const {f,trained}=importFixture();
 f.render(activity({...trained,status:'running'}));f.render(activity(trained));
 f.advance(15001);f.render(activity(trained));assert.equal(f.node('retraining-panel').hidden,true);
 await f.run('pollActiveJob()');await flush();assert.equal(f.node('retraining-panel').hidden,true);
});
