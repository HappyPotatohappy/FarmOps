/* Explicit heatwave practice run lifecycle at real frontend HTTP/render boundaries. */
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
const source=fs.readFileSync(path.join(__dirname,'../../app/static/app.js'),'utf8').split('init();setInterval')[0];
const html=fs.readFileSync(path.join(__dirname,'../../app/static/index.html'),'utf8');
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
 for(const match of html.matchAll(/\bid="([^"]+)"/g))node(match[1]);[...panelIds,'retraining-acknowledge'].forEach(node);
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

const HIVE='TEMP-DRIFT-PRACTICE';
const STORAGE='beeops.practice.run.v1';
function scenario({storage=new Map(),intercept=null}={}){
 let count=0,currentActivity=activity(null);
 const imports=new Map(),trainingJobs=new Map();
 const f=fixture(async(url,options)=>{
  if(intercept){const result=intercept(url,options);if(result!==undefined)return await result;}
  if(url==='/imports/commit'){
   const queued={job_id:'practice-import-'+(++count),workspace_id:HIVE,filename:'02_heatwave.csv',status:'queued',stage:'queued',progress:0};
   imports.set(queued.job_id,queued);return queued;
  }
  if(url.startsWith('/imports/jobs/'))return imports.get(decodeURIComponent(url.split('/').at(-1)));
  // Registration may lag behind the accepted import. Keep this real catalog empty of TEMP.
  if(url==='/workspaces')return {workspaces:[{workspace_id:'SELECTED-HIVE',name:'기존 벌통'}],default_workspace_id:'SELECTED-HIVE'};
  if(url.startsWith('/monitoring/retraining/jobs/'))return trainingJobs.get(decodeURIComponent(url.split('/').at(-1)));
  if(url==='/monitoring/retraining')return currentActivity;
  return [];
 },storage);
 f.run('refreshWorkspace=async()=>{};loadCatalog=async()=>{};loadHorizonCatalog=async()=>{};loadRows=async()=>{};');
 const preview=(id='practice-preview-1',heat=true)=>{
  f.state.preview={preview_id:id,filename:heat?'02_heatwave.csv':'ordinary.csv',workspace_id:HIVE,hive_id:HIVE,
   workspace_name:'고온 변화 실습',can_commit:true,row_count:504,insert_count:504,duplicate_count:0,conflict_count:0,
   sample:[],warnings:[],errors:[],...(heat?{temperature_practice:{phase:'heatwave',includes_baseline:true}}:{})};
 };
 const finish=(id='practice-import-1',trainingId='practice-training-1',requested=true)=>{
  const trained=job('completed',{job_id:trainingId,workspace_id:HIVE});
  imports.set(id,{...imports.get(id),status:'completed',stage:'completed',progress:100,
   result:{ingest:{inserted:504,duplicate:0,rows:504},retraining_requested:requested,retraining_job_id:trainingId}});
  trainingJobs.set(trained.job_id,trained);currentActivity=activity(trained);return trained;
 };
 preview();
 return {f,storage,imports,preview,finish,setActivity:value=>{currentActivity=value;for(const item of value.jobs||[])trainingJobs.set(item.job_id,item);}};
}

test('accepted heatwave commit immediately selects TEMP and shows honest preparation before registration',async()=>{
 const {f,storage}=scenario();
 await f.run('commitPreview()');await flush();
 assert.equal(f.state.workspaceId,HIVE);assert.equal(f.state.view,'overview');
 assert.equal(f.state.practiceRun?.import_job_id,'practice-import-1');
 assert.equal(f.state.practiceRun?.retraining_job_id,null);
 assert.equal(JSON.parse(storage.get(STORAGE)).import_job_id,'practice-import-1');
 assert.equal(f.node('retraining-panel').hidden,false);
 assert.notEqual(f.node('retraining-panel').dataset.status,'running');
 assert.notEqual(f.node('retraining-panel').dataset.status,'completed');
 assert.match(f.node('retraining-title').textContent,/준비|저장|접수/);
 assert.equal(f.node('retraining-progress').hidden,true);
 assert.match(f.node('workspace-select').innerHTML,/TEMP-DRIFT-PRACTICE/);
 await f.run('loadWorkspaces()');
 assert.equal(f.state.workspaceId,HIVE,'a fresh pre-registration catalog must preserve the selected pending hive');
});

test('late pre-commit workspace response cannot undo immediate practice selection',async()=>{
 let resolve,workspaceCalls=0;
 const {f}=scenario({intercept:url=>url==='/workspaces'&&++workspaceCalls===1?new Promise(done=>{resolve=done;}):undefined});
 const pending=f.run('loadWorkspaces()');
 await f.run('commitPreview()');await flush();
 resolve({workspaces:[{workspace_id:'SELECTED-HIVE',name:'기존 벌통'}],default_workspace_id:'SELECTED-HIVE'});
 await pending;
 assert.equal(f.state.workspaceId,HIVE);assert.equal(f.state.view,'overview');
});

test('fast actual practice completion stays visible past 15 seconds and restores after reload',async()=>{
 const {f,storage,finish}=scenario();
 await f.run('commitPreview()');await flush();
 const trained=finish();
 await f.run('pollActiveJob()');await flush();await f.run('loadRetrainingActivity()');
 assert.equal(f.state.practiceRun?.retraining_job_id,trained.job_id);
 assert.equal(f.node('retraining-panel').dataset.status,'completed');
 f.advance(60000);f.render(activity(trained));
 assert.equal(f.node('retraining-panel').hidden,false);
 const restored=scenario({storage});
 restored.setActivity(activity(trained));await restored.f.run('loadRetrainingActivity()');
 assert.equal(restored.f.state.practiceRun?.import_job_id,'practice-import-1');
 assert.equal(restored.f.node('retraining-panel').hidden,false);
 assert.equal(restored.f.node('retraining-panel').dataset.status,'completed');
});

test('a coalesced explicit practice run follows the real job ID even when no new worker was requested',async()=>{
 const {f,finish}=scenario();await f.run('commitPreview()');await flush();
 const trained=finish('practice-import-1','already-running-training',false);
 await f.run('pollActiveJob()');await flush();await f.run('loadRetrainingActivity()');
 assert.equal(f.state.practiceRun?.retraining_job_id,trained.job_id);
 assert.equal(f.node('retraining-panel').dataset.status,'completed');
});

test('a linked practice run displays only its actual queued and completed epoch states',async()=>{
 const {f,finish,setActivity}=scenario();await f.run('commitPreview()');await flush();
 finish();const queued=job('queued',{job_id:'practice-training-1',workspace_id:HIVE});setActivity(activity(queued));
 await f.run('pollActiveJob()');await flush();await f.run('loadRetrainingActivity()');
 assert.equal(f.node('retraining-panel').dataset.status,'queued');assert.match(f.node('retraining-title').textContent,/대기/);
 f.run('acknowledgePracticeRun()');assert.equal(f.state.practiceRun?.retraining_job_id,queued.job_id,'a live run cannot be acknowledged away');
 const running=job('running',{job_id:queued.job_id,workspace_id:HIVE});setActivity(activity(running));
 await f.run('loadRetrainingActivity()');
 assert.equal(f.node('retraining-panel').dataset.status,'running');
 assert.equal(f.node('retraining-progress').value,3);assert.equal(f.node('retraining-progress').max,8);
});

test('reload while import is pending restores preparation and follows completion without navigation intent',async()=>{
 const first=scenario();await first.f.run('commitPreview()');await flush();
 const restored=scenario({storage:first.storage});
 restored.imports.set('practice-import-1',clone(first.imports.get('practice-import-1')));
 restored.f.state.workspaceId=null;
 await restored.f.run('init()');await flush();
 assert.equal(restored.f.state.importNavigation.size,0);
 assert.equal(restored.f.state.workspaceId,HIVE);assert.equal(restored.f.node('retraining-panel').hidden,false);
 assert.equal(restored.f.node('retraining-panel').dataset.status,'preparing');
 restored.finish();await restored.f.run('loadRetrainingActivity()');
 assert.equal(restored.f.state.practiceRun?.retraining_job_id,'practice-training-1');
 assert.equal(restored.f.node('retraining-panel').dataset.status,'completed');
});

test('a restored practice result remains available when it leaves the global recent job list',async()=>{
 const {f,storage,finish}=scenario();await f.run('commitPreview()');await flush();const trained=finish();
 await f.run('pollActiveJob()');await flush();await f.run('loadRetrainingActivity()');
 const restored=scenario({storage});restored.setActivity(activity(trained));restored.setActivity(activity(null));
 await restored.f.run('loadRetrainingActivity()');
 assert.equal(restored.f.node('retraining-panel').hidden,false);
 assert.equal(restored.f.node('retraining-panel').dataset.status,'completed');
 assert.equal(restored.f.state.practiceRun?.retraining_job_id,trained.job_id);
});

test('reload resolves a formerly running practice by exact ID when its completion aged out of recent history',async()=>{
 const first=scenario();await first.f.run('commitPreview()');await flush();const trained=first.finish();
 await first.f.run('pollActiveJob()');await flush();await first.f.run('loadRetrainingActivity()');
 const saved=JSON.parse(first.storage.get(STORAGE));saved.last_job=job('running',{job_id:trained.job_id,workspace_id:HIVE});
 first.storage.set(STORAGE,JSON.stringify(saved));
 const restored=scenario({storage:first.storage});restored.setActivity(activity(trained));restored.setActivity(activity(null));
 await restored.f.run('loadRetrainingActivity()');
 assert(restored.f.calls.some(call=>call.url==='/monitoring/retraining/jobs/'+trained.job_id));
 assert.equal(restored.f.node('retraining-panel').dataset.status,'completed');
 assert.equal(restored.f.state.practiceRun.last_job.job_id,trained.job_id);
});

test('a later deliberate hive choice survives practice completion while its result remains visible',async()=>{
 const {f,finish}=scenario();await f.run('commitPreview()');await flush();
 f.run("state.navigationRevision++;selectWorkspace('SELECTED-HIVE');setView('models')");
 finish();await f.run('pollActiveJob()');await flush();await f.run('loadRetrainingActivity()');
 assert.equal(f.state.workspaceId,'SELECTED-HIVE');assert.equal(f.state.view,'models');
 assert.equal(f.node('retraining-panel').hidden,false);assert.equal(f.node('retraining-panel').dataset.status,'completed');
});

test('a new accepted practice run owns its new import ID and ignores an older pending response',async()=>{
 let delayOld=false,resolveOld;
 const s=scenario({intercept:url=>delayOld&&url==='/imports/jobs/practice-import-1'?new Promise(done=>{resolveOld=done;}):undefined});
 const {f,preview}=s;await f.run('commitPreview()');await flush();
 delayOld=true;const oldPoll=f.run('pollActiveJob()');
 preview('practice-preview-2');await f.run('commitPreview()');await flush();
 assert.equal(f.state.practiceRun?.import_job_id,'practice-import-2');
 resolveOld({...s.imports.get('practice-import-1'),status:'completed',result:{ingest:{inserted:504},retraining_requested:true,retraining_job_id:'old-training'}});
 await oldPoll;
 assert.equal(f.state.practiceRun?.import_job_id,'practice-import-2');assert.notEqual(f.state.practiceRun?.retraining_job_id,'old-training');
 const next=s.finish('practice-import-2','new-training');
 await f.run('pollActiveJob()');await flush();await f.run('loadRetrainingActivity()');
 assert.equal(f.state.practiceRun?.retraining_job_id,next.job_id);
});

test('failed import without a real training job shows failure instead of endless preparation or success',async()=>{
 const {f,imports}=scenario();await f.run('commitPreview()');await flush();
 imports.set('practice-import-1',{...imports.get('practice-import-1'),status:'failed',stage:'failed',error:'canonical data conflict',result:{}});
 await f.run('pollActiveJob()');await flush();
 assert.equal(f.node('retraining-panel').hidden,false);assert.equal(f.node('retraining-panel').dataset.status,'failed');
 assert.match(f.text(),/canonical data conflict/);assert.doesNotMatch(f.node('retraining-title').textContent,/재학습 완료|재학습 중/);
 assert.equal(f.state.practiceRun?.retraining_job_id,null);
});

test('completed import without training correlation reports a request problem rather than preparing forever',async()=>{
 const {f,imports}=scenario();await f.run('commitPreview()');await flush();
 imports.set('practice-import-1',{...imports.get('practice-import-1'),status:'completed',stage:'completed',result:{ingest:{inserted:504},retraining_requested:false,retraining_job_id:null}});
 await f.run('pollActiveJob()');await flush();
 assert.equal(f.node('retraining-panel').dataset.status,'failed');assert.match(f.node('retraining-title').textContent,/요청 실패/);
 assert.equal(f.node('retraining-acknowledge').hidden,false);assert.equal(f.node('retraining-progress').hidden,true);
});

test('acknowledging a failed unregistered import removes its temporary hive selection',async()=>{
 const {f,imports}=scenario();await f.run('commitPreview()');await flush();
 imports.set('practice-import-1',{...imports.get('practice-import-1'),status:'failed',stage:'failed',error:'workspace creation failed',result:{}});
 await f.run('pollActiveJob()');await flush();
 assert.equal(f.run('current()?.pending_import'),true);
 f.run('acknowledgePracticeRun()');await flush();
 assert.equal(f.state.practiceRun,null);
 assert.equal(f.state.workspaceId,'SELECTED-HIVE');
 assert.equal(f.state.workspaces.some(item=>item.workspace_id===HIVE&&item.pending_import),false);
});

test('acknowledging an actual practice result clears the pin and does not replay after reload',async()=>{
 const {f,storage,finish}=scenario();await f.run('commitPreview()');await flush();const trained=finish();
 await f.run('pollActiveJob()');await flush();await f.run('loadRetrainingActivity()');
 assert.equal(f.node('retraining-acknowledge').hidden,false);
 const acknowledge=f.events.get('retraining-acknowledge:click');assert.equal(typeof acknowledge,'function');acknowledge();
 assert.equal(f.state.practiceRun,null);assert.equal(f.node('retraining-panel').hidden,true);
 const restored=scenario({storage});restored.setActivity(activity(trained));await restored.f.run('loadRetrainingActivity()');
 assert.equal(restored.f.state.practiceRun,null);assert.equal(restored.f.node('retraining-panel').hidden,true);
});

test('an in-flight global response cannot revive an acknowledged practice result',async()=>{
 let hold=false,resolve;
 const s=scenario({intercept:url=>hold&&url==='/monitoring/retraining'?new Promise(done=>{resolve=done;}):undefined});
 await s.f.run('commitPreview()');await flush();const trained=s.finish();
 await s.f.run('pollActiveJob()');await flush();await s.f.run('loadRetrainingActivity()');
 hold=true;const pending=s.f.run('loadRetrainingActivity()');
 s.f.run('acknowledgePracticeRun()');assert.equal(s.f.state.practiceRun,null);
 resolve(activity(trained));await pending;
 assert.equal(s.f.state.practiceRun,null);assert.equal(s.f.node('retraining-panel').hidden,true);
});

test('ordinary CSV commit does not create a persistent practice run',async()=>{
 const {f,storage,preview}=scenario();preview('ordinary-preview',false);
 await f.run('commitPreview()');await flush();
 assert.equal(f.state.practiceRun??null,null);assert.equal(storage.has(STORAGE),false);
 assert.equal(f.state.workspaceId,'SELECTED-HIVE');
});
