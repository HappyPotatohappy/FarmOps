/* Read-only dashboard context and selected CSV integration regressions. */
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
const source=fs.readFileSync(path.join(__dirname,'../../app/static/app.js'),'utf8').split('init();setInterval')[0];
const html=fs.readFileSync(path.join(__dirname,'../../app/static/index.html'),'utf8');
const at=hour=>new Date(Date.UTC(2025,0,1,hour)).toISOString();
const cases=['apis2','vecauce3','meliponini1'].map((id,index)=>({id,hive_id:'trend_'+id,workspace_id:'trend_'+id,name:['브라질 Apis 2','Vecauce 3','브라질 무침벌 1'][index],label:['브라질 Apis 2','Vecauce 3','브라질 무침벌 1'][index],kind:'public_real',rows:609,start:at(0),end:at(608),download_url:'/data/trend-cases/'+id+'.csv'}));
const observations=Array.from({length:609},(_,i)=>({timestamp:at(i),hive_id:cases[0].hive_id,weight_kg:50+i/100,temperature_c:24,event:'normal'}));
const model={registry_id:'BeeOPS_Horizon_Weight',version:'3',run_id:'trained-3',context_hours:168,max_horizon_hours:168};
function handler(url){const target=new URL(url,'http://beeops.test'),hive=target.searchParams.get('workspace_id')||cases[0].hive_id;if(target.pathname==='/workspaces')return {workspaces:cases,default_workspace_id:cases[0].workspace_id};if(target.pathname==='/health')return {workspace_id:hive,version:'1',model_loaded:true,data:{rows:609,hive_id:hive,snapshot_id:'snapshot-'+hive},quality:{},retraining:{},demo:{}};if(target.pathname==='/observations')return observations;if(target.pathname==='/config')return {runtime_type:'public_real'};if(target.pathname==='/monitoring/temperature')return {workspace_id:hive,enabled:false};if(target.pathname==='/data/samples')return {samples:cases,default_sample_id:cases[0].id};return [];}
function fixture(handler){
 const nodes=new Map(),calls=[],events=new Map();
 function node(id){if(!nodes.has(id)){let content='';const element={id,textContent:'',hidden:false,disabled:false,value:'',dataset:{},style:{},clientWidth:850,clientHeight:300,className:'',classList:{add(){},remove(){},toggle(){},contains(){return false;}},lastElementChild:{textContent:''},addEventListener(){},setAttribute(){},removeAttribute(){},append(){},scrollIntoView(){}};Object.defineProperty(element,'innerHTML',{get(){return content;},set(value){content=value;for(const match of value.matchAll(/\bid="([^"]+)"/g))node(match[1]);}});nodes.set(id,element);}return nodes.get(id);}
 for(const match of html.matchAll(/\bid="([^"]+)"/g))node(match[1]);
 const context={console,URL,AbortController,FormData,File,Blob,Map,Set,Date,Number,Math,JSON,Promise,setTimeout,clearTimeout,location:{origin:'http://beeops.test',hash:'#overview'},history:{replaceState(_state,_title,hash){context.location.hash=hash;}},localStorage:{getItem(){return null;},setItem(){}},window:{addEventListener(name,callback){events.set(name,callback);},scrollTo(){}},document:{getElementById:id=>nodes.get(id)||null,querySelectorAll(){return [];},createElement:type=>node('created-'+type),addEventListener(){}},fetch:async(url,options)=>{calls.push({url,options});const payload=await handler(url,options);return {ok:true,text:async()=>JSON.stringify(payload),blob:async()=>payload instanceof Blob?payload:new Blob([JSON.stringify(payload)])};}};
 vm.createContext(context);vm.runInContext(source,context);return {calls,nodes,node,events,context,state:vm.runInContext('state',context),run:code=>vm.runInContext(code,context)};
}
const flush=()=>new Promise(resolve=>setTimeout(resolve,0));

test('stale saved hive falls back to the selected default and displays only friendly workspace names',async()=>{
 const f=fixture(handler);f.context.localStorage.getItem=()=> 'TEMP-DRIFT-DEMO';await f.run('loadWorkspaces()');await flush();
 assert.equal(f.state.workspaceId,cases[0].workspace_id);assert.equal(f.state.workspaces.length,3);
 const dropdown=f.node('workspace-select').innerHTML;
 for(const item of cases)assert(dropdown.includes('>'+item.name+'</option>'));
 assert(!dropdown.includes('TEMP-DRIFT-DEMO'));
 assert(f.calls.every(call=>call.options.method==='GET'));
});

test('the main dashboard requests the complete observed interval and aligns current-model history with it',async()=>{
 const history={workspace_id:cases[0].workspace_id,snapshot_id:'snapshot-'+cases[0].workspace_id,model,predictions:observations.slice(-48).map((row,index)=>({target_timestamp:row.timestamp,origin_timestamp:at(560+index),predicted_weight_kg:row.weight_kg-.125})),metrics:{count:48,mae_kg:.125}};
 const f=fixture(url=>url.startsWith('/forecast/history')?history:handler(url));
 Object.assign(f.state,{workspaceId:cases[0].workspace_id,workspaces:cases});await f.run('refreshWorkspace()');
 Object.assign(f.state,{horizonCatalog:{versions:[model]},horizonCatalogLoaded:true,horizonVersion:'3'});await f.run('loadHistoryReport()');
 assert(f.calls.some(call=>call.url==='/observations?limit=720&workspace_id=trend_apis2'));
 assert(f.calls.some(call=>call.url==='/forecast/history?limit=720&workspace_id=trend_apis2'));
 assert.equal(f.state.observations.length,609);assert.equal(f.state.historyReport.predictions.length,48);
 const chart=f.node('observation-chart').innerHTML;
 assert.equal((chart.match(/fill="#3b5960"/g)||[]).length,609);assert.equal((chart.match(/fill="#d79e2c"/g)||[]).length,48);
 assert.match(chart,/50\.000/);assert.match(chart,/56\.080/);assert.match(f.node('history-metrics').textContent,/48쌍/);
});

test('the old gallery hash resolves to the main dashboard without gallery requests',async()=>{
 const f=fixture(handler);f.run("setView('trend-cases')");await flush();assert.equal(f.state.view,'overview');assert.equal(f.context.location.hash,'#overview');
 assert(!f.calls.some(call=>call.url.startsWith('/data/trend-cases')||call.url.startsWith('/analysis/trend-cases')));
});

test('the selected observed CSV stays in the existing data preview flow without automatic storage',async()=>{
 let uploaded;const f=fixture((url,options)=>{if(url==='/data/trend-cases/vecauce3.csv')return new Blob(['timestamp,hive_id,weight_kg,temperature_c,event\n2025-01-01T00:00:00Z,trend_vecauce3,50,24,normal\n']);if(url==='/imports/preview'){uploaded=options.body.get('file');return {preview_id:'selected',filename:uploaded.name,can_commit:true,workspace_id:'trend_vecauce3',hive_id:'trend_vecauce3',row_count:1,insert_count:0,duplicate_count:1,conflict_count:0,sample:[],errors:[],warnings:[]};}return handler(url);});
 await f.run('loadSamples()');assert.equal(f.state.samples.length,3);assert(f.calls.every(call=>call.options.method==='GET'));
 f.state.sampleId='vecauce3';f.run('renderSampleSelection()');assert.equal(f.node('sample-download-link').href,'/data/trend-cases/vecauce3.csv');await f.run('previewSelectedSample()');
 assert.equal(uploaded.name,'vecauce3.csv');assert.match(await uploaded.text(),/trend_vecauce3/);assert.equal(f.state.preview.preview_id,'selected');assert(!f.calls.some(call=>call.url==='/imports/commit'));
});

test('the data catalog accepts the selected local CSV routes and excludes external download destinations',async()=>{
 const f=fixture(()=>({samples:[...cases,{id:'outside',download_url:'https://outside.invalid/data/trend-cases/apis2.csv'}],default_sample_id:'apis2'}));await f.run('loadSamples()');
 assert.deepEqual(Array.from(f.state.samples,item=>item.id),['apis2','vecauce3','meliponini1']);assert.equal(f.state.sampleId,'apis2');
});

test('a failed initial CSV catalog does not offer an unrelated legacy dataset',async()=>{
 const f=fixture(()=>{throw Error('Catalog unavailable');});await f.run('loadSamples()');assert.equal(f.state.samples.length,0);assert.equal(f.state.sampleId,null);assert.equal(f.node('sample-download-link').hidden,true);assert.equal(f.node('sample-preview-button').disabled,true);assert.match(f.node('sample-catalog-error').textContent,/Catalog unavailable/);
});

for(const visible of [false,true])test(`startup ${visible?'restores the visible hive import job':'discards a hidden legacy hive import job'}`,async()=>{
 const job={job_id:'saved-job',workspace_id:visible?cases[0].workspace_id:'TEMP-DRIFT-DEMO',status:'queued',stage:'queued',progress:0};
 const f=fixture(url=>url==='/imports/jobs/saved-job'?job:handler(url));let removed=false;
 f.context.localStorage.getItem=key=>key==='beeops.import.job'?'saved-job':null;
 f.context.localStorage.removeItem=key=>{if(key==='beeops.import.job')removed=true;};
 await f.run('init()');await flush();
 assert.equal(f.state.activeJob?.job_id||null,visible?'saved-job':null);assert.equal(removed,!visible);assert.equal(f.state.workspaceId,cases[0].workspace_id);
});

for(const source of ['공개 관측: 브라질 Apis 2 · 12–1월','사용자 CSV: observations_2025-01.csv'])test(`health source keeps provenance while using the current hive name: ${source}`,()=>{
 const f=fixture(handler);Object.assign(f.state,{workspaceId:cases[0].workspace_id,workspaces:cases});
 f.context.health={...handler('/health'),data:{...handler('/health').data,source}};f.run('renderHealth(health)');
 assert.equal(f.node('overview-source').textContent,source.startsWith('공개 관측: ')?'공개 관측: 브라질 Apis 2':source);
 assert.equal(f.state.health.data.source,source);
});
