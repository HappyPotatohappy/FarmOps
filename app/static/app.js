'use strict';
const $=id=>document.getElementById(id);
const esc=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const finite=v=>v!==null&&v!==undefined&&v!==''&&Number.isFinite(Number(v));
const fmt=(v,d=3)=>finite(v)?Number(v).toLocaleString('ko-KR',{minimumFractionDigits:d,maximumFractionDigits:d}):'—';
const when=(v,full=false)=>{if(!v)return '—';const date=new Date(v);if(Number.isNaN(date.getTime()))return String(v);return date.toLocaleString('ko-KR',{timeZone:'UTC',year:full?'numeric':undefined,month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',hour12:false});};
const names={queued:'대기 중',running:'진행 중',completed:'완료',partial:'일부 처리 완료',failed:'실패',interrupted:'중단됨',idle:'대기 중',pending:'대기 중',skipped:'해당 없음',ok:'정상',degraded:'품질 확인 필요',underperforming:'기준선 미달',insufficient_data:'정답 수집 중',already_ready:'기존 모델 준비됨',initialized:'모델 준비 완료',quality_rejected:'품질 게이트 미통과',unavailable:'준비 필요'};
const statusName=s=>names[s]||s||'확인 중';
const terminal=s=>['completed','partial','failed','interrupted'].includes(s);
const badge=(text,tone='')=>`<span class="badge ${tone}">${esc(text)}</span>`;
const kindName=k=>({synthetic:'합성 교육',public_real:'공개 실측',single_hive:'개별 벌통',real:'공개 실측'}[k]||'벌통 기록');
const state={workspaceId:null,workspaces:[],epoch:0,view:'overview',health:null,config:null,observations:[],observationsLoaded:false,predictions:[],catalog:[],catalogLoaded:false,modelFilter:'current',jobs:[],page:1,totalRows:0,preview:null,file:null,activeJob:null,rollback:null,busy:new Set(),controllers:new Set(),refreshEpoch:null,catalogSequence:0,previewSequence:0,jobSequence:0,jobOpenSequence:0,rowsSequence:0,lastJobStatus:null,workspaceSequence:0,refreshedJobs:new Set(),forecastReport:null,comparisonReport:null,forecastSequence:0,comparisonSequence:0,forecastKey:null,forecastSettledKey:null,forecastPending:null,comparisonVersion:'',predictionVersion:'',predictionSelectionRevision:0,navigationRevision:0,importNavigation:new Map(),fileSelectionRevision:0,samples:[],sampleId:null,sampleCatalogSequence:0,hiddenCompletedJobId:null,noticeJobId:null};
Object.assign(state,{historyReport:null,historyKey:null,historySequence:0,historyPending:null,temperature:null,temperatureObservations:[],temperatureSequence:0});
Object.assign(state,{retrainingActivity:null,retrainingSequence:0,retrainingPending:null,retrainingError:'',retrainingOpen:false,retrainingJobId:null,retrainingModelKey:null,retrainingBaselineTime:null,retrainingObservedActive:restoreRetrainingIds('pending'),retrainingPendingImports:restoreRetrainingIds('imports'),retrainingDelivered:restoreRetrainingIds('delivered'),retrainingSeenTerminal:new Set(),retrainingTerminalQueue:[],retrainingNoticeJob:null,retrainingNoticeUntil:0,retrainingHideTimer:null});
const HORIZON_REGISTRY='BeeOPS_Horizon_Weight';
Object.assign(state,{practiceRun:restorePracticeRun()});
Object.assign(state,{horizonCatalog:null,horizonCatalogLoaded:false,horizonCatalogSequence:0,horizonVersion:'2',horizonHours:168,horizonReport:null,horizonSequence:0,horizonKey:null,horizonSettledKey:null,horizonPending:null});
function notice(message,error=false,jobId=null){state.noticeJobId=jobId;const el=$('global-notice');el.textContent='';const text=document.createElement('span');text.textContent=message;const close=document.createElement('button');close.className='notice-close';close.type='button';close.textContent='×';close.setAttribute('aria-label','알림 닫기');close.addEventListener('click',()=>{el.hidden=true;});el.append(text,close);el.className='notice'+(error?' error':'');el.hidden=false;}
function inline(id,message,error=false){const el=$(id);el.textContent=message;el.hidden=!message;if(el.classList.contains('inline-result'))el.classList.toggle('error',error);}
function empty(title,description=''){return `<div class="empty-state"><strong>${esc(title)}</strong>${description?`<span>${esc(description)}</span>`:''}</div>`;}
function scopeUrl(path,id=state.workspaceId){const url=new URL(path,location.origin);if(id)url.searchParams.set('workspace_id',id);return url.pathname+url.search;}
function errorText(error){let text=error.message||'요청을 완료하지 못했습니다.';if(error.status)text=`HTTP ${error.status} · ${text}`;return text;}
async function api(path,{workspace,method='GET',json,body,blob=false,timeout=25000,cancelOnSwitch=false}={}){
 const controller=new AbortController();const entry={controller,workspace};if(cancelOnSwitch)state.controllers.add(entry);const timer=setTimeout(()=>controller.abort('timeout'),timeout);
 try{const headers={};if(json!==undefined){headers['Content-Type']='application/json';body=JSON.stringify(json);}const response=await fetch(workspace?scopeUrl(path,workspace):path,{method,body,headers,signal:controller.signal,cache:'no-store'});
  if(blob&&response.ok)return response.blob();const text=await response.text();let data;try{data=JSON.parse(text);}catch{data=text;}
  if(!response.ok){const detail=data?.detail??data;const message=typeof detail==='string'?detail:detail?.message|| (Array.isArray(detail)?detail.map(e=>`${Array.isArray(e.loc)?e.loc.join(' · '):''}: ${e.msg||e.message||''}`).join('\n'):JSON.stringify(detail));const error=new Error(message||response.statusText);error.status=response.status;error.detail=detail;throw error;}return data;
 }catch(error){if(controller.signal.aborted){const e=new Error(controller.signal.reason==='workspace_changed'?'벌통이 전환되어 이전 조회를 취소했습니다.':'응답 대기 시간이 지났습니다. 상태를 새로고침하거나 다시 시도하세요.');e.cancelled=controller.signal.reason==='workspace_changed';throw e;}throw error;
 }finally{clearTimeout(timer);state.controllers.delete(entry);}
}
const current=()=>state.workspaces.find(w=>w.workspace_id===state.workspaceId);
const workspaceName=(id=state.workspaceId)=>state.workspaces.find(w=>w.workspace_id===id)?.name||id||'벌통';
const busy=key=>state.busy.has(key);
const sharedModelGroup=()=>state.catalog.find(g=>g.workspace_id===state.workspaceId&&g.model_scope==='shared')||state.catalog.find(g=>g.model_scope==='shared');
const sharedModelScope=()=>!!sharedModelGroup();
function updateControls(){const h=state.health,rows=h?.data?.rows??current()?.rows??0,ready=!!h?.model_loaded;const training=!!(state.retrainingActivity?.running_job||state.retrainingActivity?.queued_jobs?.length),legacyTraining=['running','queued'].includes(h?.retraining?.status),demo=h?.demo?.status==='running';
 $('retrain-button').disabled=!state.workspaceId||!h||training||demo||busy('operation:'+state.workspaceId);$('retrain-button').textContent='앙상블 재학습';$('choose-file-button').disabled=busy('preview')||busy('commit');$('sample-preview-button').disabled=!selectedSample()||busy('preview')||busy('sample')||busy('samplecatalog')||busy('commit');$('sample-select').disabled=busy('sample')||busy('samplecatalog');$('reload-samples-button').disabled=busy('sample')||busy('samplecatalog');
 $('commit-button').disabled=!state.preview?.can_commit||busy('commit')||busy('preview')||state.preview?.committed===true;$('commit-button').textContent=busy('commit')?'처리 작업을 만드는 중…':state.preview?.committed?'처리 작업 생성됨':state.preview?.temperature_practice?.phase==='heatwave'?'저장하고 재학습 →':'확인하고 저장 · 분석 시작 →';
 const synthetic=(state.config?.runtime_type||current()?.kind)==='synthetic';$('scenario-buttons').hidden=!synthetic;document.querySelectorAll('[data-scenario]').forEach(b=>b.disabled=!ready||legacyTraining||demo||busy('operation:'+state.workspaceId));
 $('model-operation-description').textContent='현재 벌통의 관측으로 LSTM과 결합 가중치를 학습합니다. TiRex-2는 기존 모델을 유지하며, 학습한 앙상블은 모든 벌통에 적용합니다.';
 $('scenario-description').textContent=synthetic?(sharedModelScope()?'정상·품질 저하·벌통·센서 시나리오는 합성 관측을 추가합니다. 배포 게이트 실패는 저장된 v3 차단 기록을 다시 확인하며 관측을 추가하지 않습니다.':'합성 교육 벌통의 원장에 관측을 추가하는 실행입니다.'):'현재 벌통에서는 합성 관측을 추가하지 않습니다. 합성 교육 벌통으로 전환하면 실행할 수 있습니다.';
 $('gate-fail-note').textContent=sharedModelScope()?'저장된 v3 차단 기록 확인':'후보 차단 · 기존 유지';
 $('previous-page').disabled=state.page<=1;$('next-page').disabled=state.page*25>=state.totalRows;
}
async function loadWorkspaces(){const seq=++state.workspaceSequence;const data=await api('/workspaces');if(seq!==state.workspaceSequence)return data;state.workspaces=practiceWorkspaces(data.workspaces||[]);const selected=state.workspaceId;$('workspace-select').innerHTML=state.workspaces.map(w=>`<option value="${esc(w.workspace_id)}">${esc(w.name||w.hive_id||w.workspace_id)}</option>`).join('')||'<option value="">등록된 벌통 없음</option>';if(selected&&state.workspaces.some(w=>w.workspace_id===selected))$('workspace-select').value=selected;else{let saved;try{saved=localStorage.getItem('beeops.workspace.v2');}catch{}const preferred=state.workspaces.some(w=>w.workspace_id===saved)?saved:state.workspaces.some(w=>w.workspace_id===data.default_workspace_id)?data.default_workspace_id:state.workspaces[0]?.workspace_id;if(preferred)selectWorkspace(preferred);}return data;}
function selectWorkspace(id){if(!id)return;$('overview-import-result').hidden=true;invalidateHistory();state.temperature=null;state.temperatureObservations=[];state.temperatureSequence++;resetTemperature();invalidateHorizonReport('벌통을 전환하여 다일 예측을 확인합니다.');state.workspaceId=id;state.epoch++;state.refreshEpoch=null;state.page=1;state.health=null;state.config=null;state.observations=[];state.observationsLoaded=false;state.predictions=[];state.jobs=[];
 for(const entry of state.controllers)entry.controller.abort('workspace_changed');state.controllers.clear();try{localStorage.setItem('beeops.workspace.v2',id);}catch{}$('workspace-select').value=id;
 $('workspace-kind').textContent=kindName(current()?.kind);$('overview-source').textContent=`${current()?.name||id} · 데이터 확인 중`;$('overview-count').textContent='';$('stat-model').textContent='—';$('stat-model-note').textContent='현재 벌통의 모델 확인 중';$('stat-weight').innerHTML='— <small>kg</small>';$('stat-weight-note').textContent='관측 확인 중';$('stat-prediction').innerHTML='— <small>kg</small>';$('stat-prediction-note').textContent='예측 확인 중';$('stat-quality').innerHTML='— <small>kg</small>';$('stat-quality-note').textContent='현재 벌통의 오차 기록 확인 중';
 $('observation-chart').innerHTML=empty('관측을 불러오는 중입니다.');$('data-rows').innerHTML='<tr><td colspan="5" class="empty-cell">선택한 벌통의 관측을 불러오는 중입니다.</td></tr>';$('data-table-description').textContent=`${workspaceName(id)}의 관측 기록`;$('alerts').innerHTML=empty('알림 확인 중');$('logs').textContent='현재 벌통의 로그 확인 중';$('training-state').innerHTML=empty('작업 상태 확인 중');$('data-jobs').innerHTML=$('operation-jobs').innerHTML=empty('이 벌통의 작업 이력 확인 중');
 inline('operation-result','');$('export-link').href=scopeUrl('/data/export');$('logs-link').href=scopeUrl('/logs/aiops.log');renderAutomaticSummary();updateControls();refreshWorkspace();if(state.view==='data')loadRows();}
function setView(view){if(!['overview','data','models','operations'].includes(view))view='overview';state.view=view;document.querySelectorAll('.view').forEach(el=>el.hidden=el.id!=='view-'+view);document.querySelectorAll('.nav-button').forEach(button=>{const active=button.dataset.view===view;button.classList.toggle('active',active);if(active)button.setAttribute('aria-current','page');else button.removeAttribute('aria-current');});if(location.hash!=='#'+view)history.replaceState(null,'','#'+view);if(view==='overview'&&state.observationsLoaded){renderObservations();renderAutomaticSummary();}if(view==='overview'&&state.horizonReport)renderHorizonReport();if(view==='operations')renderTemperatureChart();if(view==='models')loadHorizonCatalog();if(view==='data'&&state.workspaceId)loadRows();if(state.workspaceId)refreshWorkspace();window.scrollTo({top:0,behavior:'instant'});}
async function refreshWorkspace(){if(!state.workspaceId||current()?.pending_import)return;const epoch=state.epoch,id=state.workspaceId;if(state.refreshEpoch===epoch)return;state.refreshEpoch=epoch;
 const tasks=[['/health',renderHealth,e=>{$('connection').classList.remove('online');$('connection').lastElementChild.textContent='연결 확인 필요';notice(`${id} 상태 조회 실패 · ${errorText(e)}`,true);}],['/config',data=>{state.config=data;$('workspace-kind').textContent=kindName(data.runtime_type);updateControls();},()=>{}],['/observations?limit=720',data=>{state.observations=Array.isArray(data)?data:[];state.observationsLoaded=true;inline('observation-error','');renderObservations();renderAutomaticSummary();},e=>inline('observation-error',errorText(e))],['/imports/jobs',data=>{state.jobs=Array.isArray(data)?data:[];renderJobs();},e=>{$('data-jobs').innerHTML=$('operation-jobs').innerHTML=empty('작업 이력 조회 실패',errorText(e));}]];
 tasks.push(['/monitoring/temperature',renderTemperature,e=>inline('temperature-error',errorText(e))]);
 if(state.view==='operations')tasks.push(['/observations?limit=720',data=>{state.temperatureObservations=Array.isArray(data)?data:[];renderTemperatureChart();},()=>{}],['/alerts',renderAlerts,e=>{$('alerts').innerHTML=empty('알림 조회 실패',errorText(e));}],['/logs/aiops.log',data=>{const el=$('logs'),nearBottom=el.scrollHeight-el.scrollTop-el.clientHeight<35;el.textContent=typeof data==='string'?data:JSON.stringify(data,null,2);if(nearBottom)el.scrollTop=el.scrollHeight;},e=>{$('logs').textContent=errorText(e);}]);
 await Promise.allSettled(tasks.map(async([path,render,onError])=>{try{const result=await api(path,{workspace:id,cancelOnSwitch:true});if(epoch===state.epoch){render(result);$('last-updated').textContent=`${workspaceName(id)} · ${when(new Date().toISOString())} UTC 조회`;}}catch(e){if(epoch===state.epoch&&!e.cancelled)onError(e);}}));if(epoch===state.epoch)state.refreshEpoch=null;if(state.view==='operations')loadMetrics();}
function renderHealth(h){
 state.health=h;$('connection').classList.add('online');$('connection').lastElementChild.textContent='API 연결됨';
 const source=h.data?.source;$('overview-source').textContent=source?.startsWith('공개 관측: ')?`공개 관측: ${workspaceName()}`:source||'출처 정보 없음';$('overview-count').textContent=`${workspaceName()} · ${fmt(h.data?.rows,0)}개 관측`;
 renderTraining(h);renderAutomaticSummary();updateControls();if(state.horizonCatalogLoaded){loadHorizonReport();loadHistoryReport();}
}
function renderAutomaticSummary(){
 const model=horizonModel(),report=state.horizonReport,forecast=report?.forecast;
 const valid=!!model&&report?.workspace_id===state.workspaceId&&report?.snapshot_id===state.health?.data?.snapshot_id&&String(forecast?.model?.version)===state.horizonVersion&&forecast?.model?.run_id===model.run_id;
 const point=valid&&forecast?.status==='ok'?forecast.trajectory?.[0]:null;
 $('stat-model').textContent=model?'TiRex-2 + LSTM':'준비 필요';$('stat-model').classList.add('label');
 $('stat-model-note').textContent=model?`모든 벌통 공통 · ${horizonWeightText(model.weights)}`:'공통 예측 모델을 확인하고 있습니다.';
 $('stat-prediction').innerHTML=`${fmt(point?.weight_kg)} <small>kg</small>`;
 $('stat-prediction-note').textContent=point?`${when(point.timestamp)} · 기록 시각`:valid&&forecast?.status!=='ok'?'연속 관측과 모델 준비 상태를 확인하세요.':'예측 계산 중';
 const score=model?.validation?.horizons?.['1'],adapted=model?.validation?.temperature_retraining;
 const adaptation=finite(adapted?.new_ensemble_mae_kg),mae=adaptation?adapted.new_ensemble_mae_kg:score?.test_mae_kg;
 $('quality-label').textContent=adaptation?'재학습 검증 MAE · 1시간':'모델 평가 MAE · 1시간';
 $('stat-quality').innerHTML=`${fmt(mae,4)} <small>kg</small>`;
 $('stat-quality-note').textContent=adaptation?`추가 관측의 검증 ${fmt(adapted.validation_samples,0)}쌍 · 실시간 오차 아님`:finite(mae)?`기존 평가 자료 ${fmt(score.origin_count,0)}쌍 · 현재 벌통 실시간 오차 아님`:'모델 평가 기록 확인 중';
}
function historyMatches(){const report=state.historyReport;return report?.workspace_id===state.workspaceId&&report?.snapshot_id===state.health?.data?.snapshot_id&&String(report?.model?.version)===state.horizonVersion&&report?.model?.run_id===horizonModel()?.run_id;}
function renderObservations(){
 const rows=[...state.observations].sort((a,b)=>new Date(a.timestamp)-new Date(b.timestamp)),latest=rows.at(-1),report=historyMatches()?state.historyReport:null;
 const predicted=new Map((report?.predictions||[]).map(row=>[Date.parse(row.target_timestamp),row.predicted_weight_kg]));
 const chart=rows.map(row=>({...row,predicted_weight_kg:predicted.get(Date.parse(row.timestamp))??null}));
 $('stat-weight').innerHTML=`${fmt(latest?.weight_kg)} <small>kg</small>`;$('stat-weight-note').textContent=latest?`${when(latest.timestamp)} UTC · 외기온 ${fmt(latest.temperature_c,1)} °C`:'저장된 관측이 없습니다.';
 $('observations-caption').textContent=rows.length?`관측 ${fmt(rows.length,0)}개 · 예측은 각 시점 직전 관측으로 계산합니다.`:'CSV를 저장하면 실측과 예측을 확인할 수 있습니다.';
 $('history-metrics').textContent=report?.metrics?.count?`표시 구간 MAE ${fmt(report.metrics.mae_kg,4)} kg · ${report.metrics.count}쌍 · 현재 모델의 과거 재계산`:(report?'예측에 필요한 과거 관측이 부족합니다. 연속 169시간 이상부터 실측과 비교할 수 있습니다.':'실측에 맞춰 예측을 불러오는 중입니다.');
 $('observation-range').textContent=rows.length?`${when(rows[0].timestamp,true)} — ${when(latest.timestamp,true)} UTC`:'데이터 화면에서 CSV를 가져오세요.';
 drawChart('observation-chart',chart,'timestamp',[{key:'weight_kg',color:'#3b5960',label:'실측'},{key:'predicted_weight_kg',color:'#d79e2c',label:'예측'}],'아직 저장된 관측이 없습니다.');
 if(state.view==='operations')renderTemperatureChart();
}
function invalidateHistory(){state.historySequence++;state.historyReport=null;state.historyKey=null;state.historyPending=null;}
async function loadHistoryReport(){
 const key=horizonContextKey();if(!key||key===state.historyKey)return state.historyPending;
 const sequence=++state.historySequence,workspace=state.workspaceId,model=horizonModel(),snapshot=state.health.data.snapshot_id;
 state.historyKey=key;state.historyReport=null;renderObservations();
 const pending=(async()=>{try{
  const report=await api('/forecast/history?limit=720',{workspace,cancelOnSwitch:true,timeout:120000});
  if(sequence!==state.historySequence||key!==horizonContextKey())return;
  if(report.workspace_id!==workspace||report.snapshot_id!==snapshot||String(report.model?.version)!==state.horizonVersion||report.model?.run_id!==model.run_id||!Array.isArray(report.predictions))throw new Error('현재 벌통의 예측 기록을 확인할 수 없습니다.');
  const observed=new Map(state.observations.map(row=>[Date.parse(row.timestamp),row]));
  for(const row of report.predictions){const actual=observed.get(Date.parse(row.target_timestamp));if(!actual||!finite(row.predicted_weight_kg)||Date.parse(row.origin_timestamp)>=Date.parse(row.target_timestamp))throw new Error('실측과 예측의 시각이 일치하지 않습니다.');}
  state.historyReport=report;inline('observation-error','');renderObservations();
 }catch(error){if(sequence===state.historySequence&&!error.cancelled){state.historyReport=null;state.historyKey=null;inline('observation-error',errorText(error));$('history-metrics').textContent='예측을 불러오지 못했습니다. 새로고침으로 다시 시도하세요.';}}finally{if(sequence===state.historySequence)state.historyPending=null;}})();state.historyPending=pending;return pending;
}
function drawChart(id,rows,timeKey,series,emptyText,unit='kg'){const valid=rows.filter(r=>series.some(s=>finite(r[s.key]))&&Number.isFinite(new Date(r[timeKey]).getTime()));if(!valid.length){$(id).innerHTML=empty(emptyText);return;}const width=Math.max(280,$(id).clientWidth-28||900),height=Math.max(190,$(id).clientHeight-5||265),p={l:53,r:14,t:18,b:36};const times=valid.map(r=>new Date(r[timeKey]).getTime()),minimumTime=Math.min(...times),maximumTime=Math.max(...times);const values=valid.flatMap(r=>series.flatMap(s=>finite(r[s.key])?[Number(r[s.key])]:[]));let low=Math.min(...values),high=Math.max(...values);const pad=Math.max((high-low)*.15,.05);low-=pad;high+=pad;const x=t=>p.l+(maximumTime===minimumTime ? .5 : (t-minimumTime)/(maximumTime-minimumTime))*(width-p.l-p.r),y=v=>p.t+(high-Number(v))/(high-low)*(height-p.t-p.b);let svg=`<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(id==='observation-chart'?'시간별 실측과 예측 무게':'시간별 관측')} · ${esc(unit)}">`;
 for(let i=0;i<5;i++){const yy=p.t+i*(height-p.t-p.b)/4;svg+=`<line x1="${p.l}" y1="${yy}" x2="${width-p.r}" y2="${yy}" stroke="#e7edde" stroke-dasharray="3 5"/><text x="${p.l-10}" y="${yy+4}" text-anchor="end" fill="#64726a" font-size="12">${fmt(high-i*(high-low)/4,2)}</text>`;}
 const tickCount=Math.min(valid.length,width<500?3:5);for(let i=0;i<tickCount;i++){const j=tickCount===1?0:Math.round(i*(valid.length-1)/(tickCount-1)),d=new Date(times[j]),label=`${d.getUTCMonth()+1}.${d.getUTCDate()} ${String(d.getUTCHours()).padStart(2,'0')}:00`;svg+=`<text x="${x(times[j])}" y="${height-10}" text-anchor="${i===0?'start':i===tickCount-1?'end':'middle'}" fill="#64726a" font-size="12">${label}</text>`;}
 for(const s of series){let path='',previous=null;valid.forEach((r,i)=>{if(!finite(r[s.key])){previous=null;return;}const t=times[i],continuous=previous!==null&&t-previous<=3600000;path+=`${continuous?'L':'M'}${x(t).toFixed(2)},${y(r[s.key]).toFixed(2)} `;previous=t;});svg+=`<path d="${path}" fill="none" stroke="${s.color}" stroke-width="2.3" ${s.dash?`stroke-dasharray="${s.dash}"`:""} stroke-linecap="round" stroke-linejoin="round"/>`;valid.forEach((r,i)=>{if(finite(r[s.key]))svg+=`<circle cx="${x(times[i])}" cy="${y(r[s.key])}" r="${s.dash?0:valid.length<5?3:1.7}" fill="${s.color}"><title>${esc(when(r[timeKey],true))} UTC · ${s.label} ${fmt(r[s.key])} ${esc(unit)}${r.event&&r.event!=='normal'?' · '+esc(r.event):''}</title></circle>`;});}$(id).innerHTML=svg+'</svg>';}
async function loadRows(){if(!state.workspaceId||current()?.pending_import)return;const epoch=state.epoch,seq=++state.rowsSequence,page=state.page;try{const data=await api(`/data/rows?page=${page}&page_size=25`,{workspace:state.workspaceId,cancelOnSwitch:true});if(epoch!==state.epoch||seq!==state.rowsSequence)return;inline('rows-error','');state.totalRows=data.total||0;state.page=data.page||page;$('data-rows').innerHTML=observationRows(data.items||[]);$('data-page-label').textContent=`총 ${fmt(data.total,0)}개 관측 · ${state.page} / ${Math.max(1,Math.ceil((data.total||0)/25))}페이지`;updateControls();}catch(error){if(epoch===state.epoch&&seq===state.rowsSequence&&!error.cancelled)inline('rows-error',errorText(error));}}
const eventNames={normal:'일반 관측',harvest:'채밀 기록',feeding:'급이 기록',inspection:'현장 점검',sensor_fault:'센서 점검',colony_alert:'벌통 점검 신호'};
function observationRows(rows){return rows.length?rows.map(r=>`<tr><td>${esc(when(r.timestamp,true))}</td><td>${esc(r.hive_id)}</td><td>${fmt(r.weight_kg)}</td><td>${fmt(r.temperature_c,1)}</td><td>${badge(eventNames[r.event]||r.event,r.event==='normal'?'':'amber')}</td></tr>`).join(''):'<tr><td colspan="5" class="empty-cell">저장된 관측이 없습니다.</td></tr>';}
function messageList(items,title,error=false){return items?.length?`<div class="message-list${error?' error':''}"><strong>${esc(title)}</strong><ul>${items.map(v=>`<li>${esc(typeof v==='string'?v:JSON.stringify(v))}</li>`).join('')}</ul></div>`:'';}
function sampleDownloadPath(sample){try{const url=new URL(sample?.download_url||'',location.origin);return url.origin===location.origin&&(url.pathname==='/data/sample.csv'||(/^\/data\/(?:samples|trend-cases)\/[^/]+\.csv$/.test(url.pathname)))?url.pathname+url.search:null;}catch{return null;}}
const selectedSample=()=>state.samples.find(sample=>sample.id===state.sampleId);
function sampleSourceUrl(value){try{const url=new URL(value);return ['http:','https:'].includes(url.protocol)?url.href:null;}catch{return null;}}
async function loadSamples(){
 const seq=++state.sampleCatalogSequence;state.busy.add('samplecatalog');updateControls();
 try{const catalog=await api('/data/samples');if(seq!==state.sampleCatalogSequence)return;const samples=(catalog.samples||[]).filter(sample=>sample.id&&sampleDownloadPath(sample));if(!samples.length)throw new Error('사용 가능한 공개 데이터가 목록에 없습니다.');state.samples=samples;const preserved=samples.some(sample=>sample.id===state.sampleId);if(!preserved)state.sampleId=samples.some(sample=>sample.id===catalog.default_sample_id)?catalog.default_sample_id:samples[0].id;inline('sample-catalog-error','');}
 catch(error){if(seq!==state.sampleCatalogSequence)return;inline('sample-catalog-error','공개 데이터 목록을 불러오지 못했습니다. '+errorText(error));}
 finally{if(seq===state.sampleCatalogSequence){state.busy.delete('samplecatalog');renderSampleSelection();updateControls();}}
}
function renderSampleSelection(){
 const select=$('sample-select');select.innerHTML=state.samples.map(sample=>`<option value="${esc(sample.id)}">${esc(sample.label||sample.id)}</option>`).join('')||'<option value="">공개 데이터 확인 필요</option>';select.value=state.sampleId||'';const sample=selectedSample();$('sample-download-link').hidden=!sample;if(!sample){$('sample-description').textContent='공개 데이터 목록을 다시 불러와 주세요.';for(const id of ['sample-facts','sample-usage','sample-provenance'])$(id).textContent='';return;}
 const path=sampleDownloadPath(sample),filename=String(sample.id).replace(/[^a-zA-Z0-9_.-]/g,'_')+'.csv';$('sample-download-link').href=path||'/data/sample.csv';$('sample-download-link').download=filename;$('sample-download-link').textContent='선택한 공개 CSV 다운로드 ↓';$('sample-description').textContent=sample.description||sample.label||sample.id;
 const facts=[`벌통 ${sample.hive_id||'확인 필요'}`];if(finite(sample.rows))facts.push(`${fmt(sample.rows,0)}개 관측`);if(sample.start||sample.end)facts.push(`${when(sample.start,true)} — ${when(sample.end,true)} UTC`);if(finite(sample.weight_min_kg)&&finite(sample.weight_max_kg))facts.push(`무게 ${fmt(sample.weight_min_kg,2)}–${fmt(sample.weight_max_kg,2)} kg`);$('sample-facts').textContent=facts.join(' · ');$('sample-usage').textContent=sample.usage_note||'미리보기에서 벌통·신규·중복 관측을 확인하세요. 최근 연속 168시간 관측으로 공통 모델이 자동 예측합니다.';
 const source=sampleSourceUrl(sample.source_url);$('sample-provenance').innerHTML=`${sample.license?'<span>라이선스 · '+esc(sample.license)+'</span>':''}${source?`<a href="${esc(source)}" target="_blank" rel="noopener">원본 출처 확인 ↗</a>`:''}`;
}
async function previewSelectedSample(){
 if(busy('sample')||busy('preview')||busy('commit'))return;const sample=selectedSample(),path=sampleDownloadPath(sample);if(!sample||!path){inline('file-error','공개 데이터를 먼저 선택하거나 목록을 다시 불러와 주세요.');return;}const selection=++state.fileSelectionRevision;state.busy.add('sample');updateControls();inline('file-error','');
 try{const blob=await api(path,{blob:true});if(selection!==state.fileSelectionRevision||sample.id!==state.sampleId)return;await previewFile(new File([blob],String(sample.id).replace(/[^a-zA-Z0-9_.-]/g,'_')+'.csv',{type:'text/csv'}));}
 catch(error){if(selection===state.fileSelectionRevision)inline('file-error',`${sample.label||sample.id} 가져오기 실패 · ${errorText(error)}`);}
 finally{state.busy.delete('sample');updateControls();}
}

async function previewFile(file,temperaturePractice=false){if(!file||busy('preview')||busy('commit'))return;state.file=file;state.preview=null;if(state.activeJob&&terminal(state.activeJob.status)){state.hiddenCompletedJobId=state.activeJob.job_id;state.importNavigation.delete(state.activeJob.job_id);renderJob();$('overview-import-result').hidden=true;if(state.noticeJobId===state.activeJob.job_id){$('global-notice').hidden=true;state.noticeJobId=null;}}const seq=++state.previewSequence;state.busy.add('preview');$('preview-panel').hidden=true;inline('file-error','');$('selected-file-name').textContent=file.name;$('selected-file-note').textContent=temperaturePractice?'실습 자료와 기존 관측을 확인하고 있습니다.':`${fmt(file.size/1024,1)} KB · 서버에서 파일과 기존 원장을 확인하고 있습니다.`;updateControls();
 try{const form=new FormData();form.append('file',file);const result=temperaturePractice?await api('/data/simulations/temperature/preview',{method:'POST',json:{filename:file.name},timeout:60000}):await api('/imports/preview',{method:'POST',body:form,timeout:60000});if(seq!==state.previewSequence)return;state.preview=result;renderPreview();$('selected-file-note').textContent='파일을 보관했습니다. 아래 미리보기를 확인한 뒤 저장하세요.';}
 catch(error){if(seq===state.previewSequence){inline('file-error',errorText(error)+'\n선택한 파일은 유지됩니다. 파일을 다시 선택하거나 아래 버튼으로 재확인할 수 있습니다.');const retry=document.createElement('button');retry.className='button small secondary';retry.textContent='선택한 파일 다시 확인';retry.addEventListener('click',()=>previewFile(file,temperaturePractice));$('file-error').append(document.createElement('br'),retry);$('selected-file-note').textContent='미리보기를 완료하지 못했습니다. 파일은 아직 저장하지 않았습니다.';}}
 finally{state.busy.delete('preview');updateControls();}}
function renderPreview(){const p=state.preview;if(!p)return;$('preview-panel').hidden=false;$('preview-filename').textContent=p.filename;$('preview-badge').textContent=p.can_commit?'저장 전 확인':'수정 필요';$('preview-badge').className='badge '+(p.can_commit?'green':'red');const exists=state.workspaces.some(w=>w.workspace_id===p.workspace_id);$('preview-destination').innerHTML=`저장 대상 · <strong>${esc(p.workspace_name||p.hive_id||'확인 불가')}</strong> &nbsp; ${esc(p.hive_id||'')}<small>${exists?'기존 벌통의 원장과 모델 이력을 유지합니다.':'새 벌통으로 등록합니다.'}${p.workspace_id!==state.workspaceId?' 현재 화면의 선택과 다른 벌통입니다.':''}</small>`;
 $('preview-counts').innerHTML=[['파일 전체',p.row_count],['신규 관측',p.insert_count],['동일한 중복',p.duplicate_count],['충돌',p.conflict_count]].map(([label,value],i)=>`<div class="mini-stat${i===3&&value?' warn':''}"><span>${label}</span><strong>${fmt(value,0)}</strong></div>`).join('');$('preview-meta').textContent=`기간 ${when(p.start,true)} — ${when(p.end,true)} UTC · 저장 후 공통 TiRex-2 + LSTM으로 자동 예측`;$('preview-warnings').innerHTML=messageList(p.warnings,'확인할 내용');$('preview-errors').innerHTML=messageList((p.errors||[]).map(e=>`${e.row?'CSV '+e.row+'행 · ':''}${e.message}`),'파일을 수정해 주세요',true);$('preview-sample').innerHTML=observationRows(p.sample||[]);$('commit-explanation').textContent=!p.can_commit?'오류 또는 충돌을 해결한 뒤 같은 파일을 다시 확인하세요.':p.temperature_practice?.phase==='heatwave'?'정상 기간 336시간과 고온 기간 168시간을 비교해 앙상블을 재학습합니다. 이미 저장된 관측은 그대로 사용하며, 다시 실행하면 새 재학습을 진행합니다.':p.insert_count===0?`신규 관측은 0개입니다. 동일한 ${p.duplicate_count}개 관측은 다시 저장하지 않고, 가능한 예측 결과를 확인합니다.`:`신규 ${p.insert_count}개를 저장하고 가능한 예측을 생성합니다. 기존 ${p.duplicate_count}개 중복은 유지합니다.`;updateControls();}
async function commitPreview(){
 if(!state.preview?.can_commit||state.preview.committed||busy('commit'))return;
 const p=state.preview,navigationRevision=state.navigationRevision;
 state.busy.add('commit');updateControls();inline('file-error','');
 try{
  const job=await api('/imports/commit',{method:'POST',json:{preview_id:p.preview_id},timeout:30000});
  p.committed=true;state.retrainingPendingImports.add(job.job_id);persistPendingRetraining();
  state.importNavigation.set(job.job_id,{revision:navigationRevision,workspace:job.workspace_id});
  setActiveJob(job);
  if(p.temperature_practice?.phase==='heatwave'){
   beginPracticeRun(p,job);
   if(navigationRevision===state.navigationRevision){selectWorkspace(job.workspace_id);setView('overview');}
  }
  notice(`${p.workspace_name||p.hive_id} · 저장 및 분석 작업을 접수했습니다.`,false,job.job_id);
  loadWorkspaces().catch(()=>{});pollActiveJob();
 }catch(error){inline('file-error',errorText(error)+'\n선택한 파일과 미리보기는 유지했습니다.');}
 finally{state.busy.delete('commit');updateControls();}
}
function setActiveJob(job){state.hiddenCompletedJobId=null;state.activeJob=job;state.jobSequence++;state.lastJobStatus=job.status;try{localStorage.setItem('beeops.import.job',job.job_id);}catch{}renderJob();}
const stageNames={model_versions:'기존 실습 기록 확인 중',data_saved:'관측 저장 완료',model_prepared:'모델 준비 확인 완료',analysis_complete:'과거 예측 분석 완료',finished:'처리 완료',partial:'일부 처리 완료',queued:'처리 대기',saving_data:'관측 저장 중',saveddata:'관측 저장 완료',model_initialization:'모델 준비 확인 중',analyzing:'과거 예측 분석 중',forecasting:'다음 시간 예측 중',completed:'처리 완료',failed:'처리 실패',interrupted:'작업 중단'};
function renderJob(){const j=state.activeJob;if(!j)return;if(terminal(j.status)&&state.hiddenCompletedJobId===j.job_id){$('import-job-panel').hidden=true;return;}$('import-job-panel').hidden=false;$('import-job-identity').textContent=`${j.filename||'CSV 가져오기'} · ${j.workspace_id||'저장 대상 확인 중'} · ${when(j.created_at,true)} UTC`;$('import-job-badge').textContent=statusName(j.status);$('import-job-badge').className='badge '+(['failed','interrupted'].includes(j.status)?'red':j.status==='partial'?'amber':j.status==='completed'?'green':'amber');$('import-job-stage').textContent=stageNames[j.stage]||j.stage||statusName(j.status);const progress=Math.max(0,Math.min(100,Number(j.progress)||0));$('import-job-percent').textContent=progress+'%';$('import-job-progress').style.width=progress+'%';const r=j.result||{},ing=r.ingest,model=r.model_initialization,pair=r.model_versions,analysis=r.analysis,forecast=r.forecast,parts=[];
 const item=(title,value,note,unavailable=false)=>`<div class="result-item${unavailable?' unavailable':''}"><h3>${esc(title)}</h3><strong>${esc(value)}</strong><p>${esc(note||'')}</p></div>`;
 if(ing){
  parts.push(item('관측 저장',`신규 ${ing.inserted??0}개 · 중복 ${ing.duplicate??0}개`,ing.inserted===0?`원장 전체 ${ing.rows??'—'}개 · 같은 관측은 다시 저장하지 않습니다.`:`원장 전체 ${ing.rows??'—'}개 · 저장한 벌통의 관측에 반영했습니다.`));
  parts.push(item('무게 예측','TiRex-2 + LSTM','대시보드에서 무게 예측과 채밀 추천 기간을 확인하세요.'));
 }
 const error=j.error?(typeof j.error==='string'?j.error:JSON.stringify(j.error)):null;$('import-job-result').innerHTML=(parts.length?'<div class="result-grid">'+parts.join('')+'</div>':terminal(j.status)?empty('처리 결과를 확인해 주세요.'):empty('파일을 처리하고 있습니다.','화면을 이동해도 작업은 계속됩니다.'))+messageList(r.warnings,'처리 참고 사항')+(error?messageList([error],'작업 오류',true):'')+`<details class="job-detail-toggle"><summary>작업 ID와 원시 결과 확인</summary><pre>${esc(JSON.stringify(j,null,2))}</pre></details>`;
 $('open-job-workspace').hidden=!j.workspace_id||!terminal(j.status);$('check-job-button').disabled=busy('jobpoll');}
async function pollActiveJob(){
 if(!state.activeJob||busy('jobpoll'))return;
 const id=state.activeJob.job_id,seq=state.jobSequence;state.busy.add('jobpoll');
 try{
  const job=await api('/imports/jobs/'+encodeURIComponent(id));
  if(seq!==state.jobSequence||id!==state.activeJob?.job_id)return;
  state.activeJob=job;renderJob();
  updatePracticeImport(job);renderRetrainingActivity();
  if(terminal(job.status)&&!state.refreshedJobs.has(id)){
   await loadWorkspaces();
   if(seq!==state.jobSequence||id!==state.activeJob?.job_id)return;
   state.refreshedJobs.add(id);loadCatalog();
   const intent=state.importNavigation.get(id),ing=job.result?.ingest;
   const linked=(!!intent||state.retrainingPendingImports.has(id))&&job.result?.retraining_requested===true&&job.result?.retraining_job_id&&!state.retrainingDelivered.has(job.result.retraining_job_id)&&job.result.retraining_job_id;
   if(linked){
    state.retrainingObservedActive.add(linked);
    state.retrainingSeenTerminal.delete(linked);
    persistPendingRetraining();
   }
   state.retrainingPendingImports.delete(id);persistPendingRetraining();
   const canNavigate=intent&&intent.revision===state.navigationRevision&&job.workspace_id&&ing;
   state.importNavigation.delete(id);
   if(canNavigate){
    selectWorkspace(job.workspace_id);setView('overview');$('overview-import-result').hidden=false;
    $('overview-import-title').textContent=ing.inserted===0?`변경 없음 · 신규 0개 · 중복 ${ing.duplicate??0}개`:`${job.workspace_id} · 신규 ${ing.inserted??0}개 반영 · 중복 ${ing.duplicate??0}개`;
    $('overview-import-note').textContent=(linked||state.practiceRun?.import_job_id===id&&job.result?.retraining_job_id)?'온도 변화에 따른 앙상블 재학습을 요청했습니다. 실제 진행 상황과 완료 결과를 위 알림에서 확인하세요.':ing.inserted===0?'이미 저장된 관측입니다. 새 관측이 없으므로 예측값은 동일합니다.':`${job.workspace_id}의 관측을 반영했습니다. 무게 예측과 채밀 추천을 갱신합니다.`;
   }else{refreshWorkspace();if(state.view==='data')loadRows();}
   if(state.hiddenCompletedJobId!==id)notice(`${job.workspace_id} · 가져오기 ${statusName(job.status)}.${ing?` 신규 ${ing.inserted??0}개 · 중복 ${ing.duplicate??0}개${ing.inserted===0?' · 데이터 변경 없음':''}.`:''}${canNavigate?' 해당 벌통의 대시보드로 이동했습니다.':' 처리 결과에서 모델 상태를 확인하세요.'}`,['failed','interrupted'].includes(job.status),id);
   if(linked||state.practiceRun?.import_job_id===id){if(state.retrainingPending)await state.retrainingPending;await loadRetrainingActivity();}
  }
 }catch(error){if(seq===state.jobSequence)inline('file-error','작업 상태 조회 실패 · '+errorText(error)+'\n작업을 새로 만들지 않고 같은 작업 ID로 다시 확인할 수 있습니다.');}
 finally{state.busy.delete('jobpoll');$('check-job-button').disabled=false;}
}
function renderJobs(){const markup=state.jobs.length?state.jobs.map(j=>`<div class="job-item"><div class="job-item-main"><strong>${esc(j.filename||'CSV 가져오기')}</strong><p>${esc(when(j.created_at,true))} UTC · ${esc(j.workspace_id)}${j.result?.ingest?` · 신규 ${j.result.ingest.inserted??0} / 중복 ${j.result.ingest.duplicate??0}`:''}</p></div>${badge(statusName(j.status),j.status==='completed'?'green':['failed','interrupted'].includes(j.status)?'red':'amber')}<button class="button small secondary" data-open-job="${esc(j.job_id)}">결과 보기</button></div>`).join(''):empty('아직 가져오기 기록이 없습니다.','새 CSV를 확인하고 저장하면 작업 이력이 남습니다.');$('data-jobs').innerHTML=$('operation-jobs').innerHTML=markup;}
async function loadCatalog(){
 const sequence=++state.catalogSequence;
 try{const data=await api('/models/catalog');if(sequence!==state.catalogSequence)return;state.catalog=Array.isArray(data)?data:[];state.catalogLoaded=true;updateControls();}catch{state.catalogLoaded=false;}
}
const gateLabels={absolute_mae:'MAE 허용 기준 초과',incumbent_regression:'기존 모델보다 검증 오차 증가',persistence_regression:'직전 값 기준선보다 검증 오차 증가',seasonal_24h_regression:'24시간 전 기준선보다 검증 오차 증가',baseline_improvement_below_minimum:'최강 기준선 대비 최소 5% 개선 미달',load_failed:'모델 번들 로드 검증 실패',alias_update_failed:'배포 버전 전환 실패',champion_changed_during_training:'학습 중 배포 모델 변경'};
function renderTraining(h){
 const data=state.retrainingActivity,j=data?.running_job||data?.queued_jobs?.[0]||data?.latest_job||{},model=data?.active_model||horizonModel();
 const active=['running','queued'].includes(j.status),version=model?.version;
 $('training-badge').textContent=j.status?statusName(j.status):'기록 없음';$('training-badge').className='badge '+(j.status==='failed'?'red':j.status==='completed'?'green':'amber');
 let title=j.status?`앙상블 재학습 ${statusName(j.status)}`:'앙상블 학습 기록이 없습니다.',description='현재 벌통의 관측으로 공통 앙상블을 학습할 수 있습니다.';
 if(active)description='작업은 백그라운드에서 실행되며 현재 앙상블로 예측을 계속 제공합니다.';
 if(j.status==='completed')description=`모델 개정 ${j.new_version||version||'—'}을 모든 벌통에 적용했습니다.`;
 if(j.status==='failed'||j.status==='interrupted')description=j.error||'현재 모델을 유지합니다. 작업 기록을 확인하세요.';
 if(j.status==='skipped')description='현재 모델을 유지합니다. 추가 학습 조건에 해당하지 않습니다.';
 $('training-state').innerHTML=`<div class="training-summary"><h3>${esc(title)}</h3><p>${esc(description)}</p><div class="training-facts"><span>현재 모델<strong>${version?'개정 '+esc(version):'준비 필요'}</strong></span><span>결합 가중치<strong>${esc(horizonWeightText(model?.weights))}</strong></span>${finite(j.training_progress?.completed_epochs)?`<span>LSTM 학습 회차<strong>${fmt(j.training_progress.completed_epochs,0)} / ${fmt(j.training_progress.total_epochs,0)}</strong></span>`:''}</div>${j.completed_at?`<p class="helper">${esc(when(j.completed_at,true))} UTC</p>`:''}</div>`;
 $('scenario-state').textContent=h.demo?.status==='idle'?'기존 LSTM 실습':statusName(h.demo?.status);if(h.demo?.status==='failed')inline('operation-result',h.demo.error||'시나리오 실행에 실패했습니다.',true);
}
function renderAlerts(data){const alerts=Array.isArray(data)?data:[];const labels={model:'모델 품질',colony:'벌통 점검',sensor:'센서 점검',operation:'현장 작업'};$('alert-count').textContent=alerts.length+'개';$('alerts').innerHTML=alerts.length?alerts.slice(0,80).map(a=>`<div class="alert-item"><i class="alert-dot"></i><div class="alert-body"><div><span>${esc(labels[a.category]||a.category)} · ${esc(a.level)}</span><time>${esc(when(a.created_at))}</time></div><p>${esc(a.message)}</p></div></div>`).join(''):empty('등록된 운영 알림이 없습니다.');}
async function loadMetrics(){try{const m=await api('/metrics/summary');$('metric-requests').textContent=fmt(m.requests,0);$('metric-p95').textContent=fmt(m.predict_p95_latency_ms,1)+' ms';$('metric-average').textContent=fmt(m.average_latency_ms,1)+' ms';$('metric-errors').textContent=fmt(Number(m.error_rate)*100,2)+'%';}catch{}}
async function runOperation(scenario=null){
 const workspace=state.workspaceId,epoch=state.epoch,key='operation:'+workspace;if(busy(key))return;
 state.busy.add(key);updateControls();inline('operation-result',scenario?'합성 시나리오 작업을 요청하고 있습니다.':'앙상블 학습 데이터를 확인하고 있습니다.');
 try{
  const response=await api(scenario?'/demo/run':'/models/retrain',{workspace,method:'POST',json:scenario?{scenario}:{}});
  if(!scenario&&response.job?.job_id){
   if(response.status==='already_evaluated')state.retrainingSeenTerminal.add(response.job.job_id);
   else state.retrainingObservedActive.add(response.job.job_id);
  }
  const message=scenario?'실습 작업을 접수했습니다. 실행 기록에서 결과를 확인하세요.':response.status==='already_evaluated'?'현재 데이터의 학습 기록이 있습니다.':'앙상블 재학습을 접수했습니다. 진행 상황에서 결과를 확인하세요.';
  if(epoch===state.epoch){inline('operation-result',message);refreshWorkspace();}else notice(`${workspaceName(workspace)} · ${message}`);
  if(!scenario){if(state.retrainingPending)await state.retrainingPending;await loadRetrainingActivity();}
 }catch(error){if(epoch===state.epoch)inline('operation-result',errorText(error),true);else notice(`${workspaceName(workspace)} · ${errorText(error)}`,true);}
 finally{state.busy.delete(key);updateControls();}
}
// The product uses the current shared ensemble and an automatic seven-day forecast.
const horizonLabels={ok:'예측 완료',inspection_window:'채밀 추천 기간',forecast_unavailable:'예측 결과 확인 필요',insufficient_history:'예측 입력 관측 확인 필요',model_unavailable:'다일 모델 준비 필요',unsupported_horizon:'예측 기간 확인 필요'};
const horizonModel=()=>state.horizonCatalog?.versions?.find(model=>String(model.version)===state.horizonVersion);
function horizonContextKey(){
 const model=horizonModel(),data=state.health?.data;
 if(!state.workspaceId||!data||!model||!state.horizonCatalogLoaded)return null;
 return JSON.stringify([state.workspaceId,state.epoch,data.snapshot_id||[data.rows,data.start,data.end],HORIZON_REGISTRY,state.horizonVersion,model.run_id,state.horizonHours]);
}
function invalidateHorizonReport(message='관측 기준과 다일 예측 모델을 확인하고 있습니다.'){
 state.horizonSequence++;state.horizonReport=null;state.horizonKey=null;state.horizonSettledKey=null;state.horizonPending=null;
 $('horizon-identity').textContent=message;$('horizon-identity').className='report-identity loading';
 $('horizon-chart').innerHTML=empty(message);$('harvest-card').innerHTML=empty('채밀 추천 기간 계산 중','미래 무게 경로를 확인하고 있습니다.');
 $('horizon-time-note').textContent='';$('horizon-interval-note').textContent='';$('horizon-validation').innerHTML='';$('horizon-details-content').innerHTML='';
 inline('horizon-error','');$('refresh-horizon-button').disabled=false;renderAutomaticSummary();
}
function renderHorizonOptions(){
 const model=horizonModel();
 $('horizon-model-note').textContent=model?`최근 ${model.context_hours/24}일 관측 기준`:'공통 TiRex-2 + LSTM 모델을 확인하고 있습니다.';
 renderAutomaticSummary();
}
async function loadHorizonCatalog(){
 const sequence=++state.horizonCatalogSequence;$('refresh-horizon-models-button').disabled=true;
 try{
  const catalog=await api('/models/horizon');if(sequence!==state.horizonCatalogSequence)return;
  if(catalog?.registry_id!==HORIZON_REGISTRY||!Array.isArray(catalog.versions))throw new Error('공통 모델 레지스트리 응답을 확인할 수 없습니다.');
  const chosen=String(catalog.active_version||'2');
  const active=catalog.versions.find(model=>String(model.version)===chosen);
  if(!active)throw new Error('공통 TiRex-2 + LSTM 모델이 준비되지 않았습니다.');
  const previous=JSON.stringify([state.horizonVersion,horizonModel()?.run_id]);
  state.horizonCatalog={...catalog,versions:[active],default_version:chosen};state.horizonCatalogLoaded=true;state.horizonVersion=chosen;state.horizonHours=168;
  if(previous!==JSON.stringify([state.horizonVersion,horizonModel()?.run_id])){invalidateHorizonReport();invalidateHistory();}
  inline('horizon-model-error','');renderHorizonOptions();renderHorizonModels();loadHorizonReport();loadHistoryReport();
 }catch(error){
  if(sequence!==state.horizonCatalogSequence)return;
  state.horizonCatalog=null;state.horizonCatalogLoaded=false;state.horizonVersion='2';state.horizonHours=168;
  invalidateHorizonReport('공통 예측 모델을 불러오지 못했습니다.');renderHorizonOptions();renderHorizonModels();
  inline('horizon-model-error',errorText(error));inline('horizon-error',errorText(error));
 }finally{if(sequence===state.horizonCatalogSequence)$('refresh-horizon-models-button').disabled=false;}
}
function assertHorizonResponse(report,context){
 const forecast=report?.forecast,fail=()=>{throw new Error('응답의 벌통·기간·모델·관측 식별자가 요청과 일치하지 않습니다. 다시 계산해 주세요.');};
 if(!forecast||!report.forecast_id||!report.snapshot_id)fail();
 if(report.selection?.mode!=='automatic'||String(report.selection?.model_version)!==context.version||Number(report.selection?.horizon_hours)!==168)fail();
 for(const item of [report,forecast]){
  if(item.workspace_id!==context.workspace||Number(item.horizon_hours)!==context.hours||Number(item.requested_horizon_hours)!==context.hours)fail();
  if(item.snapshot_id!==report.snapshot_id||item.forecast_id!==report.forecast_id||item.as_of!==report.as_of)fail();
  if(context.snapshot&&item.snapshot_id!==context.snapshot)fail();
  if(item.model?.registry_id!==HORIZON_REGISTRY||String(item.model?.version)!==context.version||item.model?.run_id!==context.run)fail();
 }
 if(!Array.isArray(forecast.trajectory))fail();
 if(forecast.status==='ok'){
  const start=new Date(forecast.as_of).getTime();
  if(!Number.isFinite(start)||forecast.trajectory.length!==context.hours)throw new Error('요청한 기간 전체의 미래 예측이 없습니다.');
  for(const [index,row] of forecast.trajectory.entries()){
   if(new Date(row.timestamp).getTime()!==start+(index+1)*3600000||!finite(row.weight_kg)||row.weight_kg<=0||row.weight_kg>300)throw new Error('미래 예측의 시각 또는 무게 값이 올바르지 않습니다.');
   const lower=row.lower_kg,upper=row.upper_kg;
   if((lower==null)!==(upper==null)||(lower!=null&&(!finite(lower)||!finite(upper)||lower>upper)))throw new Error('예측 범위가 올바르지 않습니다.');
  }
 }else if(forecast.trajectory.length)throw new Error('제공할 수 없는 예측에 경로가 포함되어 있습니다.');
 if(report.status==='inspection_window'){
  const start=new Date(report.candidate_window?.start).getTime(),end=new Date(report.candidate_window?.end).getTime(),origin=new Date(forecast.as_of).getTime();
  if(forecast.status!=='ok'||!Number.isFinite(start)||!Number.isFinite(end)||start<=origin||end<start||end>origin+context.hours*3600000)throw new Error('채밀 추천 기간이 미래 예측의 시각과 일치하지 않습니다.');
 }else if(report.candidate_window!=null)throw new Error('채밀 추천 응답의 날짜 형식을 확인해 주세요.');
}
async function loadHorizonReport(force=false){
 const key=horizonContextKey();if(!key)return;
 if(!force&&key===state.horizonSettledKey)return state.horizonReport;
 if(!force&&key===state.horizonKey&&state.horizonPending)return state.horizonPending;
 invalidateHorizonReport(`${workspaceName()} · 예측을 계산하고 있습니다.`);
 const sequence=state.horizonSequence,context={workspace:state.workspaceId,version:state.horizonVersion,hours:state.horizonHours,run:horizonModel().run_id,snapshot:state.health?.data?.snapshot_id};
 state.horizonKey=key;$('refresh-horizon-button').disabled=true;
 const current=()=>sequence===state.horizonSequence&&key===horizonContextKey();
 const pending=(async()=>{
  try{
   const report=await api('/harvest/recommendation',{workspace:context.workspace,cancelOnSwitch:true,timeout:60000});
   if(!current())return;
   assertHorizonResponse(report,context);state.horizonReport=report;state.horizonSettledKey=key;renderHorizonReport();return report;
  }catch(error){
   if(!current()||error.cancelled)return;
   state.horizonReport=null;state.horizonSettledKey=key;
   $('horizon-identity').textContent=`${context.workspace} · 다일 ${context.version} · ${context.hours}시간 조회 실패`;
   $('horizon-chart').innerHTML=empty('미래 예측을 표시할 수 없습니다.','다시 계산하여 현재 모델과 관측을 확인하세요.');
   $('harvest-card').innerHTML=empty('채밀 추천 조회 오류','다시 계산하여 예측 결과를 확인해 주세요.');
   inline('horizon-error',errorText(error));renderAutomaticSummary();
  }finally{if(current()){state.horizonPending=null;$('refresh-horizon-button').disabled=false;}}
 })();state.horizonPending=pending;return pending;
}
function horizonText(value){
 if(typeof value!=='string')return JSON.stringify(value);
 const hours=value.match(/^requires_(\d+)_continuous_hours$/);
 if(hours)return `최근 연속 ${hours[1]}시간 관측이 필요합니다.`;
 if(value.startsWith('model_dependency_or_artifact_unavailable:'))return '모델 실행에 필요한 파일 또는 라이브러리를 불러오지 못했습니다.';
 return ({context_is_not_one_continuous_hive:'같은 벌통의 연속 관측인지 시간 간격을 확인해 주세요.',requested_model_version_or_artifacts_unavailable:'선택한 모델 버전의 파일을 찾을 수 없습니다.',supported_horizons_are_1_24_72_168:'자동 7일 예측을 제공할 수 없습니다. 모델 구성을 확인해 주세요.'}[value]||value);
}
function horizonMetricRows(validation){
 const records=validation?.per_horizon||validation?.horizons||{};
 return Array.isArray(records)?records.map(item=>[item.horizon_hours,item]):Object.entries(records);
}
function horizonValidationTable(validation){
 const rows=horizonMetricRows(validation);
 if(!rows.length){const evidence=validation?.temperature_retraining;return evidence?`<div class="table-scroll"><table class="horizon-evaluation"><caption>추가 관측 중 학습에서 제외한 ${fmt(evidence.validation_samples,0)}개 시점 · 1시간 예측</caption><thead><tr><th>이전 모델 MAE</th><th>재학습 모델 MAE</th><th>학습 횟수</th></tr></thead><tbody><tr><td>${fmt(evidence.previous_ensemble_mae_kg,4)} kg</td><td>${fmt(evidence.new_ensemble_mae_kg,4)} kg</td><td>${fmt(evidence.epochs_completed,0)}회</td></tr></tbody></table></div>`:'';}
 return `<div class="table-scroll"><table class="horizon-evaluation"><caption>기존 데이터의 시간 순서 분리 평가 · 낮을수록 오차가 작습니다</caption><thead><tr><th>예측 길이</th><th>검증 MAE</th><th>테스트 MAE</th><th>평가 기준시각 수</th><th>독립 구간 수</th></tr></thead><tbody>${rows.map(([hours,score])=>`<tr><td>${esc(hours)}시간</td><td>${fmt(score.validation_mae_kg,4)} kg</td><td>${fmt(score.test_mae_kg,4)} kg</td><td>${fmt(score.origin_count,0)}</td><td>${fmt(score.independent_windows,0)}</td></tr>`).join('')}</tbody></table></div>`;
}
function renderHorizonReport(){
 const report=state.horizonReport;if(!report)return;renderAutomaticSummary();
 const forecast=report.forecast,validation=forecast.validation||{},model=forecast.model;
 $('horizon-identity').className='report-identity';
 $('horizon-identity').textContent=`${workspaceName(report.workspace_id)} · 7일 예측 · 관측 기준 ${when(forecast.as_of,true)}`;
 const historical=forecast.historical||report.historical;
 $('horizon-time-note').textContent=historical?'과거 데이터의 마지막 관측일을 기준으로 계산한 결과입니다.':'마지막 관측 기준 이후의 미래 예측과 추천 기간입니다.';
 const bandCount=forecast.trajectory.filter(row=>finite(row.lower_kg)&&finite(row.upper_kg)).length;
 $('horizon-interval-note').textContent=bandCount?`선은 예측 총무게, 음영은 모델이 계산한 시점별 예측 범위입니다. (${bandCount}/${forecast.trajectory.length}개 시점)`:'선은 관측 기준시각 이후의 예측 총무게입니다.';
 $('horizon-validation').innerHTML=forecast.status==='ok'?`<span>${forecast.trajectory.length}시간의 예측에서 무게가 높은 기간을 추천합니다.</span>`:'';
 drawHorizonChart(forecast,report.candidate_window);
 renderHarvestCard(report);
 $('horizon-details-content').innerHTML=`<dl><dt>예측 ID</dt><dd>${esc(report.forecast_id)}</dd><dt>레지스트리</dt><dd>${esc(model.registry_id)}</dd><dt>모델 실행</dt><dd>${esc(model.run_id)}</dd><dt>관측 기준</dt><dd>${esc(forecast.as_of||'관측 없음')}</dd><dt>기록 시각 기준</dt><dd>${esc(horizonText(forecast.time_basis||report.time_basis||'확인 필요'))}</dd><dt>추천 규칙</dt><dd>${esc(report.rule_version)}</dd></dl>`;
}
function renderHarvestCard(report){
 const candidate=report.status==='inspection_window'&&report.candidate_window,forecast=report.forecast;
 const reasons=[...(report.reasons||[]),...(forecast.status!=='ok'?forecast.reasons||[]:[])];
 const evidenceLabels={recent_gain_kg:'최근 관측 증가',observed_gain_kg:'최근 관측 증가',recent_growth_kg:'최근 관측 증가',forecast_gain_kg:'예상 증가',forecast_decline_kg:'예상 감소',forecast_peak_kg:'하루별 대표 무게의 최대값',peak_weight_kg:'예측 최대 총무게'};
 const numbers=Object.entries(report.evidence||{}).filter(([key,value])=>Object.hasOwn(evidenceLabels,key)&&finite(value));
 $('harvest-card').innerHTML=`<div class="harvest-heading"><span class="eyebrow">HARVEST WINDOW</span>${candidate?badge('무게 예측 기반','green'):''}</div><h3>${esc(horizonLabels[report.status]||'예측 결과 확인 중')}</h3>${candidate?`<p class="harvest-dates">${esc(when(candidate.start,true))}<br><span>— ${esc(when(candidate.end,true))}</span></p><p class="harvest-context">관측 기준 이후 ${fmt((Date.parse(candidate.start)-Date.parse(forecast.as_of))/86400000,1)}–${fmt((Date.parse(candidate.end)-Date.parse(forecast.as_of))/86400000,1)}일 · 기록 시각</p>`:'<p class="harvest-context">예측 결과를 다시 조회해 주세요.</p>'}${reasons.length?`<ul class="harvest-reasons">${[...new Set(reasons.map(horizonText))].map(reason=>`<li>${esc(reason)}</li>`).join('')}</ul>`:''}${numbers.length?`<dl class="harvest-evidence">${numbers.map(([key,value])=>`<dt>${esc(evidenceLabels[key])}</dt><dd>${fmt(value)} kg</dd>`).join('')}</dl>`:''}${renderHarvestPlanning(report.planning,candidate)}`;
}
function safeHarvestLink(value){
 try{const url=new URL(String(value));return ['https:','http:'].includes(url.protocol)?url.href:null;}catch{return null;}
}
function harvestValue(value){if(value==null)return '미입력';if(typeof value==='boolean')return value?'확인함':'추가 확인 필요';return typeof value==='object'?JSON.stringify(value):String(value);}
function harvestReference(reference){
 if(!reference||typeof reference!=='object')return reference?String(reference):'';
 if(reference.value!=null)return `참고 목표 ${reference.value}${reference.unit||''} ${reference.operator==='>='?'이상':reference.operator==='<='?'이하':reference.operator||''}`;
 return '';
}
function renderHarvestPlanning(planning,candidate){
 const checks=Array.isArray(planning?.field_checks)?planning.field_checks:[];
 const sources=(Array.isArray(planning?.sources)?planning.sources:[]).map(source=>{const url=safeHarvestLink(source.url);return url?`<li><a href="${esc(url)}" target="_blank" rel="noopener noreferrer">${esc(source.title||source.id||'공식 자료')} ↗</a></li>`:'';}).filter(Boolean);
 return `${candidate?'<p class="harvest-checks">채밀 전 꿀의 수분·봉개 상태와 군체의 먹이 비축량을 함께 확인하세요.</p>':''}${checks.length||sources.length?`<details class="harvest-source-details"><summary>현장 확인 기준과 근거</summary>${checks.length?`<dl>${checks.map(check=>`<dt>${esc(check.label||check.key)}</dt><dd>${esc(harvestReference(check.reference)||check.note||'현장에서 확인')}</dd>`).join('')}</dl>`:''}${sources.length?`<ul>${sources.join('')}</ul>`:''}</details>`:''}`;
}
function drawHorizonChart(forecast,candidate){
 const rows=forecast.trajectory||[],element=$('horizon-chart');
 if(forecast.status!=='ok'||!rows.length){element.innerHTML=empty(horizonLabels[forecast.status]||'미래 예측 없음',(forecast.reasons||[]).map(horizonText).join(' · '));return;}
 const width=Math.max(300,element.clientWidth-24||800),height=Math.max(225,element.clientHeight-8||285),p={l:59,r:18,t:22,b:42};
 const origin=Date.parse(forecast.as_of),end=origin+forecast.horizon_hours*3600000;
 const values=rows.flatMap(row=>[row.weight_kg,...(finite(row.lower_kg)&&finite(row.upper_kg)?[row.lower_kg,row.upper_kg]:[])]).map(Number);
 let low=Math.min(...values),high=Math.max(...values);const pad=Math.max((high-low)*.14,.04);low-=pad;high+=pad;
 const x=time=>p.l+(time-origin)/(end-origin)*(width-p.l-p.r),y=value=>p.t+(high-value)/(high-low)*(height-p.t-p.b);
 let svg=`<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="관측 기준 이후 ${forecast.horizon_hours}시간의 예상 벌통 총무게"><title>미래 무게 예측 · 관측 기준 ${esc(forecast.as_of)}</title>`;
 if(candidate)svg+=`<rect class="harvest-window" x="${x(Date.parse(candidate.start)).toFixed(2)}" y="${p.t}" width="${Math.max(1,x(Date.parse(candidate.end))-x(Date.parse(candidate.start))).toFixed(2)}" height="${height-p.t-p.b}" fill="#e9ce7770"><title>채밀 추천 기간</title></rect>`;
 for(let i=0;i<4;i++){const value=low+(high-low)*i/3,py=y(value);svg+=`<line x1="${p.l}" x2="${width-p.r}" y1="${py}" y2="${py}" stroke="#e0e7dc"/><text x="${p.l-9}" y="${py+4}" text-anchor="end" fill="#5f7368" font-size="11">${fmt(value,2)}</text>`;}
 svg+=`<text x="${p.l-9}" y="13" text-anchor="end" fill="#5f7368" font-size="11">kg</text>`;
 for(let i=0;i<4;i++){const hours=forecast.horizon_hours*i/3,label=i===0?'관측 기준':hours%24===0?`+${hours/24}일`:`+${fmt(hours,0)}시간`;svg+=`<text x="${x(origin+hours*3600000)}" y="${height-12}" text-anchor="${i===0?'start':i===3?'end':'middle'}" fill="#5f7368" font-size="11">${label}</text>`;}
 // Split bands at missing bounds; never bridge an unprovided uncertainty interval.
 const groups=[];let group=[];
 for(const row of rows){if(finite(row.lower_kg)&&finite(row.upper_kg))group.push(row);else if(group.length){groups.push(group);group=[];}}
 if(group.length)groups.push(group);
 for(const points of groups){if(points.length<2)continue;const upper=points.map(row=>`${x(Date.parse(row.timestamp)).toFixed(2)},${y(row.upper_kg).toFixed(2)}`),lower=[...points].reverse().map(row=>`${x(Date.parse(row.timestamp)).toFixed(2)},${y(row.lower_kg).toFixed(2)}`);svg+=`<polygon class="horizon-band" points="${[...upper,...lower].join(' ')}" fill="#749b8b33"/>`;}
 const points=rows.map(row=>`${x(Date.parse(row.timestamp)).toFixed(2)},${y(row.weight_kg).toFixed(2)}`).join(' ');
 svg+=`<polyline points="${points}" fill="none" stroke="#327561" stroke-width="2.5" stroke-linejoin="round"/><circle cx="${x(end)}" cy="${y(rows.at(-1).weight_kg)}" r="4" fill="#327561"/></svg>`;
 element.innerHTML=svg;
}
function horizonComponentLabel(value){
 const name=typeof value==='string'?value:value?.name||value?.model||value?.id||'구성 모델';
 return ({lstm:'LSTM',lightgbm:'LightGBM',chronos2:'Chronos-2','chronos-2':'Chronos-2',tirex2:'TiRex-2','tirex-2':'TiRex-2'}[name]||name);
}
function horizonWeightText(weights){
 if(!weights||typeof weights!=='object')return '가중치 정보 없음';
 return Object.entries(weights).map(([name,value])=>finite(value)?`${horizonComponentLabel(name)} ${fmt(Number(value)*100,1)}%`:`${name}: ${horizonWeightText(value)}`).join(' · ');
}
function horizonProtocolLabel(value){return value==='retrospective_regression_existing_test_already_inspected'?'과거 자료의 시간 순서 분리 평가 · 이미 검토한 테스트 자료에 대한 회귀 비교':value==='retrospective_transfer'?'사후 전이 실험 · 서로 다른 연도의 벌통을 이용한 평가이며 당시 실시간 운영 검증은 아닙니다.':value||'평가 방식 확인 필요';}
function renderHorizonModels(){
 const versions=(state.horizonCatalog?.versions||[]).filter(model=>String(model.version)===state.horizonVersion);
 $('horizon-model-catalog').innerHTML=versions.length?versions.map(model=>`<article class="horizon-model-card"><div class="model-card-top"><div><p class="model-identity">${esc(HORIZON_REGISTRY)}</p><h3>TiRex-2 + LSTM</h3></div>${badge('사용 중인 공통 모델','green')}</div><p class="horizon-components">${(model.components||[]).map(component=>esc(horizonComponentLabel(component))).join(' + ')||'구성 정보 없음'}</p><p class="horizon-weights">${esc(horizonWeightText(model.weights))}</p><p class="helper">입력 ${fmt(model.context_hours,0)}시간 · 최대 ${fmt(model.max_horizon_hours,0)}시간 예측</p>${horizonValidationTable(model.validation)}<details class="model-details"><summary>모델 실행과 평가 정보</summary><dl><dt>Run ID</dt><dd>${esc(model.run_id)}</dd><dt>평가 방식</dt><dd>${esc(horizonProtocolLabel(model.evaluation_protocol))}</dd></dl></details></article>`).join(''):empty('등록된 다일 앙상블이 없습니다.','모델 파일과 학습 결과가 준비되면 실제 구성과 평가 점수를 표시합니다.');
}
const temperatureLabels={uninitialized:'기준 준비',reference_updated:'새 기준 적용',reference_initialized:'기준 저장',waiting:'관측 수집 중',stable:'안정',drift:'변화 감지',unavailable:'준비 필요'};
const temperatureStepNames={collect:'데이터 수집',monitor:'온도 비교',detect:'변화 감지',train:'모델 학습',register:'모델 등록',deploy:'모델 적용',collection:'데이터 수집',monitoring:'온도 비교',detection:'변화 감지',trigger:'재학습 요청',training:'모델 학습',registration:'모델 등록',deployment:'모델 적용'};
function temperatureStepDetail(step){
 const d=step.detail;if(typeof d==='string')return d;if(!d)return '';
 if(step.key==='train')return `${d.training_windows??'—'}개 입력 구간 · ${d.epochs_completed??'—'}회 학습`;
 if(step.key==='register')return `MLflow 등록 ${d.mlflow_model_version??'—'} · 모델 개정 ${d.version??'—'}`;
 if(step.key==='deploy')return `모든 벌통에 모델 개정 ${d.version??'—'} 적용`;
 return '';
}
function renderTemperatureResult(job){
 const e=job.evaluation;if(!e)return '';
 const blended=job.training?.trained_components?.includes('ensemble_weights'),weights=job.training?.new_weights;
 const trained=blended?'TiRex-2는 기존 모델을 사용하고, LSTM과 결합 가중치를 학습했습니다.':'TiRex-2와 결합 비중은 유지하고 LSTM을 추가 학습했습니다.';
 return `<div class="temperature-result-summary"><p>${esc(when(e.validation_start,true))} — ${esc(when(e.validation_end,true))} · 학습에서 제외한 ${fmt(e.validation_samples,0)}개 시점</p><dl><dt>이전 모델 MAE</dt><dd>${fmt(e.previous_ensemble_mae_kg,4)} kg</dd><dt>재학습 모델 MAE</dt><dd>${fmt(e.new_ensemble_mae_kg,4)} kg</dd></dl><p>${esc(trained)}${blended&&weights?` ${esc(horizonWeightText(weights))}`:''}</p><details><summary>등록·학습 기록</summary><pre class="temperature-result">${esc(JSON.stringify({job_id:job.job_id,parent_version:job.parent_version,new_version:job.new_version,run_id:job.run_id,mlflow_model_version:job.mlflow_model_version,training:job.training},null,2))}</pre></details></div>`;
}
// A practice run belongs to an accepted import, never to whichever history item
// happens to be latest. Keep its verified result until the user acknowledges it.
function restorePracticeRun(){
 try{
  const run=JSON.parse(localStorage.getItem('beeops.practice.run.v1')||'null');
  if(!run||typeof run.import_job_id!=='string'||!run.import_job_id||run.workspace_id!=='TEMP-DRIFT-PRACTICE')return null;
  if(run.retraining_job_id!==null&&typeof run.retraining_job_id!=='string')return null;
  if(run.last_job&&(run.last_job.job_id!==run.retraining_job_id||run.last_job.workspace_id!==run.workspace_id))run.last_job=null;
  return run;
 }catch{return null;}
}
function persistPracticeRun(){
 try{if(state.practiceRun)localStorage.setItem('beeops.practice.run.v1',JSON.stringify(state.practiceRun));else localStorage.removeItem('beeops.practice.run.v1');}catch{}
}
function practiceWorkspaces(workspaces){
 const run=state.practiceRun;
 if(!run||workspaces.some(w=>w.workspace_id===run.workspace_id))return workspaces;
 return [...workspaces,{workspace_id:run.workspace_id,name:run.workspace_id,hive_id:run.workspace_id,kind:'synthetic',pending_import:true}];
}
function beginPracticeRun(preview,job){
 if(job.workspace_id!=='TEMP-DRIFT-PRACTICE')return;
 if(state.practiceRun?.import_job_id!==job.job_id){
  state.practiceRun={preview_id:preview.preview_id,import_job_id:job.job_id,workspace_id:job.workspace_id,workspace_name:preview.workspace_name||job.workspace_id,retraining_job_id:null,last_job:null,import_status:job.status,error:null};
 }
 persistPracticeRun();
 state.workspaces=practiceWorkspaces(state.workspaces);
 $('workspace-select').innerHTML=state.workspaces.map(w=>`<option value="${esc(w.workspace_id)}">${esc(w.name||w.hive_id||w.workspace_id)}</option>`).join('');
 if(state.workspaceId)$('workspace-select').value=state.workspaceId;
 updatePracticeImport(job);renderRetrainingActivity();
}
function updatePracticeImport(job){
 const run=state.practiceRun;
 if(!run||job?.job_id!==run.import_job_id||job.workspace_id!==run.workspace_id)return;
 run.import_status=job.status;run.import_stage=job.stage;run.lookup_error=null;
 const trainingId=job.result?.retraining_job_id;
 if(typeof trainingId==='string'&&trainingId){run.retraining_job_id=trainingId;run.error=null;}
 else if(terminal(job.status)){
  run.error=job.error?(typeof job.error==='string'?job.error:JSON.stringify(job.error)):'관측 처리는 끝났지만 재학습 작업이 생성되지 않았습니다. 데이터 화면의 처리 결과를 확인해 주세요.';
 }
 persistPracticeRun();
}
function practiceFinished(job){return ['completed','failed','interrupted','skipped','partial'].includes(job?.status);}
function rememberPracticeJob(job){
 const run=state.practiceRun;
 if(!run||!job||job.job_id!==run.retraining_job_id||job.workspace_id!==run.workspace_id||job.source==='legacy_lstm')return false;
 run.last_job=job;run.lookup_error=null;
 if(practiceFinished(job)){
  state.retrainingDelivered.add(job.job_id);state.retrainingObservedActive.delete(job.job_id);
  state.retrainingTerminalQueue=state.retrainingTerminalQueue.filter(item=>item.job_id!==job.job_id);
  if(state.retrainingNoticeJob?.job_id===job.job_id){clearTimeout(state.retrainingHideTimer);state.retrainingNoticeJob=null;}
  persistPendingRetraining();
 }
 persistPracticeRun();return true;
}
function practiceNotificationJob(data){
 const run=state.practiceRun;if(!run)return null;
 const job=[data?.running_job,...(data?.queued_jobs||[]),...(data?.jobs||[]),data?.latest_job].find(item=>item?.job_id===run.retraining_job_id&&item.workspace_id===run.workspace_id);
 if(job)rememberPracticeJob(job);
 if(run.last_job)return run.last_job;
 return {job_id:'practice-import:'+run.import_job_id,workspace_id:run.workspace_id,status:run.error?'failed':'preparing',stage:run.import_stage,error:run.error,preparing_import:true};
}
async function refreshPracticeRun(data){
 const run=state.practiceRun;if(!run)return;
 const isCurrent=()=>state.practiceRun===run;
 try{
  if(!run.retraining_job_id&&!terminal(run.import_status)){
   const imported=await api('/imports/jobs/'+encodeURIComponent(run.import_job_id),{timeout:10000});
   if(!isCurrent())return;
   if(imported?.job_id!==run.import_job_id||imported.workspace_id!==run.workspace_id)throw new Error('고온 실습의 저장 작업을 확인할 수 없습니다.');
   updatePracticeImport(imported);
  }
  if(!isCurrent()||!run.retraining_job_id)return;
  const found=[data?.running_job,...(data?.queued_jobs||[]),...(data?.jobs||[]),data?.latest_job].find(job=>job?.job_id===run.retraining_job_id&&job.workspace_id===run.workspace_id);
  if(found){rememberPracticeJob(found);return;}
  // A completed result can age out of the recent-history list. Its exact ID is
  // still queryable, and a verified terminal response is also saved for reload.
  if(practiceFinished(run.last_job))return;
  const job=await api('/monitoring/retraining/jobs/'+encodeURIComponent(run.retraining_job_id),{timeout:10000});
  if(!isCurrent())return;
  if(!rememberPracticeJob(job))throw new Error('요청한 앙상블 재학습 작업을 확인할 수 없습니다.');
 }catch(error){if(isCurrent()){run.lookup_error=errorText(error);persistPracticeRun();}}
}
function acknowledgePracticeRun(){
 const run=state.practiceRun;if(!run)return;
 const job=practiceNotificationJob(state.retrainingActivity);
 if(!practiceFinished(job))return;
 if(run.retraining_job_id){state.retrainingDelivered.add(run.retraining_job_id);state.retrainingObservedActive.delete(run.retraining_job_id);state.retrainingTerminalQueue=state.retrainingTerminalQueue.filter(item=>item.job_id!==run.retraining_job_id);}
 if(state.retrainingNoticeJob?.job_id===run.retraining_job_id){clearTimeout(state.retrainingHideTimer);state.retrainingNoticeJob=null;}
 state.retrainingPendingImports.delete(run.import_job_id);state.practiceRun=null;
 persistPracticeRun();persistPendingRetraining();renderRetrainingActivity();
 if(state.workspaces.some(w=>w.workspace_id===run.workspace_id&&w.pending_import)){
  loadWorkspaces().then(()=>refreshWorkspace()).catch(error=>notice('벌통 목록 확인 실패 · '+errorText(error),true));
 }
}
function restoreRetrainingIds(kind){
 try{const ids=JSON.parse(localStorage.getItem('beeops.retraining.'+kind+'.v1')||'[]');return new Set(Array.isArray(ids)?ids.filter(id=>typeof id==='string'&&id.length<150).slice(-100):[]);}catch{return new Set();}
}
function persistPendingRetraining(){
 try{for(const [kind,ids] of [['pending',state.retrainingObservedActive],['imports',state.retrainingPendingImports],['delivered',state.retrainingDelivered]])localStorage.setItem('beeops.retraining.'+kind+'.v1',JSON.stringify([...ids].slice(-100)));}catch{}
}
function startRetrainingNotice(job){
 state.retrainingNoticeJob=job;state.retrainingNoticeUntil=Date.now()+15000;
 state.retrainingObservedActive.delete(job.job_id);state.retrainingDelivered.add(job.job_id);persistPendingRetraining();
 clearTimeout(state.retrainingHideTimer);
 state.retrainingHideTimer=setTimeout(()=>{
  if(document.hidden){state.retrainingNoticeUntil=0;return;}
  renderRetrainingActivity();
 },15000);
}
function retrainingNotificationJob(data){
 if(!data)return null;
 const first=state.retrainingBaselineTime===null;
 if(first)state.retrainingBaselineTime=Date.parse(data.server_time)||Date.now();
 const jobs=[...new Map([data.running_job,...(data.queued_jobs||[]),data.latest_job,...(data.jobs||[])].filter(job=>job?.job_id&&job.source!=='legacy_lstm').map(job=>[job.job_id,job])).values()];
 const active=jobs.find(job=>job.status==='running')||jobs.find(job=>job.status==='queued');
 jobs.filter(job=>['running','queued'].includes(job.status)).forEach(job=>state.retrainingObservedActive.add(job.job_id));
 const finished=jobs.filter(job=>['completed','failed','interrupted','skipped','partial'].includes(job.status));
 const fresh=finished.filter(job=>!state.retrainingSeenTerminal.has(job.job_id)&&!state.retrainingDelivered.has(job.job_id)&&(state.retrainingObservedActive.has(job.job_id)||!first&&(Date.parse(job.requested_at)||0)>state.retrainingBaselineTime)).sort((a,b)=>(Date.parse(a.completed_at)||0)-(Date.parse(b.completed_at)||0));
 for(const job of fresh){
  if(state.retrainingNoticeJob?.job_id!==job.job_id&&!state.retrainingTerminalQueue.some(item=>item.job_id===job.job_id))state.retrainingTerminalQueue.push(job);
  state.retrainingObservedActive.add(job.job_id);
 }
 finished.forEach(job=>state.retrainingSeenTerminal.add(job.job_id));persistPendingRetraining();
 if(active){clearTimeout(state.retrainingHideTimer);state.retrainingNoticeJob=null;return active;}
 // A background tab must not consume a completion before the user can see it.
 if(document.hidden)return state.retrainingNoticeJob;
 if(state.retrainingNoticeJob&&(state.retrainingNoticeUntil===0||Date.now()<state.retrainingNoticeUntil)){
  if(state.retrainingNoticeUntil===0)startRetrainingNotice(state.retrainingNoticeJob);
  return jobs.find(job=>job.job_id===state.retrainingNoticeJob.job_id)||state.retrainingNoticeJob;
 }
 state.retrainingNoticeJob=null;
 const next=state.retrainingTerminalQueue.shift();
 if(next){startRetrainingNotice(next);return next;}
 return null;
}
function renderRetrainingActivity(data=state.retrainingActivity){
 if(data!==undefined)state.retrainingActivity=data;
 const panel=$('retraining-panel'),pinned=practiceNotificationJob(data),transient=retrainingNotificationJob(data),job=pinned||transient;
 if($('retraining-acknowledge'))$('retraining-acknowledge').hidden=!pinned||!practiceFinished(pinned);
 renderTraining(state.health||{});updateControls();
 panel.hidden=!job;
 if(panel.hidden)return;
 const status=job?.status||'unavailable',active=['running','queued'].includes(status),lookupError=pinned?state.practiceRun?.lookup_error:'',stale=!!(state.retrainingError||lookupError)&&!practiceFinished(job);
 panel.dataset.status=stale?'stale':status;
 if(job&&job.job_id!==state.retrainingJobId){state.retrainingJobId=job.job_id;state.retrainingOpen=active||!!pinned&&status==='completed';}
 const phase=status==='running'&&job.stage==='train'?({lstm_training:'LSTM 학습 중',ensemble_calibration:'앙상블 결합 비중 학습 중',held_out_evaluation:'예측 오차 확인 중'}[job.training_progress?.phase]||''):'';
 const titles={preparing:'앙상블 재학습 준비 중',running:'앙상블 재학습 중',queued:'앙상블 재학습 대기 중',completed:'앙상블 재학습 완료',failed:'앙상블 재학습 실패',interrupted:'앙상블 재학습 중단',skipped:'앙상블 재학습 건너뜀'};
 $('retraining-title').textContent=job.preparing_import&&status==='failed'?'재학습 요청 실패':titles[status]||'앙상블 재학습 상태';
 $('retraining-badge').textContent=stale?'상태 확인 필요':status==='preparing'?'관측 저장 · 요청 처리 중':status==='running'?(phase||temperatureStepNames[job.stage]||'진행 중'):statusName(status);
 const check=job?.check,shift=finite(check?.mean_shift_c)?` · 온도 ${fmt(check.reference_mean_c,1)} → ${fmt(check.current_mean_c,1)} °C`:'';
 const queued=(data?.queued_jobs||[]).filter(item=>item.job_id!==job?.job_id).length;
 $('retraining-context').textContent=job?`${workspaceName(job.workspace_id)} 관측${shift} · 모든 벌통 공통 앙상블${queued?` · 다음 작업 ${queued}건 대기`:''}`:'서버 연결을 확인하고 있습니다.';
 const progress=job?.training_progress,hasProgress=finite(progress?.completed_epochs)&&finite(progress?.total_epochs)&&progress.total_epochs>0;
 $('retraining-progress').hidden=!hasProgress;
 if(hasProgress){$('retraining-progress').max=Number(progress.total_epochs);$('retraining-progress').value=Math.min(Number(progress.total_epochs),Math.max(0,Number(progress.completed_epochs)));}
 $('retraining-progress-label').textContent=hasProgress?`${phase?phase+' · ':''}LSTM 학습 ${progress.completed_epochs} / ${progress.total_epochs}회 완료${finite(progress.training_windows)?` · ${fmt(progress.training_windows,0)}개 입력 구간`:''}`:status==='preparing'?'고온 관측을 확인하고 재학습을 준비하고 있습니다. 학습이 시작되면 진행 상황을 표시합니다.':status==='queued'?'앞선 작업이 끝나면 학습을 시작합니다.':status==='running'?'학습 데이터를 준비하고 있습니다.':job?.training?.epochs_completed?`학습 ${job.training.epochs_completed}회 완료`:'';
 $('retraining-updated').textContent=data?.server_time?`${when(data.server_time)} UTC 확인`:'';
 const warnings=job?.reporting_warnings?.length?'모델 적용은 완료했지만 일부 작업 기록을 저장하지 못했습니다. '+job.reporting_warnings.join(' · '):'';
 $('retraining-error').textContent=stale?'상태 조회가 지연되고 있습니다. 아래는 마지막으로 확인한 기록입니다. '+(lookupError||state.retrainingError):job?.error||warnings;
 $('retraining-error').hidden=!$('retraining-error').textContent;
 const steps=job?.steps||[];
 $('retraining-steps').innerHTML=steps.map((step,index)=>`<li class="${esc(step.status)}" data-step="${esc(step.key)}"><span class="retraining-step-index" aria-hidden="true">${step.status==='completed'?'✓':index+1}</span><strong>${esc(temperatureStepNames[step.key]||step.key)}</strong><span>${esc(statusName(step.status))}</span></li>`).join('');
 const version=data?.active_model?.version;
 let result=status==='preparing'?'화면을 이동하거나 새로고침해도 이 작업의 진행 상황과 결과를 확인할 수 있습니다.':active?`학습 중에도 현재 모델${version?` 개정 ${version}`:''}로 예측할 수 있습니다.`:status==='completed'?`모델 개정 ${job.new_version||version||'—'} 적용 완료 · ${when(job.completed_at,true)} UTC`:status==='failed'||status==='interrupted'?`현재 모델을 유지합니다${version?` (개정 ${version})`:''}. 예측을 계속 사용할 수 있습니다.`:status==='skipped'?'앞선 학습에서 새 온도 기준이 반영되어 현재 모델을 유지합니다.':'';
 if(status==='completed'&&job.evaluation&&finite(job.evaluation.new_ensemble_mae_kg))result+=` · 검증 MAE ${fmt(job.evaluation.previous_ensemble_mae_kg,4)} → ${fmt(job.evaluation.new_ensemble_mae_kg,4)} kg (${fmt(job.evaluation.validation_samples,0)}개 시점)`;
 if(pinned&&status==='completed')result+=' · 확인을 누르면 이 알림을 닫습니다.';
 $('retraining-result').textContent=result;
 $('retraining-details').hidden=!state.retrainingOpen;
 $('retraining-toggle').hidden=!job||!steps.length;
 $('retraining-toggle').textContent=state.retrainingOpen?'진행 상황 접기':'진행 상황 보기';
 $('retraining-toggle').setAttribute('aria-expanded',String(state.retrainingOpen));
}
function toggleRetrainingDetails(){state.retrainingOpen=!state.retrainingOpen;renderRetrainingActivity();}
function loadRetrainingActivity(){
 if(state.retrainingPending)return state.retrainingPending;
 const sequence=++state.retrainingSequence;
 const pending=(async()=>{
  try{
   const data=await api('/monitoring/retraining',{timeout:10000});
   if(sequence!==state.retrainingSequence)return;
   if(typeof data?.enabled!=='boolean'||!Array.isArray(data.jobs)||!Array.isArray(data.queued_jobs))throw new Error('재학습 상태 응답을 확인할 수 없습니다.');
   state.retrainingError='';await refreshPracticeRun(data);
   if(sequence!==state.retrainingSequence)return;
   renderRetrainingActivity(data);
   const active=data.active_model,desired=active?.version?JSON.stringify([String(active.version),active.run_id]):null;
   const currentKey=JSON.stringify([state.horizonVersion,horizonModel()?.run_id]);
   if(desired&&desired!==currentKey&&(!state.horizonCatalogLoaded||desired!==state.retrainingModelKey)){
    state.retrainingModelKey=desired;await loadHorizonCatalog();
    if(!state.horizonCatalogLoaded)state.retrainingModelKey=null;
   }
  }catch(error){if(sequence===state.retrainingSequence){state.retrainingError=errorText(error);renderRetrainingActivity();}}
  finally{if(state.retrainingPending===pending)state.retrainingPending=null;}
 })();
 state.retrainingPending=pending;return pending;
}
function resetTemperature(){
 for(const id of ['temperature-reference','temperature-current','temperature-shift'])$(id).textContent='—';
 $('temperature-state').textContent='확인 중';$('temperature-context').textContent='';$('temperature-job').innerHTML='';$('temperature-jobs').innerHTML='';$('temperature-chart').innerHTML='';$('temperature-summary').textContent='온도 변화를 확인하고 있습니다.';inline('temperature-error','');
}
function renderTemperature(data){
 if(data?.workspace_id!==state.workspaceId)return;state.temperature=data;inline('temperature-error','');
 const check=data.check||{},reference=data.reference||{},job=data.job;
 $('temperature-reference').textContent=`${fmt(check.reference_mean_c??reference.mean_c,1)} °C`;
 $('temperature-current').textContent=`${fmt(check.current_mean_c,1)} °C`;
 $('temperature-shift').textContent=`${finite(check.mean_shift_c)&&check.mean_shift_c>0?'+':''}${fmt(check.mean_shift_c,1)} °C`;
 $('temperature-state').textContent=temperatureLabels[check.status]||check.status||'기준 준비';
 $('temperature-context').textContent=reference.count?`기준 ${reference.count}시간 · ${when(reference.start,true)} — ${when(reference.end,true)} · 최근 24시간과 비교`:'CSV를 저장하거나 온도 확인을 누르면 최근 관측으로 기준을 준비합니다.';
 $('temperature-summary').textContent=`온도 ${temperatureLabels[check.status]||'확인 중'}${finite(check.mean_shift_c)?' · 기준 대비 '+fmt(check.mean_shift_c,1)+' °C':''}${job?' · 재학습 '+statusName(job.status):''}`;
 const steps=job?.steps||[];
 $('temperature-job').innerHTML=job?`<div class="temperature-job-heading"><h3>재학습 ${esc(statusName(job.status))}</h3>${badge(statusName(job.status),job.status==='completed'?'green':job.status==='failed'?'red':'amber')}</div>${finite(job.check?.mean_shift_c)?`<p class="temperature-context">감지 당시 ${fmt(job.check.reference_mean_c,1)} °C → ${fmt(job.check.current_mean_c,1)} °C · 평균 차이 ${fmt(job.check.mean_shift_c,1)} °C</p>`:''}<ol class="temperature-steps">${steps.map(step=>`<li class="${esc(step.status)}"><strong>${esc(temperatureStepNames[step.key]||step.key)}</strong><span>${esc(statusName(step.status))}</span><small>${esc(step.completed_at?when(step.completed_at):step.started_at?when(step.started_at):'')}${step.detail?' · '+esc(temperatureStepDetail(step)):''}</small></li>`).join('')}</ol>${job.error?`<p class="inline-error">${esc(job.error)}</p>`:''}${renderTemperatureResult(job)}`:empty('재학습 기록이 없습니다.','온도 변화가 감지되면 학습이 시작됩니다.');
 $('temperature-jobs').innerHTML=(data.jobs||[]).map(item=>`<p>${esc(when(item.created_at||item.requested_at,true))} · ${esc(statusName(item.status))} · ${esc(temperatureStepNames[item.stage]||item.stage||'')}</p>`).join('');
 renderTemperatureChart();
 if(data.active_model?.version&&state.horizonCatalogLoaded&&String(data.active_model.version)!==state.horizonVersion)loadHorizonCatalog();
}
function renderTemperatureChart(){if(state.view!=='operations')return;const baseline=state.temperature?.reference?.mean_c;const rows=(state.temperatureObservations?.length?state.temperatureObservations:state.observations).map(row=>({...row,reference_c:finite(baseline)?baseline:null}));drawChart('temperature-chart',rows,'timestamp',[{key:'temperature_c',color:'#bc7736',label:'외기온'},{key:'reference_c',color:'#82968d',label:'기준 평균',dash:true}],'관측을 저장하면 온도 변화를 확인할 수 있습니다.','°C');}
async function checkTemperature(){const workspace=state.workspaceId,epoch=state.epoch;if(!workspace)return;$('temperature-check-button').disabled=true;try{const data=await api('/monitoring/temperature/check',{workspace,method:'POST'});if(epoch===state.epoch)renderTemperature(data);}catch(error){if(epoch===state.epoch)inline('temperature-error',errorText(error));}finally{$('temperature-check-button').disabled=false;}}
async function previewTemperatureFile(filename){
 if(!['01_baseline.csv','02_heatwave.csv','03_followup.csv'].includes(filename)||busy('preview')||busy('commit'))return;
 const revision=++state.fileSelectionRevision;
 await previewFile(new File([],filename,{type:'text/csv'}),true);
 if(revision===state.fileSelectionRevision&&state.preview)$('preview-panel').scrollIntoView({behavior:'smooth',block:'start'});
}
async function fullRefresh(){loadRetrainingActivity();$('global-notice').hidden=true;invalidateHorizonReport();invalidateHistory();try{await loadWorkspaces();}catch(e){notice(errorText(e),true);}await Promise.allSettled([refreshWorkspace(),loadCatalog(),loadHorizonCatalog()]);if(state.view==='data')loadRows();}
$('workspace-select').addEventListener('change',event=>{state.navigationRevision++;selectWorkspace(event.target.value);});document.querySelectorAll('[data-view]').forEach(button=>button.addEventListener('click',()=>{state.navigationRevision++;setView(button.dataset.view);}));window.addEventListener('hashchange',()=>{state.navigationRevision++;setView(location.hash.slice(1));});
$('refresh-horizon-button').addEventListener('click',()=>state.horizonCatalogLoaded?loadHorizonReport(true):loadHorizonCatalog());
$('retraining-toggle').addEventListener('click',toggleRetrainingDetails);
$('retraining-acknowledge')?.addEventListener('click',acknowledgePracticeRun);
$('temperature-check-button').addEventListener('click',checkTemperature);document.querySelectorAll('[data-temperature-file]').forEach(button=>button.addEventListener('click',()=>previewTemperatureFile(button.dataset.temperatureFile)));
$('refresh-horizon-models-button').addEventListener('click',loadHorizonCatalog);
$('refresh-button').addEventListener('click',fullRefresh);$('show-import-result').addEventListener('click',()=>{state.navigationRevision++;setView('data');$('import-job-panel').scrollIntoView({behavior:'smooth',block:'start'});});
$('choose-file-button').addEventListener('click',()=>$('csv-file').click());$('csv-file').addEventListener('change',event=>{const file=event.target.files[0];if(file){state.fileSelectionRevision++;previewFile(file);}event.target.value='';});$('sample-preview-button').addEventListener('click',previewSelectedSample);$('reload-samples-button').addEventListener('click',loadSamples);$('sample-select').addEventListener('change',event=>{state.fileSelectionRevision++;state.sampleId=event.target.value;renderSampleSelection();});
for(const name of ['dragenter','dragover'])$('drop-zone').addEventListener(name,event=>{event.preventDefault();$('drop-zone').classList.add('dragging');});for(const name of ['dragleave','drop'])$('drop-zone').addEventListener(name,event=>{event.preventDefault();$('drop-zone').classList.remove('dragging');});$('drop-zone').addEventListener('drop',event=>{const file=event.dataTransfer.files[0];if(file){state.fileSelectionRevision++;previewFile(file);}});
$('commit-button').addEventListener('click',commitPreview);$('check-job-button').addEventListener('click',pollActiveJob);$('open-job-workspace').addEventListener('click',async()=>{state.navigationRevision++;try{await loadWorkspaces();selectWorkspace(state.activeJob.workspace_id);setView('data');$('data-rows').closest('.panel').scrollIntoView({behavior:'smooth',block:'start'});}catch(e){notice(errorText(e),true);}});
$('previous-page').addEventListener('click',()=>{if(state.page>1){state.page--;loadRows();}});$('next-page').addEventListener('click',()=>{if(state.page*25<state.totalRows){state.page++;loadRows();}});$('retrain-button').addEventListener('click',()=>runOperation());document.querySelectorAll('[data-scenario]').forEach(button=>button.addEventListener('click',()=>runOperation(button.dataset.scenario)));
for(const id of ['data-jobs','operation-jobs'])$(id).addEventListener('click',async event=>{const button=event.target.closest('[data-open-job]');if(!button)return;state.navigationRevision++;const seq=++state.jobOpenSequence;try{const job=await api('/imports/jobs/'+encodeURIComponent(button.dataset.openJob));if(seq!==state.jobOpenSequence)return;setActiveJob(job);setView('data');$('import-job-panel').scrollIntoView({behavior:'smooth',block:'start'});}catch(e){notice(errorText(e),true);}});
async function init(){loadRetrainingActivity();loadSamples();loadHorizonCatalog();setView(location.hash.slice(1)||'overview');$('observation-chart').innerHTML=empty('벌통 목록을 확인하고 있습니다.');try{await loadWorkspaces();if(!state.workspaceId)notice('등록된 벌통이 없습니다. 데이터 화면에서 CSV 미리보기를 시작하세요.');loadCatalog();}catch(e){notice('벌통 목록을 불러오지 못했습니다. 새로고침으로 다시 시도할 수 있습니다.\n'+errorText(e),true);}try{const jobId=localStorage.getItem('beeops.import.job');if(jobId){const seq=state.jobSequence;const job=await api('/imports/jobs/'+encodeURIComponent(jobId));if(seq===state.jobSequence){if(state.workspaces.some(workspace=>workspace.workspace_id===job.workspace_id)||state.retrainingPendingImports.has(jobId))setActiveJob(job);else localStorage.removeItem('beeops.import.job');}}}catch{}updateControls();}
let resizeTimer;window.addEventListener('resize',()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(()=>{if(state.view==='overview'){renderObservations();renderAutomaticSummary();if(state.horizonReport)renderHorizonReport();}else if(state.view==='operations')renderTemperatureChart();},120);});
init();setInterval(()=>{if(!document.hidden){refreshWorkspace();if(state.activeJob&&(!terminal(state.activeJob.status)||!state.refreshedJobs.has(state.activeJob.job_id)))pollActiveJob();}},3000);setInterval(()=>{if(!document.hidden)loadRetrainingActivity();},2000);document.addEventListener('visibilitychange',()=>{if(!document.hidden){loadRetrainingActivity();refreshWorkspace();if(state.activeJob)pollActiveJob();}});
