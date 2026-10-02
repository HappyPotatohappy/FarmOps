/* Product regressions at the real frontend DOM / HTTP boundaries. No forecasting reimplementation. */
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../../app/static/app.js'), 'utf8').split('init();setInterval')[0];
const html = fs.readFileSync(path.join(__dirname, '../../app/static/index.html'), 'utf8');
const registry = 'BeeOPS_Horizon_Weight';
const removedIds = ['horizon-hours', 'horizon-model-select', 'harvest-preferences-form', 'harvest-work-start',
  'harvest-work-end', 'prediction-model-select', 'comparison-version'];
const copy = value => JSON.parse(JSON.stringify(value));
const catalog = () => ({registry_id: registry, default_version: '1', versions: ['1', '2'].map(version => ({
  version, run_id: 'run-' + version, context_hours: 168, max_horizon_hours: 168,
  components: version === '2' ? ['tirex2', 'lstm'] : ['chronos2', 'lstm'],
  weights: version === '2' ? {tirex2: .9, lstm: .1} : {chronos2: .9, lstm: .1},
  validation: {status: 'insufficient', calibrated: false, horizons: {
    '1': {test_mae_kg: .007753281, origin_count: 18, independent_windows: 18},
    '168': {test_mae_kg: 7.65, origin_count: 2, independent_windows: 1}}}
}))});
function report(workspace = 'A', snapshot = 'snapshot-' + workspace) {
  const identity = {workspace_id: workspace, hive_id: workspace, snapshot_id: snapshot,
    forecast_id: `forecast-${workspace}-${snapshot}`, as_of: '2025-01-01T00:00:00+00:00',
    horizon_hours: 168, requested_horizon_hours: 168, historical: true, time_basis: 'source_clock_unknown',
    model: {registry_id: registry, version: '2', run_id: 'run-2', weights: {tirex2: .9, lstm: .1}}};
  const trajectory = Array.from({length: 168}, (_, i) => ({timestamp: new Date(Date.UTC(2025, 0, 1, i + 1)).toISOString(),
    weight_kg: 50.125 + i / 100, lower_kg: null, upper_kg: null}));
  return {...identity, selection: {mode: 'automatic', model_version: '2', horizon_hours: 168},
    status: 'inspection_window', candidate_window: {start: trajectory[144].timestamp, end: trajectory.at(-1).timestamp},
    reasons: ['예측 무게가 높은 구간을 채밀 추천 기간으로 표시합니다.'], evidence: {recent_gain_kg: .6},
    field_checks: {maturity: 'unknown', reserves: 'unknown'}, actionability: 'requires_field_check', rule_version: 'harvest-v2',
    planning: {work_windows: [], field_checks: [{key: 'honey_moisture', label: '꿀 수분', state: 'unknown', value: null,
      note: '채밀 전 수분을 현장에서 확인하세요.'}], sources: []},
    forecast: {...identity, status: 'ok', trajectory,
      validation: copy(catalog().versions[1].validation), reasons: []}};
}
function fixture(handler = () => report()) {
  const nodes = new Map(), calls = [], windowEvents = new Map();
  const ids = new Set([...html.matchAll(/\bid=["']([^"']+)["']/g)].map(match => match[1]));
  const node = id => {
    if (!nodes.has(id)) nodes.set(id, {id, textContent: '', innerHTML: '', hidden: false, disabled: false, value: '',
      open: false, style: {}, dataset: {}, clientWidth: 850, clientHeight: 290, className: '',
      classList: {add(){}, remove(){}, toggle(){}, contains(){return false;}}, lastElementChild: {textContent: ''},
      addEventListener(){}, reset(){}, setAttribute(){}, removeAttribute(){}, append(){}, close(){}, showModal(){},
      scrollIntoView(){}, closest(){return node('parent');}});
    return nodes.get(id);
  };
  const context = {console, URL, AbortController, FormData, File, Blob, Map, Set, Date, Number, Math, JSON, Promise,
    setTimeout, clearTimeout, location: {origin: 'http://beeops.test', hash: '#overview'}, history: {replaceState(){}},
    localStorage: {getItem(){return null;}, setItem(){}},
    window: {addEventListener(name, listener){windowEvents.set(name, listener);}, scrollTo(){}},
    document: {hidden: false, getElementById: id => ids.has(id) ? node(id) : null,
      querySelectorAll(){return [];}, createElement: type => node('created-' + type), addEventListener(){}},
    fetch: async (url, options) => {
      calls.push({url, options}); const payload = await handler(url, options);
      return {ok: true, text: async () => JSON.stringify(payload),
        blob: async () => payload instanceof Blob ? payload : new Blob([JSON.stringify(payload)])};
    }};
  vm.createContext(context); vm.runInContext(source, context);
  const state = vm.runInContext('state', context);
  Object.assign(state, {workspaceId: 'A', epoch: 1, health: {version: '1', model_loaded: true,
    data: {rows: 300, snapshot_id: 'snapshot-A', hive_id: 'A'}, quality: {}},
    workspaces: ['A', 'B'].map(id => ({workspace_id: id, hive_id: id, name: id, kind: 'public_real'})),
    horizonCatalog: catalog(), horizonCatalogLoaded: true, horizonVersion: '2', horizonHours: 168});
  return {state, context, calls, node, windowEvents, run: code => vm.runInContext(code, context)};
}
const flush = () => new Promise(resolve => setTimeout(resolve, 20));

test('product HTML contains no forecast model, date-window or working-time selectors', () => {
  for (const id of removedIds) assert.equal(new RegExp(`id=["']${id}["']`).test(html), false, `${id} remains in the product`);
});

test('automatic recommendation uses only workspace and is cached until refresh', async () => {
  const f = fixture();
  await f.run('loadHorizonReport()'); await f.run('loadHorizonReport()');
  assert.equal(f.calls.length, 1);
  assert.equal(f.calls[0].url, '/harvest/recommendation?workspace_id=A');
  assert.equal(f.calls[0].options.method, 'GET');
  assert.equal(f.state.horizonReport.model.version, '2');
  assert.equal(f.state.horizonReport.horizon_hours, 168);
  assert.match(f.node('horizon-chart').innerHTML, /<svg/);
  assert.match(f.node('harvest-card').innerHTML, /채밀 추천 기간/);
  await f.run('loadHorizonReport(true)'); assert.equal(f.calls.length, 2);
});

test('catalog always selects the supported TiRex-2/LSTM model even if legacy default comes first', async () => {
  const f = fixture(url => url === '/models/horizon' ? catalog() : report());
  f.state.horizonVersion = '1'; f.state.horizonHours = 24;
  await f.run('loadHorizonCatalog()'); await flush();
  assert.equal(f.state.horizonVersion, '2'); assert.equal(f.state.horizonHours, 168);
  assert.equal(f.state.horizonReport.model.version, '2');
  const card = f.node('horizon-model-catalog').innerHTML;
  assert.equal((card.match(/class="horizon-model-card"/g) || []).length, 1);
  assert.match(card, /TiRex-2/); assert.match(card, /LSTM/); assert.doesNotMatch(card, /Chronos|<select|<option/);
});

test('missing active model never silently falls back to a different model', async () => {
  const legacy = catalog(); legacy.versions = legacy.versions.filter(model => model.version === '1');
  const f = fixture(url => url === '/models/horizon' ? legacy : report());
  f.state.horizonReport = report(); await f.run('loadHorizonCatalog()'); await flush();
  assert.equal(f.state.horizonReport, null);
  assert(!f.calls.some(call => call.url.startsWith('/harvest/')));
  assert.doesNotMatch(f.node('horizon-model-catalog').innerHTML, /Chronos/);
  assert.doesNotMatch(f.node('horizon-chart').innerHTML, /<svg/);
});

test('the top forecast and test MAE belong to the active automatic model', async () => {
  const f = fixture(); await f.run('loadHorizonReport()');
  assert.match(f.node('stat-prediction').innerHTML, /50\.125/);
  assert.equal(f.node('stat-model').textContent, 'TiRex-2 + LSTM');
  assert.match(f.node('stat-quality').innerHTML, /0\.0078/);
  assert.match(f.node('stat-quality-note').textContent, /test|테스트|평가/);
  assert.doesNotMatch(f.node('stat-quality-note').textContent, /운영.*18|실측.*18/);
});

test('late legacy health results cannot overwrite the automatic summary', async () => {
  const f = fixture(); await f.run('loadHorizonReport()');
  f.run("renderTraining=()=>{};loadCatalog=async()=>{};");
  f.run("renderHealth({version:'99',model_loaded:true,data:{rows:300,snapshot_id:'snapshot-A'},quality:{mae:99},retraining:{},demo:{}})");
  await flush();
  assert.equal(f.node('stat-model').textContent, 'TiRex-2 + LSTM');
  assert.match(f.node('stat-prediction').innerHTML, /50\.125/);
  assert.match(f.node('stat-quality').innerHTML, /0\.0078/);
});

test('missing test MAE is shown as unavailable, without substituting development or legacy scores', async () => {
  const payload = report(); payload.forecast.validation.horizons['1'] = {equal_hive_mae_kg: .3333, origin_count: 9};
  const f = fixture(() => payload); f.state.horizonCatalog.versions[1].validation = copy(payload.forecast.validation);
  await f.run('loadHorizonReport()');
  assert.doesNotMatch(f.node('stat-quality').innerHTML, /0\.3333|0\.0000/);
  assert.match(f.node('stat-quality').innerHTML, /—|확인|없/);
});

test('only the recommended period is displayed, without saved working-window alternatives', async () => {
  const payload = report(); payload.planning.work_windows = [{start: '2099-05-01T08:00:00Z', end: '2099-05-01T12:00:00Z'}];
  payload.preferences = {recorded_timezone: 'Europe/Riga', working_hours: {start: '08:00', end: '12:00'}};
  const f = fixture(() => payload); await f.run('loadHorizonReport()');
  const content = f.node('harvest-card').innerHTML;
  assert.match(content, /채밀 추천 기간/);
  assert.doesNotMatch(content, /2099|<input|<select|<form|작업 시간 선택/);
  assert.match(content, /수분|숙성/); assert.match(content, /비축/);
  assert(!f.calls.some(call => call.url.startsWith('/harvest/preferences')));
});

test('resize redraws the forecast for actual width without additional inference', async () => {
  const f = fixture(); await f.run('loadHorizonReport()');
  assert.match(f.node('horizon-chart').innerHTML, /viewBox="0 0 826 282"/);
  f.node('horizon-chart').clientWidth = 375;
  f.windowEvents.get('resize')(); await new Promise(resolve => setTimeout(resolve, 150));
  assert.match(f.node('horizon-chart').innerHTML, /viewBox="0 0 351 282"/);
  assert.equal(f.calls.length, 1);
});

test('returning from models redraws a forecast computed while its chart was hidden', async () => {
  const f = fixture(); f.state.view = 'models'; f.node('horizon-chart').clientWidth = 0;
  await f.run('loadHorizonReport()');
  assert.match(f.node('horizon-chart').innerHTML, /viewBox="0 0 300 282"/);
  f.node('horizon-chart').clientWidth = 850; f.run("refreshWorkspace=async()=>{};setView('overview')");
  assert.match(f.node('horizon-chart').innerHTML, /viewBox="0 0 826 282"/);
  assert.equal(f.calls.length, 1);
});

test('workspace navigation clears the old recommendation and ignores its late response', async () => {
  let finish; const f = fixture(url => /^\/harvest\/(report|recommendation)\?/.test(url) ? new Promise(resolve => {finish = resolve;}) : {});
  const pending = f.run('loadHorizonReport()'); f.run("refreshWorkspace=async()=>{};selectWorkspace('B')");
  assert.equal(f.state.horizonReport, null); assert.doesNotMatch(f.node('harvest-card').innerHTML, /2025/);
  finish(report()); await pending;
  assert.equal(f.state.horizonReport, null); assert.doesNotMatch(f.node('stat-prediction').innerHTML, /50\.125/);
});

test('a changed upload snapshot invalidates a pending result even before polling finishes', async () => {
  let finish; const f = fixture(() => new Promise(resolve => {finish = resolve;}));
  const pending = f.run('loadHorizonReport()');
  f.state.health.data.snapshot_id = 'after-upload'; finish(report()); await pending;
  assert.equal(f.state.horizonReport, null); assert.doesNotMatch(f.node('horizon-chart').innerHTML, /<svg/);
});

for (const field of ['workspace', 'version', 'horizon', 'snapshot', 'embedded_identity', 'run', 'registry', 'selection']) {
  test(`reject a mismatched ${field} identity without displaying its period or forecast`, async () => {
    const payload = report();
    if (field === 'workspace') payload.workspace_id = 'B';
    if (field === 'version') payload.forecast.model.version = '1';
    if (field === 'horizon') payload.forecast.horizon_hours = 72;
    if (field === 'snapshot') payload.snapshot_id = 'other';
    if (field === 'embedded_identity') payload.forecast.forecast_id = 'different';
    if (field === 'run') payload.forecast.model.run_id = 'other-run';
    if (field === 'registry') payload.forecast.model.registry_id = 'BeeOPS_Common_Weight';
    if (field === 'selection') payload.selection = {mode: 'manual', model_version: '1', horizon_hours: 24};
    const f = fixture(() => payload); await f.run('loadHorizonReport()');
    assert.equal(f.state.horizonReport, null); assert.equal(f.node('horizon-error').hidden, false);
    assert.doesNotMatch(f.node('horizon-chart').innerHTML, /<svg/); assert.doesNotMatch(f.node('harvest-card').innerHTML, /2025/);
  });
}

test('operational validation labels do not block a finite complete forecast or recommendation', async () => {
  const payload = report(); payload.forecast.validation = {status: 'insufficient', calibrated: false};
  payload.species = 'unknown'; payload.evidence.recent_event = 'inspection';
  const f = fixture(() => payload); await f.run('loadHorizonReport()');
  assert(f.state.horizonReport); assert.match(f.node('horizon-chart').innerHTML, /harvest-window/);
  assert.match(f.node('harvest-card').innerHTML, /채밀 추천 기간/);
  assert.doesNotMatch(f.node('harvest-card').innerHTML + f.node('horizon-validation').innerHTML, /보류|추천불가|미검증|조건부/);
});

test('historical recommendation remains anchored to observation date', async () => {
  const payload = report(); payload.forecast.validation.calibrated = true;
  payload.forecast.trajectory.forEach(row => {row.lower_kg = row.weight_kg - .2; row.upper_kg = row.weight_kg + .2;});
  const f = fixture(() => payload); await f.run('loadHorizonReport()');
  assert.match(f.node('horizon-time-note').textContent, /과거.*관측일/);
  assert.match(f.node('horizon-chart').innerHTML, /horizon-band|예측 범위/);
  assert.match(f.node('harvest-card').innerHTML, /2025/);
  assert.doesNotMatch(f.node('harvest-card').innerHTML, /성공률|꿀 수확량/);
});

for (const status of ['insufficient_history', 'model_unavailable', 'unsupported_horizon']) {
  test(`${status} clears the forecast without fabricating a candidate`, async () => {
    const payload = report(); payload.status = 'forecast_unavailable'; payload.candidate_window = null;
    payload.forecast.status = status; payload.forecast.trajectory = []; payload.forecast.reasons = ['예측 입력과 모델을 확인해 주세요.'];
    const f = fixture(() => payload); await f.run('loadHorizonReport()');
    assert.equal(f.state.horizonReport.forecast.status, status);
    assert.doesNotMatch(f.node('horizon-chart').innerHTML, /<svg/);
    assert.doesNotMatch(f.node('stat-prediction').innerHTML, /50\.125|0\.000/);
  });
}

test('an empty success and subsequent network error cannot leave old recommendation or summary', async () => {
  let mode = 'ok'; const f = fixture(() => {
    if (mode === 'error') throw new Error('Inference unavailable');
    const payload = report(); if (mode === 'empty') payload.forecast.trajectory = []; return payload;
  });
  await f.run('loadHorizonReport()'); assert(f.state.horizonReport);
  mode = 'empty'; await f.run('loadHorizonReport(true)'); assert.equal(f.state.horizonReport, null);
  mode = 'error'; await f.run('loadHorizonReport(true)'); assert.equal(f.state.horizonReport, null);
  assert.match(f.node('horizon-error').textContent, /Inference unavailable/);
  assert.doesNotMatch(f.node('horizon-chart').innerHTML, /<svg/);
  assert.doesNotMatch(f.node('stat-prediction').innerHTML, /50\.125/);
});

test('candidate outside forecast and invalid trajectory timestamps are rejected', async () => {
  const payload = report(); payload.candidate_window.end = '2025-02-04T00:00:00Z';
  const f = fixture(() => payload); await f.run('loadHorizonReport()'); assert.equal(f.state.horizonReport, null);
  payload.candidate_window = report().candidate_window; payload.forecast.trajectory[0].timestamp = payload.as_of;
  await f.run('loadHorizonReport(true)'); assert.equal(f.state.horizonReport, null);
});

test('untrusted reason text is escaped when rendering the recommendation', async () => {
  const payload = report(); payload.reasons = ['<img src=x onerror=alert(1)>'];
  const f = fixture(() => payload); await f.run('loadHorizonReport()');
  assert.match(f.node('harvest-card').innerHTML, /&lt;img/); assert.doesNotMatch(f.node('harvest-card').innerHTML, /<img/);
});

test('selected public CSV uses multipart preview and does not auto-commit', async () => {
  let uploaded;
  const samples = {default_sample_id: 'other_hive', samples: [{id: 'other_hive', label: '다른 공개 벌통', hive_id: 'B', download_url: '/data/samples/other_hive.csv'}]};
  const f = fixture(async (url, options) => {
    if (url === '/data/samples') return samples;
    if (url === '/data/samples/other_hive.csv') return new Blob(['timestamp,hive_id,weight_kg,temperature_c,event\n2026-01-01T00:00:00Z,B,20,15,normal\n']);
    if (url === '/imports/preview') {
      uploaded = options.body.get('file');
      return {preview_id: 'chosen', filename: uploaded.name, can_commit: true, workspace_id: 'B', hive_id: 'B',
        row_count: 1, insert_count: 1, duplicate_count: 0, conflict_count: 0, sample: [], errors: [], warnings: []};
    }
    return [];
  });
  await f.run('loadSamples()'); await f.run('previewSelectedSample()');
  assert.equal(uploaded.name, 'other_hive.csv'); assert.match(await uploaded.text(), /B,20,15/);
  assert.equal(f.state.preview.preview_id, 'chosen'); assert(!f.calls.some(call => call.url === '/imports/commit'));
  assert.equal(f.node('sample-download-link').href, '/data/samples/other_hive.csv');
});

for (const switched of [false, true]) test(`completed CSV import ${switched ? 'preserves later deliberate hive choice' : 'opens imported hive with automatic model'}`, async () => {
  const job = {job_id: 'job-1', workspace_id: 'B', status: 'completed', stage: 'finished', progress: 100,
    result: {ingest: {inserted: 0, duplicate: 724, rows: 724}}};
  const workspaces = ['A', 'B'].map(id => ({workspace_id: id, hive_id: id, name: id, kind: 'public_real'}));
  const f = fixture(url => {
    if (url.startsWith('/imports/jobs/job-1')) return job;
    if (url === '/workspaces') return {workspaces, default_workspace_id: 'A'};
    if (url === '/models/catalog') return [];
    if (url.startsWith('/health')) {const id = new URL(url, 'http://beeops.test').searchParams.get('workspace_id');
      return {model_loaded: true, version: '1', data: {rows: 724, snapshot_id: 'snapshot-' + id}, quality: {}, retraining: {}, demo: {}};}
    if (url.startsWith('/config')) return {runtime_type: 'public_real'};
    if (url.startsWith('/harvest/recommendation')) return report('B');
    return [];
  });
  Object.assign(f.state, {workspaces, activeJob: {job_id: 'job-1', status: 'running'}, jobSequence: 1, navigationRevision: switched ? 2 : 1});
  f.state.importNavigation.set('job-1', {revision: 1, workspace: 'B'});
  await f.run('pollActiveJob()'); await flush();
  assert.equal(f.state.workspaceId, switched ? 'A' : 'B'); assert.equal(f.state.horizonVersion, '2');
  if (!switched) {assert.equal(f.state.view, 'overview'); assert.equal(f.state.horizonReport?.workspace_id, 'B');
    assert.match(f.node('overview-import-title').textContent, /변경 없음.*신규 0개.*724/);}
  assert(!f.calls.some(call => call.url.startsWith('/forecast/report') || call.url.startsWith('/harvest/preferences')));
});

test('observation chart restores real target-aligned active model predictions', () => {
 const f=fixture(); f.state.observations=[{timestamp:'2025-01-01T01:00:00Z',weight_kg:50,temperature_c:24}];
 f.state.historyReport={workspace_id:'A',snapshot_id:'snapshot-A',model:{version:'2',run_id:'run-2'},predictions:[{target_timestamp:'2025-01-01T01:00:00Z',predicted_weight_kg:49.25}],metrics:{mae_kg:.75,count:1}};
 f.run('renderObservations()');
 assert.match(f.node('observation-chart').innerHTML,/#d79e2c/);
 assert.match(f.node('observation-chart').innerHTML,/49\.250/);
 assert.match(f.node('history-metrics').textContent,/0\.750/);
});

test('observation chart refuses predictions from another hive or model', () => {
 const f=fixture();f.state.observations=[{timestamp:'2025-01-01T01:00:00Z',weight_kg:50,temperature_c:24}];
 f.state.historyReport={workspace_id:'B',snapshot_id:'snapshot-A',model:{version:'2',run_id:'run-2'},predictions:[{target_timestamp:'2025-01-01T01:00:00Z',predicted_weight_kg:49.25}],metrics:{mae_kg:.75,count:1}};
 f.run('renderObservations()');assert.doesNotMatch(f.node('observation-chart').innerHTML,/49\.250/);
});

test('a retrained active ensemble replaces the fixed initial bundle without choices', async () => {
 const next=catalog();const trained=copy(next.versions[1]);trained.version='3';trained.run_id='trained-3';next.versions.push(trained);next.default_version='3';next.active_version='3';
 const f=fixture(url=>url==='/models/horizon'?next:report());
 await f.run('loadHorizonCatalog()');assert.equal(f.state.horizonVersion,'3');
 assert.equal((f.node('horizon-model-catalog').innerHTML.match(/class="horizon-model-card"/g)||[]).length,1);
 assert.doesNotMatch(f.node('horizon-model-catalog').innerHTML,/<select/);
});

test('temperature pipeline renders actual failed stage and never reports deployment success', () => {
 const f=fixture();f.run(`renderTemperature({workspace_id:'A',enabled:true,reference:{count:168,mean_c:24},check:{status:'drift',current_mean_c:32,mean_shift_c:8},job:{job_id:'job-1',status:'failed',stage:'training',error:'learning failed',steps:[{key:'training',status:'failed'}]}})`);
 assert.match(f.node('temperature-job').innerHTML,/learning failed/);
 assert.doesNotMatch(f.node('temperature-job').innerHTML,/배포 완료|적용 완료/);
 assert.match(f.node('temperature-current').textContent,/32/);
});

test('temperature results cannot cross workspace navigation', () => {
 const f=fixture();f.run("renderTemperature({workspace_id:'B',enabled:true,check:{status:'drift',current_mean_c:55}})");
 assert.doesNotMatch(f.node('temperature-current').textContent,/55/);
});
