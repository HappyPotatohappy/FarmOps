"""Durable observation/prediction ledger; no silent overwrites or duplicate scoring."""
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
import hashlib
import json
import math
import sqlite3
import threading
import uuid
from .contracts import validate_series


# Provisional educational added-value policy, not field-validated business value.
MINIMUM_BASELINE_GAIN = .05
BASELINE_ZERO_TOLERANCE_KG = 1e-12


def utcnow(): return datetime.now(timezone.utc).isoformat()
def digest(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}-{uuid.uuid4().hex}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def same_observation_values(left,right):
    # CSV and pandas parsers can differ by a final binary digit. Accept only
    # numerical roundoff; retain the originally persisted measurement verbatim.
    return left['hive_id']==right['hive_id'] and all(
        math.isclose(left[key],right[key],rel_tol=0,abs_tol=1e-10)
        for key in ('weight_kg','temperature_c'))


class Store:
    def __init__(self,runtime,prediction_namespace=None):
        if prediction_namespace not in (None,'shared'):
            raise ValueError('Unsupported prediction namespace')
        self.prediction_table='predictions_shared' if prediction_namespace=='shared' else 'predictions'
        self.cutoff_prefix='shared-model-cutoff-' if prediction_namespace=='shared' else 'model-cutoff-'
        self.path=Path(runtime); self.path.mkdir(parents=True,exist_ok=True)
        self.lock=threading.RLock()
        with self.db() as c:
            c.executescript('''
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS observations(timestamp TEXT PRIMARY KEY,hive_id TEXT NOT NULL,
              weight_kg REAL NOT NULL,temperature_c REAL NOT NULL,event TEXT NOT NULL,raw_event TEXT NOT NULL,source TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS predictions(prediction_id TEXT PRIMARY KEY,target_timestamp TEXT NOT NULL,
              hive_id TEXT NOT NULL,predicted_weight_kg REAL NOT NULL,model_version TEXT NOT NULL,input_hash TEXT NOT NULL,
              created_at TEXT NOT NULL,context_valid INTEGER NOT NULL,UNIQUE(target_timestamp,model_version));
            CREATE TABLE IF NOT EXISTS alerts(id INTEGER PRIMARY KEY,created_at TEXT,category TEXT,level TEXT,message TEXT,details TEXT);
            CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
            ''')
            if prediction_namespace=='shared':
                c.execute('''CREATE TABLE IF NOT EXISTS predictions_shared(
                  prediction_id TEXT PRIMARY KEY,target_timestamp TEXT NOT NULL,
                  hive_id TEXT NOT NULL,predicted_weight_kg REAL NOT NULL,model_version TEXT NOT NULL,input_hash TEXT NOT NULL,
                  created_at TEXT NOT NULL,context_valid INTEGER NOT NULL,UNIQUE(target_timestamp,model_version))''')

    @contextmanager
    def db(self):
        with self.lock:
            c=sqlite3.connect(self.path/'operations.sqlite',timeout=30)
            c.row_factory=sqlite3.Row
            try:
                yield c; c.commit()
            except BaseException:
                c.rollback(); raise
            finally: c.close()

    def set_meta(self,key,value):
        with self.db() as c: c.execute('INSERT OR REPLACE INTO meta VALUES (?,?)',(key,json.dumps(value)))

    def get_meta(self,key,default=None):
        with self.db() as c: row=c.execute('SELECT value FROM meta WHERE key=?',(key,)).fetchone()
        return json.loads(row[0]) if row else default

    def ingest(self,rows,source,*,temperature_practice=False):
        rows=validate_series(rows)
        with self.db() as c:
            additions,duplicates=self._ingest_plan(c,rows,temperature_practice=temperature_practice)
            for row,event in additions:
                c.execute('INSERT INTO observations VALUES (?,?,?,?,?,?,?)',
                          (row['timestamp'],row['hive_id'],row['weight_kg'],row['temperature_c'],event,row['event'],source))
                if event!='normal':
                    category='sensor' if event=='sensor_fault' else 'colony' if event=='colony_alert' else 'operation'
                    self._alert(c,category,'WARN',f'{row["timestamp"]}: {event} 관측 — 품질 평가·정상 학습 제외',row)
            if additions:
                c.execute('INSERT OR REPLACE INTO meta VALUES (?,?)',('source',json.dumps(source)))
        return {'inserted':len(additions),'duplicate':duplicates,**self.data_status()}

    @staticmethod
    def _ingest_plan(c,rows,*,temperature_practice=False):
        reserved=c.execute("SELECT value FROM meta WHERE key='workspace_reserved_hives'").fetchone()
        if reserved and rows[0]['hive_id'] in json.loads(reserved[0]):
            raise ValueError('This hive already has a dedicated workspace; select its workspace_id')
        binding=c.execute("SELECT value FROM meta WHERE key='workspace_binding'").fetchone()
        if binding and json.loads(binding[0])!=rows[0]['hive_id']:
            raise ValueError('This workspace is bound to a different hive_id')
        first=c.execute('SELECT hive_id FROM observations LIMIT 1').fetchone()
        if first and first[0]!=rows[0]['hive_id']: raise ValueError('This runtime belongs to a different hive_id')
        prefix=[]
        if temperature_practice:
            from .temperature_practice import validate_practice_recovery
            existing=[dict(row) for row in c.execute('SELECT * FROM observations ORDER BY timestamp')]
            validate_practice_recovery(rows,existing)
            if existing:
                prefix=[row for row in rows if row['timestamp'] < existing[0]['timestamp']]
                if prefix and datetime.fromisoformat(existing[0]['timestamp'])-datetime.fromisoformat(prefix[-1]['timestamp'])!=timedelta(hours=1):
                    raise ValueError('Built-in practice prefix must meet the existing hourly ledger')
                rows=rows[len(prefix):]
        prev=c.execute('SELECT timestamp,weight_kg FROM observations ORDER BY timestamp DESC LIMIT 1').fetchone()
        additions=[(row,row['event']) for row in prefix]; duplicates=0
        for row in rows:
            old=c.execute('SELECT * FROM observations WHERE timestamp=?',(row['timestamp'],)).fetchone()
            if old:
                if not same_observation_values(old,row) or row['event'] not in (old['raw_event'],old['event']):
                    raise ValueError(f'Conflicting observation at {row["timestamp"]}; original retained')
                duplicates+=1; continue
            if prev and row['timestamp']<prev['timestamp']:
                raise ValueError('Backdated insertion is not supported; use a new runtime for another dataset')
            if prev and datetime.fromisoformat(row['timestamp'])-datetime.fromisoformat(prev['timestamp'])!=timedelta(hours=1):
                raise ValueError('New observations must continue the hourly ledger without gaps')
            event=row['event']
            if prev and event=='normal' and abs(row['weight_kg']-prev['weight_kg'])>1: event='colony_alert'
            additions.append((row,event)); prev=row
        return additions,duplicates

    def preview_ingest(self,rows,*,temperature_practice=False):
        rows=validate_series(rows)
        with self.db() as c:
            additions,duplicates=self._ingest_plan(c,rows,temperature_practice=temperature_practice)
            count=c.execute('SELECT COUNT(*) FROM observations').fetchone()[0]
        return {'inserted':len(additions),'duplicate':duplicates,'rows':count+len(additions)}

    def set_model_cutoff(self,version,cutoff):
        # An explicit null means metadata is unavailable: fail closed. Absence
        # retains the historical test-fixture contract for non-MLflow models.
        self.set_meta(f'{self.cutoff_prefix}{version}',cutoff)

    def observations(self,limit=None):
        with self.db() as c:
            if limit:
                rows=c.execute('SELECT * FROM (SELECT * FROM observations ORDER BY timestamp DESC LIMIT ?) ORDER BY timestamp',(limit,)).fetchall()
            else: rows=c.execute('SELECT * FROM observations ORDER BY timestamp').fetchall()
        return [{k:r[k] for k in ('timestamp','hive_id','weight_kg','temperature_c','event')} for r in rows]

    def data_status(self):
        with self.db() as c:
            r=dict(c.execute('SELECT COUNT(*) rows,MIN(timestamp) start,MAX(timestamp) end,MIN(hive_id) hive_id FROM observations').fetchone())
        sources=self.sources()
        r['sources']=sources
        r['source']=sources[0] if len(sources)==1 else ('합성 교육 데이터 (복수 시나리오)' if sources and all('합성' in s for s in sources) else '혼합 출처: '+', '.join(sources)) if sources else '데이터 없음'
        r['snapshot_id']=digest(self.observations())
        return r

    def sources(self,start=None,end=None):
        with self.db() as c:
            rows=c.execute('SELECT DISTINCT source FROM observations WHERE timestamp>=? AND timestamp<=? ORDER BY source',
                           (start or '',end or '9999')).fetchall()
        return [r[0] for r in rows]

    def record_prediction(self,sequence,predicted,version):
        rows=validate_series(sequence,24,24)
        target=(datetime.fromisoformat(rows[-1]['timestamp'])+timedelta(hours=1)).isoformat()
        valid=all(r['event']=='normal' for r in rows)
        with self.db() as c:
            for row in rows:
                stored=c.execute('SELECT * FROM observations WHERE timestamp=?',(row['timestamp'],)).fetchone()
                if stored is None or not same_observation_values(stored,row):
                    raise ValueError('Prediction context must match the stored observation ledger; upload observations first')
                if stored['event']!='normal': valid=False
                row['event']=stored['event']
                for key in ('weight_kg','temperature_c'): row[key]=stored[key]
            c.execute(f'INSERT OR IGNORE INTO {self.prediction_table} VALUES (?,?,?,?,?,?,?,?)',
                      (uuid.uuid4().hex,target,rows[0]['hive_id'],float(predicted),str(version),digest(rows),utcnow(),int(valid)))
        return self.predictions(limit=1,target_timestamp=target,version=str(version))[0]

    def predictions(self,limit=100,*,target_timestamp=None,version=None):
        conditions=[]; parameters=[]
        if target_timestamp is not None:
            conditions.append('p.target_timestamp=?'); parameters.append(target_timestamp)
        if version is not None:
            conditions.append('p.model_version=?'); parameters.append(str(version))
        where='WHERE '+' AND '.join(conditions) if conditions else ''
        with self.db() as c:
            rows=c.execute(f'''SELECT p.*,o.weight_kg actual_weight_kg,o.event,
              CASE WHEN o.weight_kg IS NOT NULL THEN ABS(o.weight_kg-p.predicted_weight_kg) END error_kg,
              previous.weight_kg persistence_weight_kg,seasonal.weight_kg seasonal_24h_weight_kg,
              CASE WHEN o.weight_kg IS NOT NULL AND previous.weight_kg IS NOT NULL
                   THEN ABS(o.weight_kg-previous.weight_kg) END persistence_error_kg,
              CASE WHEN o.weight_kg IS NOT NULL AND seasonal.weight_kg IS NOT NULL
                   THEN ABS(o.weight_kg-seasonal.weight_kg) END seasonal_24h_error_kg
              FROM {self.prediction_table} p LEFT JOIN observations o ON p.target_timestamp=o.timestamp AND p.hive_id=o.hive_id
              LEFT JOIN observations previous ON previous.timestamp=strftime('%Y-%m-%dT%H:%M:%S+00:00',p.target_timestamp,'-1 hour')
                   AND previous.hive_id=p.hive_id
              LEFT JOIN observations seasonal ON seasonal.timestamp=strftime('%Y-%m-%dT%H:%M:%S+00:00',p.target_timestamp,'-24 hours')
                   AND seasonal.hive_id=p.hive_id
              '''+where+' ORDER BY p.target_timestamp DESC,p.created_at DESC LIMIT ?',(*parameters,limit)).fetchall()
            cutoffs={r['key'][len(self.cutoff_prefix):]:json.loads(r['value']) for r in c.execute(
                'SELECT key,value FROM meta WHERE key LIKE ?',(self.cutoff_prefix+'%',))}
        result=[]
        for row in rows:
            item=dict(row); version=str(item['model_version']); cutoff=cutoffs.get(version)
            temporal=version not in cutoffs or (cutoff is not None and datetime.fromisoformat(item['target_timestamp'])>datetime.fromisoformat(cutoff))
            item['monitoring_after']=cutoff
            item['monitoring_eligible']=bool(temporal and item['context_valid'] and item['event'] in (None,'normal'))
            result.append(item)
        return result

    def quality(self,version,window=24,threshold=.15):
        rows=[r for r in self.predictions(100000) if r['model_version']==str(version)
              and r['event']=='normal' and r['monitoring_eligible'] and r['actual_weight_kg'] is not None][:window]
        mae=sum(r['error_kg'] for r in rows)/len(rows) if rows else None
        scores={}
        for name in ('persistence','seasonal_24h'):
            matched=[r for r in rows if r[f'{name}_error_kg'] is not None]
            scores[name]={'count':len(matched),
                          'mae_kg':sum(r[f'{name}_error_kg'] for r in matched)/len(matched) if matched else None}
        # Compare both methods on exactly the model's selected target set.
        # A baseline with missing targets cannot win through an easier subset.
        available=[name for name,score in scores.items() if rows and score['count']==len(rows)]
        best=min(available,key=lambda name:scores[name]['mae_kg']) if available else None
        baseline=scores[best]['mae_kg'] if best else None
        support=scores[best]['count'] if best else 0
        skill=1-mae/baseline if baseline is not None and baseline>BASELINE_ZERO_TOLERANCE_KG else None
        absolute='insufficient_data' if len(rows)<window else 'degraded' if mae>threshold else 'ok'
        if len(rows)<window or support<window:
            comparison,reason='insufficient_data','not_enough_matched_targets'
        elif baseline<=BASELINE_ZERO_TOLERANCE_KG:
            comparison,reason='underperforming','baseline_is_perfect'
        elif skill+1e-12<MINIMUM_BASELINE_GAIN:
            comparison='underperforming'
            reason='model_worse_than_baseline' if skill<0 else 'gain_below_minimum'
        else:
            comparison,reason='useful','material_same_target_improvement'
        status=absolute if absolute!='ok' else 'ok' if comparison=='useful' else comparison
        return {'status':status,'absolute_status':absolute,
                'mae_kg':mae,'count':len(rows),'window':window,'threshold_kg':threshold,'model_version':str(version),
                'persistence_mae_kg':scores['persistence']['mae_kg'],'persistence_count':scores['persistence']['count'],
                'seasonal_24h_mae_kg':scores['seasonal_24h']['mae_kg'],'seasonal_24h_count':scores['seasonal_24h']['count'],
                'baseline_name':best,'baseline_mae_kg':baseline,'baseline_count':support,
                'skill_score':skill,'relative_gain_pct':skill*100 if skill is not None else None,
                'minimum_relative_gain_pct':MINIMUM_BASELINE_GAIN*100,
                'baseline_status':comparison,'baseline_reason':reason}

    def alert(self,category,level,message,details=None):
        with self.db() as c: self._alert(c,category,level,message,details or {})

    def _alert(self,c,category,level,message,details):
        now=utcnow()
        c.execute('INSERT INTO alerts(created_at,category,level,message,details) VALUES (?,?,?,?,?)',
                  (now,category,level,message,json.dumps(details,ensure_ascii=False)))

    def alerts(self,limit=100):
        with self.db() as c: rows=c.execute('SELECT * FROM alerts ORDER BY id DESC LIMIT ?',(limit,)).fetchall()
        return [{**dict(r),'details':json.loads(r['details'])} for r in rows]

    def log(self,level,message,**details):
        with self.lock:
            with (self.path/'aiops.log').open('a') as f:
                f.write(f'{utcnow()} [{level}] {message} {json.dumps(details,ensure_ascii=False)}\n')
