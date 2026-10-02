"""Persisted temperature monitoring and real retraining of the served ensemble.

The temperature thresholds are an explicit engineering demonstration policy.
They do not diagnose colony health or prove a reduction in forecast error.
The LSTM and constrained ensemble blend are fitted on separate chronological
partitions; the TiRex-2 checkpoint stays frozen. One immutable bundle is shared.
"""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
from threading import RLock
import uuid

import numpy as np
import pandas as pd

from .horizon_data import build_windows, encode_context
from .horizon_models import REGISTRY_ID
from .store import digest, utcnow

THRESHOLDS = {'mean_shift_c': 4., 'standardized_shift': 1.5,
              'reference_hours': 168, 'current_hours': 24}
STEP_KEYS = ('collect', 'monitor', 'detect', 'trigger', 'train', 'register', 'deploy')


def _link_or_copy(source, destination):
    # The frozen TiRex checkpoint is immutable and hash-checked; versions share it.
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}-{uuid.uuid4().hex}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def reference_for(rows, version, source):
    values = np.asarray([row['temperature_c'] for row in rows], dtype=float)
    return {'reference_id': digest(rows), 'count': len(rows), 'mean_c': float(values.mean()),
            'std_c': float(values.std()), 'start': rows[0]['timestamp'], 'end': rows[-1]['timestamp'],
            'model_version': str(version), 'source': source}


def normal_tail(rows):
    result = []
    for row in rows:
        if row['event'] != 'normal':
            result = []
        elif result and pd.Timestamp(row['timestamp'])-pd.Timestamp(result[-1]['timestamp']) != pd.Timedelta(hours=1):
            result = [row]
        else:
            result.append(row)
    return result


def temperature_check(rows, reference):
    current = [r for r in normal_tail(rows) if pd.Timestamp(r['timestamp']) > pd.Timestamp(reference['end'])][-24:]
    check = {'status': 'waiting', 'checked_at': utcnow(), 'snapshot_id': digest(rows),
             'thresholds': deepcopy(THRESHOLDS), 'reference_mean_c': reference['mean_c'],
             'available_hours': len(current)}
    if len(current) < 24:
        check['reason'] = 'requires_24_new_normal_hours_after_reference'
    else:
        mean = float(np.mean([r['temperature_c'] for r in current]))
        shift = mean-reference['mean_c']
        standardized = abs(shift)/max(reference['std_c'], 1.)
        drift = abs(shift) >= THRESHOLDS['mean_shift_c'] and standardized >= THRESHOLDS['standardized_shift']
        check.update(status='drift' if drift else 'stable', current_mean_c=mean, mean_shift_c=shift,
                     standardized_shift=standardized, current_start=current[0]['timestamp'],
                     current_end=current[-1]['timestamp'], reference_std_floor_c=1.)
    return check


def prepare_training(rows, after, context_hours, horizon_hours):
    """Separate LSTM labels, blend calibration and later held-out evaluation."""
    frame = pd.DataFrame(normal_tail(rows))
    if frame.empty:
        raise ValueError('No continuous normal observations for training')
    frame['timestamp'] = pd.to_datetime(frame.timestamp, utc=True)
    fresh = frame.iloc[context_hours:]
    fresh = fresh[fresh.timestamp > pd.Timestamp(after)]
    if len(fresh) < 24 or len(frame) < context_hours+24:
        raise ValueError(f'Need {context_hours} context hours and at least 24 new normal labels')
    held_out = max(6, min(24, len(fresh)//5))
    train_end = fresh.timestamp.iloc[-2*held_out-1]
    calibration_end = fresh.timestamp.iloc[-held_out-1]
    start, end = fresh.timestamp.iloc[0], fresh.timestamp.iloc[-1]
    train, _ = build_windows(frame, context_hours, horizon_hours, start, train_end,
                             stride=1, clean_targets=True, allow_partial_targets=True)
    calibration, _ = build_windows(frame, context_hours, 1, train_end+pd.Timedelta(hours=1), calibration_end, stride=1)
    validation, _ = build_windows(frame, context_hours, 1, calibration_end+pd.Timedelta(hours=1), end, stride=1)
    if len(train) < 4 or not calibration or not validation:
        raise ValueError('Insufficient observed training, calibration or held-out validation windows')
    return {'train': train, 'calibration': calibration, 'validation': validation, 'train_end': train_end.isoformat(),
            'new_observations': len(fresh), 'validation_start': validation[0].target_start.isoformat(),
            'validation_end': end.isoformat(), 'calibration_start': calibration[0].target_start.isoformat(),
            'calibration_end': calibration_end.isoformat()}


def fit_ensemble_weights(lstm, tirex, targets, parent_weights):
    """Fit a positive two-component blend using calibration targets only."""
    lstm, tirex, targets = (np.asarray(values, dtype=float) for values in (lstm, tirex, targets))
    if (targets.ndim != 1 or not len(targets) or lstm.shape != targets.shape or tirex.shape != targets.shape
            or not all(np.isfinite(values).all() for values in (lstm, tirex, targets))):
        raise ValueError('Finite aligned calibration predictions and targets are required')
    candidates = [index/20 for index in range(1,20)]
    parent_weight = float(parent_weights['lstm'])
    if .05 <= parent_weight <= .95:
        candidates.append(parent_weight)
    chosen = min(candidates, key=lambda weight: (
        round(float(np.abs(weight*lstm+(1-weight)*tirex-targets).mean()),12),
        abs(weight-parent_weight), weight))
    return {'lstm': chosen, 'tirex2': round(1-chosen,12)}


def training_cutoff(rows, parent, workspace_id, reference, manual=False, *, practice_replay=False):
    clean = normal_tail(rows)
    context = int(parent['context_hours'])
    if len(clean) < context+24:
        raise ValueError(f'Need {context} context hours and at least 24 new normal labels')
    if practice_replay:
        from .temperature_practice import validate_practice_recovery
        validate_practice_recovery(rows,rows)
        # Preserve stored values, including the Store's accepted floating-point
        # roundoff, after validating every row against the fixed source files.
        baseline = rows[168:336]
        if workspace_id != rows[0]['hive_id'] or reference['reference_id'] != digest(baseline):
            raise ValueError('Explicit practice replay requires its canonical normal baseline')
        return baseline[-1]['timestamp']
    cutoff = clean[context-1]['timestamp'] if manual else reference['end']
    previous = parent.get('temperature_training', {}).get('consumed_through_by_hive', {}).get(workspace_id)
    return max((cutoff, previous), key=pd.Timestamp) if previous else cutoff


class TemperatureDriftCoordinator:
    def __init__(self, runtime, provider):
        self.runtime = Path(runtime)/'temperature_drift'
        self.runtime.mkdir(parents=True, exist_ok=True)
        self.provider = provider
        self.lock = RLock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='temperature-retrain')
        self.path = self.runtime/'state.json'
        self.state = json.loads(self.path.read_text()) if self.path.exists() else {'workspaces': {}, 'jobs': []}
        for job in self.state['jobs']:
            if job['status'] in ('running', 'queued'):
                job.update(status='interrupted', error='Server restarted before job completion', completed_at=utcnow())
                for step in job.get('steps', []):
                    if step['status'] == 'running':
                        step.update(status='interrupted', error=job['error'], completed_at=job['completed_at'])
        if self.path.exists():
            self._save()

    def _save(self):
        atomic_json(self.path, self.state)

    def _active(self):
        catalog = self.provider.catalog()
        choose = getattr(self.provider, 'active_version', None)
        version = str(choose() if callable(choose) else catalog.get('default_version'))
        item = next((v for v in catalog['versions'] if str(v['version']) == version), {})
        return {'registry_id': REGISTRY_ID, **deepcopy(item), 'version': version}

    def status(self, workspace_id=None):
        with self.lock:
            workspace = self.state['workspaces'].get(str(workspace_id), {})
            jobs = [j for j in self.state['jobs'] if workspace_id is None or j['workspace_id'] == str(workspace_id)]
            return deepcopy({'workspace_id': workspace_id, 'enabled': True, 'active_model': self._active(),
                'reference': workspace.get('reference'), 'check': workspace.get('check', {'status': 'uninitialized'}),
                'job': jobs[-1] if jobs else None, 'jobs': list(reversed(jobs[-10:])),
                'policy': {'name': 'temperature_mean_shift_demo_v1', 'thresholds': THRESHOLDS,
                           'threshold_role': 'engineering_demonstration_not_biological_threshold',
                           'quality_metrics_role': 'descriptive_only', 'model_scope': 'shared',
                           'updated_component': 'lstm', 'trained_components': ['lstm','ensemble_weights'],
                           'frozen_component': 'tirex2', 'weight_policy': 'chronological_calibration_positive_blend',
                           'weights': self._active().get('weights'), 'reset_reference_hours': 24}})

    def request_training(self, service, workspace_id):
        """Queue one real update per immutable observation snapshot."""
        rows = service.store.observations()
        workspace_id = str(workspace_id)
        if not rows or any(row['hive_id'] != workspace_id for row in rows):
            raise ValueError('Retraining observations must belong to the requested hive')
        snapshot = digest(rows)
        with self.lock:
            prior = next((job for job in reversed(self.state['jobs'])
                          if job['workspace_id']==workspace_id and job['snapshot_id']==snapshot
                          and job['status'] not in ('failed','interrupted')), None)
            if prior:
                return deepcopy({'status':'already_evaluated','job':prior,
                                 'workspace_id':workspace_id,'snapshot_id':snapshot})
            active = self._active()
            parent, _ = self.provider._manifest(active['version'])
            if parent is None:
                raise RuntimeError('No available shared ensemble for retraining')
            after = training_cutoff(rows,parent,workspace_id,None,manual=True)
            prepare_training(rows,after,int(parent['context_hours']),int(parent['max_horizon_hours']))
            workspace = self.state['workspaces'].setdefault(workspace_id,{})
            reference = workspace.get('reference')
            if reference is None:
                recent = normal_tail(rows)[-168:]
                reference = reference_for(recent,active['version'],'manual_request_observed_reference')
                workspace['reference'] = reference
                self._persist_reference(workspace_id,reference,recent)
            check = {'status':'manual_request','snapshot_id':snapshot,'checked_at':utcnow(),
                     'training_after':after}
            job = self._queue_training(service,workspace_id,rows,reference,active,check,'manual_retraining')
            return deepcopy({'status':'queued','job':job,'workspace_id':workspace_id,'snapshot_id':snapshot})

    def _queue_training(self, service, workspace_id, rows, reference, active, check, reason, *, practice_replay=None):
        now = utcnow()
        attempt = digest({'workspace_id':workspace_id,'snapshot_id':check['snapshot_id'],
                          'reference_id':reference['reference_id']})
        job = {'job_id':uuid.uuid4().hex,'attempt_id':attempt,'workspace_id':workspace_id,
               'snapshot_id':check['snapshot_id'],'reference_id':reference['reference_id'],
               'status':'queued','stage':'trigger','requested_at':now,
               'parent_version':active['version'],'registry_id':REGISTRY_ID,'model_scope':'shared',
               'reason':reason,'check':deepcopy(check),
               'steps':[{'key':key,'status':'completed' if index<4 else 'pending',
                         **({'completed_at':now} if index<4 else {})} for index,key in enumerate(STEP_KEYS)]}
        if reason=='manual_retraining':
            for step in job['steps'][1:3]:
                step.update(status='skipped',reason='manual_request')
        if practice_replay is not None:
            job['practice_replay'] = deepcopy(practice_replay)
            job['practice_request_ids'] = [practice_replay['request_id']]
        self.state['jobs'].append(job)
        try:
            atomic_json(self.runtime/'snapshots'/f'{job["snapshot_id"]}.json',
                        {'rows':rows,'sources':service.store.sources(),'workspace_id':workspace_id,
                         'reference':reference,'check':check})
            check['trigger_status'] = 'queued'
            self._save()
            self.executor.submit(self._run,job['job_id'],deepcopy(rows),deepcopy(reference))
        except Exception as exc:
            check['trigger_status'] = 'failed'
            job['steps'][3].update(status='failed',error=str(exc),completed_at=utcnow())
            self._record_failure(job,exc)
            raise
        return job

    def _record_failure(self, job, error):
        """Keep in-memory truth even when the failure cannot itself be saved."""
        job.update(status='failed',error=str(error),completed_at=utcnow())
        for step in job['steps']:
            if step['status']=='running':
                step.update(status='failed',error=str(error),completed_at=job['completed_at'])
        try:
            self._save()
        except Exception as exc:
            job.setdefault('reporting_warnings',[]).append(f'Failed to persist failed retraining job: {exc}')

    def retraining_status(self):
        """Read the shared worker and queue without observing or changing data."""
        with self.lock:
            jobs = self.state['jobs']
            return deepcopy({'enabled': True, 'active_model': self._active(),
                'running_job': next((job for job in jobs if job['status']=='running'), None),
                'queued_jobs': [job for job in jobs if job['status']=='queued'],
                'latest_job': jobs[-1] if jobs else None,
                'jobs': list(reversed(jobs[-10:])), 'server_time': utcnow()})

    def retraining_job(self, job_id):
        """Read one durable execution, including jobs outside the recent list."""
        with self.lock:
            job = next((job for job in self.state['jobs'] if job['job_id']==job_id),None)
            if job is None:
                raise KeyError('Unknown retraining job')
            return deepcopy(job)

    def observe_practice(self, service, workspace_id, *, request_id, inserted_observations=0):
        """An explicit built-in run reuses fixed data and performs real training.

        Import identity makes retries idempotent. A new import after completion
        is a new run; a concurrent request follows the already running job.
        Operational monitoring references are not rewound for a replay.
        """
        from .temperature_practice import heatwave_rows, practice_rows, validate_practice_recovery
        if not isinstance(request_id,str) or not request_id or len(request_id)>128:
            raise ValueError('Explicit practice replay requires an import request ID')
        observed = service.store.observations()
        expected = heatwave_rows()
        validate_practice_recovery(expected, observed)
        rows = [row for row in observed if row['timestamp']<=expected[-1]['timestamp']]
        validate_practice_recovery(rows,observed)
        workspace_id = str(workspace_id)
        if workspace_id!=rows[0]['hive_id']:
            raise ValueError('Explicit practice workspace does not match its observations')
        snapshot = digest(rows)
        with self.lock:
            workspace = self.state['workspaces'].setdefault(workspace_id, {})
            jobs = [job for job in self.state['jobs'] if job['workspace_id']==workspace_id]
            def result(job,status,requested=False):
                return {**self.status(workspace_id),'job':deepcopy(job),
                    'retraining_requested':requested,'retraining_job_id':job['job_id'],
                    'retraining_status':status,'retraining_request_id':request_id}
            prior = next((job for job in reversed(jobs) if request_id in job.get('practice_request_ids',[])),None)
            if prior is not None:
                return result(prior,'already_requested')
            running = next((job for job in reversed(jobs) if job['snapshot_id']==snapshot
                            and job['status'] in ('queued','running')),None)
            if running is not None:
                running.setdefault('practice_request_ids',[]).append(request_id)
                self._save()
                return result(running,'already_running')
            active = self._active()
            baseline = rows[168:336]
            reference = reference_for(baseline,active['version'],'built_in_practice_normal_baseline')
            self._persist_reference(workspace_id,reference,baseline)
            previous = workspace.get('reference')
            hot = practice_rows('02_heatwave.csv')
            initial_hot = previous and previous.get('source')=='initial_recent_168_normal_hours' and previous.get('reference_id')==digest(hot)
            if previous is None or (not jobs and initial_hot):
                workspace['reference'] = reference
            check = temperature_check(rows,reference)
            check.update(trigger='explicit_practice_request',training_after=reference['end'])
            replay = {'request_id':request_id,'data_kind':'synthetic','fixed_observations':len(rows),
                'inserted_observations':int(inserted_observations),
                'reused_observations':max(0,len(rows)-int(inserted_observations)),
                'data_policy':'explicit_reuse_of_fixed_builtin_observations',
                'reference_start':reference['start'],'reference_end':reference['end']}
            job = self._queue_training(service,workspace_id,rows,reference,active,check,
                                       'explicit_temperature_practice',practice_replay=replay)
            return result(job,'queued',True)

    def observe(self, service, workspace_id):
        rows = service.store.observations()
        clean = normal_tail(rows)
        workspace_id = str(workspace_id)
        with self.lock:
            requested_job = None
            workspace = self.state['workspaces'].setdefault(workspace_id, {})
            reference = workspace.get('reference')
            active = self._active()
            if reference is None:
                if len(clean) >= 168:
                    reference = reference_for(clean[-168:], active['version'], 'initial_recent_168_normal_hours')
                    workspace['reference'] = reference
                    self._persist_reference(workspace_id, reference, clean[-168:])
                    check = {'status': 'reference_initialized', 'checked_at': utcnow(), 'snapshot_id': digest(rows)}
                else:
                    check = {'status': 'waiting', 'reason': 'requires_168_continuous_normal_hours',
                             'available_hours': len(clean), 'checked_at': utcnow(), 'snapshot_id': digest(rows)}
                workspace['check'] = check
                self._save()
                return {**self.status(workspace_id),'retraining_requested':False,'retraining_job_id':None}
            check = temperature_check(rows, reference)
            workspace['check'] = check
            if check['status'] == 'drift':
                attempt = digest({'workspace_id': workspace_id, 'snapshot_id': check['snapshot_id'],
                                  'reference_id': reference['reference_id']})
                duplicate = any(j['attempt_id'] == attempt for j in self.state['jobs'])
                if duplicate:
                    check['trigger_status'] = 'already_evaluated'
                else:
                    requested_job = self._queue_training(service,workspace_id,rows,reference,active,check,'temperature_distribution_drift')
            self._save()
            return {**self.status(workspace_id),'retraining_requested':requested_job is not None,
                    'retraining_job_id':requested_job['job_id'] if requested_job else None}

    def _persist_reference(self, workspace_id, reference, rows):
        path = self.runtime/'references'/digest(workspace_id)/f'{reference["reference_id"]}.json'
        if not path.exists():
            atomic_json(path, {'reference': reference, 'rows': rows})

    def _run(self, job_id, rows, reference):
        with self.lock:
            job = next(j for j in self.state['jobs'] if j['job_id'] == job_id)
        def progress(stage, detail=None):
            with self.lock:
                if stage == 'training_progress':
                    job['training_progress'] = {**deepcopy(detail), 'updated_at': utcnow()}
                    self._save()
                    return
                for step in job['steps']:
                    if step['key'] == stage:
                        step.update(status='completed', completed_at=utcnow(), detail=detail or {})
                index = STEP_KEYS.index(stage)
                if index+1 < len(STEP_KEYS):
                    job['stage'] = STEP_KEYS[index+1]
                    job['steps'][index+1].update(status='running', started_at=utcnow())
                self._save()
        try:
            with self.lock:
                latest_reference = self.state['workspaces'][job['workspace_id']]['reference']
                if job['reason'] not in ('manual_retraining','explicit_temperature_practice') and latest_reference['reference_id'] != reference['reference_id']:
                    check = temperature_check(rows, latest_reference)
                    self.state['workspaces'][job['workspace_id']]['check'] = check
                    if check['status'] != 'drift':
                        job.update(status='skipped', stage='monitor', completed_at=utcnow(),
                                   reason='no_temperature_drift_after_previous_deployment', recheck=check)
                        for step in job['steps'][4:]: step['status'] = 'skipped'
                        self._save()
                        return
                    reference = deepcopy(latest_reference)
                    job.update(reference_id=reference['reference_id'], recheck=check)
                job.update(status='running', started_at=utcnow(), stage='train')
                job['steps'][4].update(status='running', started_at=utcnow())
                self._save()
            result = self._train_bundle(deepcopy(job), rows, reference, progress)
        except Exception as exc:
            with self.lock:
                self._record_failure(job,exc)
            return
        # Publication has already succeeded. A later reporting/reference write
        # cannot turn the deployed model back into a failed training attempt.
        with self.lock:
            job.update(result,status='completed',stage='deploy',completed_at=utcnow())
            workspace = self.state['workspaces'][job['workspace_id']]
            previous_workspace = deepcopy(workspace)
            try:
                existing_reference = workspace.get('reference')
                preserve_newer = (job['reason']=='explicit_temperature_practice' and existing_reference
                    and pd.Timestamp(existing_reference['end'])>pd.Timestamp(rows[-1]['timestamp']))
                if preserve_newer:
                    job['reference_update'] = {'status':'preserved_newer_operational_reference',
                        'reference_id':existing_reference['reference_id'],'end':existing_reference['end']}
                else:
                    latest = normal_tail(rows)[-24:]
                    new_reference = reference_for(latest, result['new_version'], 'post_deployment_recent_24_normal_hours')
                    self._persist_reference(job['workspace_id'], new_reference, latest)
                    workspace['reference'] = new_reference
                    workspace['check'] = {'status':'reference_updated','checked_at':utcnow(),'snapshot_id':digest(rows),
                        'thresholds':deepcopy(THRESHOLDS),'reference_mean_c':new_reference['mean_c'],
                        'current_mean_c':new_reference['mean_c'],'mean_shift_c':0.,'standardized_shift':0.,
                        'current_start':new_reference['start'],'current_end':new_reference['end'],
                        'available_hours':24,'reference_std_floor_c':1.}
                self._save()
            except Exception as exc:
                workspace.clear()
                workspace.update(previous_workspace)
                job.setdefault('reporting_warnings',[]).append(f'Reference update failed after deployment: {exc}')
                try:
                    self._save()
                except Exception as save_error:
                    job['reporting_warnings'].append(f'Failed to persist completed retraining job: {save_error}')

    def _train_bundle(self, job, rows, reference, progress):
        import tensorflow as tf
        from mlflow.tracking import MlflowClient
        tf.keras.utils.set_random_seed(42)
        active = self._active()
        parent_version = active['version']
        parent, parent_folder = self.provider._manifest(parent_version)
        if (parent is None or set(parent['components']) != {'lstm', 'tirex2'}
                or set(parent['weights']) != {'lstm','tirex2'}
                or any(not np.isfinite(value) or value <= 0 for value in parent['weights'].values())
                or not np.isclose(sum(parent['weights'].values()),1.)):
            raise ValueError('Temperature retraining requires the active TiRex-2 + LSTM ensemble')
        context, horizon = int(parent['context_hours']), int(parent['max_horizon_hours'])
        practice_replay = job['reason']=='explicit_temperature_practice'
        after = training_cutoff(rows,parent,job['workspace_id'],reference,job['reason']=='manual_retraining',
                                practice_replay=practice_replay)
        prepared = prepare_training(rows, after, context, horizon)
        train, calibration, validation = prepared['train'], prepared['calibration'], prepared['validation']
        old_weights = deepcopy(parent['weights'])
        trained_components, frozen_components = ['lstm','ensemble_weights'], ['tirex2']
        tirex_folder = parent_folder/parent['files']['tirex2']
        tirex_before_sha256 = hashlib.sha256((tirex_folder/'model.ckpt').read_bytes()).hexdigest()
        total_epochs = 8
        progress_base = {'total_epochs':total_epochs,'training_windows':len(train),
                         'old_weights':old_weights,'trained_components':trained_components,
                         'frozen_components':frozen_components}
        progress('training_progress', {**progress_base,'phase':'lstm_training','completed_epochs':0,'last_loss':None})
        raw_x = np.stack([sample.context for sample in train])
        x = encode_context(raw_x, parent['scaler'])
        y = np.full((len(train), horizon), np.nan, dtype=np.float32)
        for i,sample in enumerate(train):
            y[i,:len(sample.target)] = (sample.target-sample.context[-1,0])/parent['scaler']['weight_scale']
        model = tf.keras.models.load_model(parent_folder/parent['files']['lstm'], compile=False)
        before_weights = hashlib.sha256(b''.join(w.tobytes() for w in model.get_weights())).hexdigest()
        def observed_mae(actual, predicted):
            observed = tf.math.is_finite(actual)
            safe = tf.where(observed, actual, predicted)
            error = tf.where(observed, tf.abs(safe-predicted), tf.zeros_like(predicted))
            return tf.math.divide_no_nan(tf.reduce_sum(error,axis=-1), tf.reduce_sum(tf.cast(observed,predicted.dtype),axis=-1))
        model.compile(optimizer=tf.keras.optimizers.Adam(.001), loss=observed_mae)
        losses = []
        rng = np.random.default_rng(42)
        for epoch in range(total_epochs):
            order = rng.permutation(len(x)); model.reset_metrics()
            for start in range(0, len(x), 32):
                selected = order[start:start+32]
                loss = model.train_on_batch(x[selected], y[selected])
            epoch_loss = float(np.asarray(loss))
            if not np.isfinite(epoch_loss):
                raise RuntimeError('Training produced a nonfinite LSTM loss')
            losses.append(epoch_loss)
            progress('training_progress', {**progress_base,'phase':'lstm_training',
                     'completed_epochs':epoch+1,'last_loss':epoch_loss})
        after_weights = hashlib.sha256(b''.join(w.tobytes() for w in model.get_weights())).hexdigest()
        if before_weights == after_weights or not np.isfinite(losses).all():
            raise RuntimeError('Training did not produce finite changed LSTM weights')
        progress('training_progress', {**progress_base,'phase':'ensemble_calibration',
                 'completed_epochs':total_epochs,'last_loss':losses[-1]})
        cal_x = np.stack([s.context for s in calibration])
        cal_target = np.asarray([s.target[0] for s in calibration])
        val_x = np.stack([s.context for s in validation])
        target = np.asarray([s.target[0] for s in validation])
        scoring_x = np.concatenate((cal_x,val_x))
        outputs = np.asarray(model(encode_context(scoring_x,parent['scaler']),training=False))
        if outputs.shape != (len(scoring_x), horizon) or not np.isfinite(outputs).all():
            raise RuntimeError('Retrained LSTM must return a finite complete horizon')
        scoring_lstm = outputs[:,0]*parent['scaler']['weight_scale']+scoring_x[:,-1,0]
        # Each TiRex input contains only observations available at its origin.
        # Calibration targets alone choose the blend; later targets only score it.
        from .tirex_adapter import TirexAdapter
        scoring_foundation = TirexAdapter(tirex_folder).predict(scoring_x,1)[:,0]
        if scoring_foundation.shape != (len(scoring_x),) or not np.isfinite(scoring_foundation).all():
            raise RuntimeError('TiRex calibration and evaluation predictions must be finite and aligned')
        cal_lstm, val_lstm = scoring_lstm[:len(calibration)], scoring_lstm[len(calibration):]
        cal_foundation, foundation = scoring_foundation[:len(calibration)], scoring_foundation[len(calibration):]
        new_weights = fit_ensemble_weights(cal_lstm,cal_foundation,cal_target,old_weights)
        cal_combined = new_weights['lstm']*cal_lstm+new_weights['tirex2']*cal_foundation
        calibration_evidence = {'selection':'minimum_calibration_mae','horizon_hours':1,
            'candidate_lstm_min':.05,'candidate_lstm_max':.95,'candidate_step':.05,
            'parent_weight_included':True,'tie_break':'closest_parent_weight_then_lower_lstm',
            'calibration_start':prepared['calibration_start'],'calibration_end':prepared['calibration_end'],
            'calibration_samples':len(calibration),'old_weights':old_weights,'new_weights':new_weights,
            'calibration_mae_kg':float(np.abs(cal_combined-cal_target).mean()),
            'rows':[{'origin':sample.origin.isoformat(),'target_timestamp':sample.target_start.isoformat(),
                     'actual_kg':float(cal_target[index]),'lstm_kg':float(cal_lstm[index]),
                     'tirex2_kg':float(cal_foundation[index]),'ensemble_kg':float(cal_combined[index])}
                    for index,sample in enumerate(calibration)]}
        progress('training_progress', {**progress_base,'phase':'held_out_evaluation','new_weights':new_weights,
                 'completed_epochs':total_epochs,'last_loss':losses[-1],
                 'calibration_samples':len(calibration),'validation_samples':len(validation)})
        parent_lstm = tf.keras.models.load_model(parent_folder/parent['files']['lstm'],compile=False)
        old_lstm = np.asarray(parent_lstm(encode_context(val_x,parent['scaler']),training=False))[:,0]*parent['scaler']['weight_scale']+val_x[:,-1,0]
        old = old_weights['tirex2']*foundation+old_weights['lstm']*old_lstm
        new = new_weights['tirex2']*foundation+new_weights['lstm']*val_lstm
        metrics = {'previous_ensemble_mae_kg': float(np.abs(old-target).mean()),
                   'new_ensemble_mae_kg': float(np.abs(new-target).mean()),
                   'lstm_mae_kg': float(np.abs(val_lstm-target).mean()),
                   'calibration_mae_kg':calibration_evidence['calibration_mae_kg'],
                   'calibration_samples':len(calibration),
                   'validation_samples': len(validation), 'training_windows': len(train), 'epochs_completed': total_epochs}
        if not np.isfinite(list(metrics.values())).all():
            raise RuntimeError('Retrained ensemble evaluation contains nonfinite predictions')
        evidence = {'role':'descriptive_only', 'horizon_hours':1, 'evaluation':'chronological_holdout_after_blend_calibration',
                    'old_weights':old_weights,'new_weights':new_weights,
                    **metrics, 'validation_start':prepared['validation_start'], 'validation_end':prepared['validation_end'],
                    'rows':[{'origin':s.origin.isoformat(),'target_timestamp':s.target_start.isoformat(),
                             'actual_kg':float(target[i]),'previous_kg':float(old[i]),'new_kg':float(new[i])}
                            for i,s in enumerate(validation)]}
        if practice_replay:
            evidence.update(evaluation='chronological_practice_replay_holdout_after_blend_calibration',
                data_reused=True,independent_future_evaluation=False)
        for index,row in enumerate(evidence['rows']):
            row.update(previous_lstm_kg=float(old_lstm[index]),lstm_kg=float(val_lstm[index]),
                       tirex2_kg=float(foundation[index]))
        component_evidence = {'old_weights':old_weights,'new_weights':new_weights,
            'trained_components':trained_components,'frozen_components':frozen_components,
            'before_weights_sha256':before_weights,'after_weights_sha256':after_weights,
            'tirex_before_sha256':tirex_before_sha256,
            'ensemble_before_sha256':digest({'lstm':before_weights,'tirex2':tirex_before_sha256,'weights':old_weights}),
            'ensemble_after_sha256':digest({'lstm':after_weights,'tirex2':tirex_before_sha256,'weights':new_weights})}
        progress('train', {**metrics,**component_evidence,
                          'calibration_start':prepared['calibration_start'],'calibration_end':prepared['calibration_end'],
                          'validation_start':prepared['validation_start'],'validation_end':prepared['validation_end']})
        root = self.provider.artifact_root
        index = json.loads((root/'index.json').read_text())
        # Failed publication may leave a valid, registered but unindexed bundle.
        # Preserve that evidence and allocate beyond every reserved directory.
        reserved = [int(v['version']) for v in index['versions']]
        reserved.extend(int(p.name) for p in root.iterdir() if p.is_dir() and p.name.isdigit())
        version = str(max(reserved)+1)
        folder = root/version
        if folder.exists():
            raise FileExistsError(f'Immutable bundle already exists: {version}')
        pending = root/f'.pending-temperature-{job["job_id"]}'
        pending.mkdir()
        tracking = self.runtime/'mlflow.db'
        client = MlflowClient(tracking_uri=f'sqlite:///{tracking}',registry_uri=f'sqlite:///{tracking}')
        experiment = client.get_experiment_by_name('BeeOPS_Horizon_Temperature')
        experiment_id = experiment.experiment_id if experiment else client.create_experiment(
            'BeeOPS_Horizon_Temperature', artifact_location=(self.runtime/'artifacts').resolve().as_uri())
        from mlflow.exceptions import MlflowException
        try: client.get_registered_model(REGISTRY_ID)
        except MlflowException as exc:
            if exc.error_code != 'RESOURCE_DOES_NOT_EXIST': raise
            client.create_registered_model(REGISTRY_ID)
        run = client.create_run(experiment_id,tags={'mlflow.runName':f'ensemble-{job["reason"]}-{version}',
            'model_scope':'shared','updated_component':'lstm_and_ensemble_weights','frozen_component':'tirex2','bundle_version':version})
        run_id = run.info.run_id
        promoted = False
        try:
            model.save(pending/'lstm.keras')
            shutil.copytree(parent_folder/parent['files']['tirex2'], pending/'tirex2', copy_function=_link_or_copy)
            snapshot = self.runtime/'snapshots'/f'{job["snapshot_id"]}.json'
            shutil.copy2(snapshot, pending/'training_snapshot.json')
            atomic_json(pending/'evaluation.json', evidence)
            atomic_json(pending/'ensemble_calibration.json', calibration_evidence)
            hashes = {str(p.relative_to(pending)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in pending.rglob('*') if p.is_file()}
            tirex_after_sha256 = hashes['tirex2/model.ckpt']
            if tirex_after_sha256 != tirex_before_sha256:
                raise RuntimeError('Frozen TiRex checkpoint changed during ensemble retraining')
            consumed = deepcopy(parent.get('temperature_training', {}).get('consumed_through_by_hive', {}))
            previous_consumed = consumed.get(job['workspace_id'])
            consumed[job['workspace_id']] = max((previous_consumed,rows[-1]['timestamp']),key=pd.Timestamp) if previous_consumed else rows[-1]['timestamp']
            training = {'reason':job['reason'],'job_id':job['job_id'],
                        'snapshot_id':job['snapshot_id'],'reference_id':reference['reference_id'],
                        'source_workspace_id':job['workspace_id'],'training_start':train[0].target_start.isoformat(),
                        'training_end':prepared['train_end'],'new_observations':0 if practice_replay else prepared['new_observations'],
                        'maximum_observed_training_horizon':max(len(s.target) for s in train),
                        'validation_start':prepared['validation_start'],'validation_end':prepared['validation_end'],
                        'calibration_start':prepared['calibration_start'],'calibration_end':prepared['calibration_end'],
                        'calibration_samples':len(calibration),'validation_samples':len(validation),
                        'consumed_through_by_hive':consumed,'epochs_completed':total_epochs,'loss_scaled':losses,
                        **component_evidence,'tirex_after_sha256':tirex_after_sha256,
                        'metrics':metrics,'promotion_policy':'finite_artifacts_and_reload_no_quality_threshold',
                        'scaler_policy':'frozen_parent_scaler','tirex_policy':'frozen_checkpoint',
                        'blend_policy':'one_hour_chronological_calibration_positive_weights',
                        'evaluation_role':'descriptive_only'}
            if practice_replay:
                training.update(practice_replay=deepcopy(job['practice_replay']),
                    replayed_target_observations=prepared['new_observations'],
                    evaluation_role='descriptive_fixed_practice_replay')
            manifest = {**deepcopy(parent),'version':version,'run_id':run_id,'parent_version':parent_version,
                        'weights':new_weights,
                        'created_at':utcnow(),'files':{'lstm':'lstm.keras','tirex2':'tirex2'},'file_sha256':hashes,
                        'lstm_origin':('Warm-started actual parent LSTM; retrained on explicitly reused fixed synthetic practice labels'
                                       if practice_replay else 'Warm-started actual parent LSTM; trained on newly observed labels'),
                        'temperature_training':training,'provenance_snapshot':job['snapshot_id'],
                        'training_hive_ids':sorted(set(parent.get('training_hive_ids',[])+[job['workspace_id']])),
                        'interval_radii_kg':[], 'validation':{'status':'insufficient','calibrated':False,
                            'role':'informational','horizons':{},'noise_kg':parent.get('validation',{}).get('noise_kg'),
                            'reasons':['new_bundle_intervals_not_calibrated'],'temperature_retraining':evidence},
                        'evaluation_protocol':'chronological_training_blend_calibration_final_one_hour_holdout',
                        'bundle_available_after':rows[-1]['timestamp'], 'benchmark_file':'evaluation.json',
                        'limitations':['LSTM parameters and ensemble blend were trained; the TiRex-2 checkpoint stays frozen.',
                            'Blend weights use separate one-hour calibration origins; later holdout targets are evaluation only.',
                            'One-hour held-out metrics describe this update, not independent pretraining data or long-horizon calibration.',
                            'Temperature thresholds are engineering demonstration thresholds, not biological criteria.']}
            atomic_json(pending/'manifest.json',manifest)
            # Verify the actual serialized new LSTM before either registry or alias publication.
            reloaded = tf.keras.models.load_model(pending/'lstm.keras',compile=False)
            np.testing.assert_allclose(np.asarray(reloaded(encode_context(val_x[:2],parent['scaler']),training=False)),
                                       np.asarray(model(encode_context(val_x[:2],parent['scaler']),training=False)),rtol=1e-5,atol=1e-5)
            for key,value in {'parent_version':parent_version,'bundle_version':version,'snapshot_id':job['snapshot_id'],
                              'source_workspace_id':job['workspace_id'],'reason':job['reason'],
                              'updated_component':'lstm_and_ensemble_weights','frozen_component':'tirex2',
                              'quality_metrics_role':'descriptive_only',**component_evidence,
                              'tirex_after_sha256':tirex_after_sha256,
                              'training_start':training['training_start'],'training_end':training['training_end'],
                              'calibration_start':training['calibration_start'],'calibration_end':training['calibration_end'],
                              'validation_start':training['validation_start'],'validation_end':training['validation_end']}.items():
                client.log_param(run_id,key,json.dumps(value,sort_keys=True) if isinstance(value,(dict,list)) else value)
            for key,value in metrics.items(): client.log_metric(run_id,key,float(value))
            client.set_tag(run_id,'tirex2_checkpoint_sha256',tirex_after_sha256)
            for path in sorted(pending.rglob('*')):
                relative = path.relative_to(pending)
                if path.is_file() and relative.as_posix() != 'tirex2/model.ckpt':
                    client.log_artifact(run_id,str(path),artifact_path='/'.join(('bundle',*relative.parent.parts)))
            record = client.create_model_version(REGISTRY_ID,source=f'{run.info.artifact_uri}/bundle',run_id=run_id,
                         tags={'bundle_version':version,'parent_bundle_version':parent_version,'model_scope':'shared'})
            mlflow_version = str(record.version)
            # Metadata is added before publishing the immutable serving directory.
            manifest['mlflow'] = {'registry_name':REGISTRY_ID,'model_version':mlflow_version,
                                 'tracking_uri':f'sqlite:///{tracking}','run_id':run_id}
            atomic_json(pending/'manifest.json',manifest)
            client.log_artifact(run_id,str(pending/'manifest.json'),artifact_path='bundle')
            progress('register', {'version':version,'mlflow_model_version':mlflow_version,'run_id':run_id})
            pending.rename(folder)
            # The index is the authoritative deployment pointer; its replacement
            # affects every workspace at once. All fallible loading precedes it.
            if self._active()['version'] != parent_version:
                raise RuntimeError('Active shared bundle changed during training; candidate remains registered')
            index = json.loads((root/'index.json').read_text())
            index['versions'].append({'version':version,'manifest':f'{version}/manifest.json'})
            index['default_version'] = version
            registry = client.get_registered_model(REGISTRY_ID)
            old_alias = registry.aliases.get('champion')
            client.set_registered_model_alias(REGISTRY_ID,'champion',mlflow_version)
            try:
                atomic_json(root/'index.json',index)
            except Exception:
                if old_alias: client.set_registered_model_alias(REGISTRY_ID,'champion',old_alias)
                else: client.delete_registered_model_alias(REGISTRY_ID,'champion')
                raise
            promoted = True
            warnings = []
            try:
                client.set_tag(run_id,'promoted','true'); client.set_terminated(run_id,'FINISHED')
            except Exception as exc:
                warnings.append(str(exc))
            try:
                progress('deploy',{'version':version,'run_id':run_id,'scope':'all_workspaces'})
            except Exception as exc:
                warnings.append(str(exc))
            return {'parent_version':parent_version,'new_version':version,'version':version,'run_id':run_id,
                    'mlflow_model_version':mlflow_version,'registry_id':REGISTRY_ID,'promoted':True,
                    **component_evidence,'tirex_after_sha256':tirex_after_sha256,
                    'training':training,'calibration':calibration_evidence,'evaluation':evidence,'reporting_warnings':warnings}
        except Exception:
            if not promoted:
                try: client.set_terminated(run_id,'FAILED')
                except Exception: pass
            raise
        finally:
            if pending.exists(): shutil.rmtree(pending)

    def close(self):
        self.executor.shutdown(wait=True)
