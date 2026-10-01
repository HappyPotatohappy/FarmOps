"""Compare a real offline TiRex-2 candidate on the frozen existing origins.

This is retrospective regression evidence, not a newly sealed test. Existing
LSTM and Chronos outputs are immutable actual-inference evidence, never refit.
Only validation labels choose weights; no serving registry is mutated.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
import uuid
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.tirex_adapter import TirexAdapter
from app.horizon_data import load_real_sources


def choose_weights(records, names=('lstm','chronos2','tirex2')):
    from app.horizon_models import select_weights
    validation = [r for r in records if r['partition'] == 'validation']
    if not validation or 'lstm' not in names or any('lstm' not in r['predictions_kg'] for r in validation):
        raise ValueError('Actual lstm predictions on validation origins are required')
    counts = Counter(r['hive_id'] for r in validation)
    return select_weights({name: np.asarray([r['predictions_kg'][name] for r in validation]) for name in names},
                          np.asarray([r['actual_kg'] for r in validation]),
                          np.asarray([1 / counts[r['hive_id']] for r in validation]), minimum_lstm=.1)


def summarize(records, name):
    if not records:
        return {'origin_count': 0, 'mae_kg': None, 'macro_hive_mae_kg': None}
    errors = np.asarray([r['predictions_kg'][name]-r['actual_kg'] for r in records])
    hives = sorted(set(r['hive_id'] for r in records))
    per_hive = {h: float(np.mean([abs(r['predictions_kg'][name]-r['actual_kg']) for r in records if r['hive_id']==h])) for h in hives}
    return {'origin_count': len(records), 'mae_kg': float(np.abs(errors).mean()),
            'rmse_kg': float(np.sqrt(np.square(errors).mean())), 'bias_kg': float(errors.mean()),
            'macro_hive_mae_kg': float(np.mean(list(per_hive.values()))), 'per_hive_mae_kg': per_hive}


def export_bundle(baseline_folder, checkpoint, report, artifact_root, version='2'):
    """Publish copied immutable artifacts; previous interval estimates are invalid.

    Export only the validation-selected positive components. All hives share
    the same copy of the actual trained LSTM and TiRex checkpoint.
    """
    baseline_folder, checkpoint, artifact_root = map(Path, (baseline_folder, checkpoint, artifact_root))
    destination = artifact_root / str(version)
    if destination.exists():
        raise FileExistsError(f'Immutable model version already exists: {destination}')
    baseline = json.loads((baseline_folder/'manifest.json').read_text())
    weights = {k: v for k,v in report['weights'].items() if v > 0}
    if set(weights) != {'lstm','tirex2'} or weights['lstm'] < .1 or not np.isclose(sum(weights.values()),1):
        raise ValueError('This exporter requires an actual LSTM + TiRex-2 selected ensemble')
    validation = {'status':'insufficient','calibrated':False,'role':'informational',
                  'noise_kg':baseline.get('validation',{}).get('noise_kg'), 'horizons':{},
                  'interval_kind':'No newly calibrated intervals for this ensemble',
                  'protocol':report['evaluation_protocol']}
    for horizon in (1,24,72,168):
        rows=[r for r in report['origins'] if r['partition']=='test' and r['horizon_hours']==horizon]
        ends={}; independent=0
        for row in sorted(rows,key=lambda r:(r['hive_id'],r['target_start'])):
            if row['hive_id'] not in ends or row['target_start'] > ends[row['hive_id']]:
                ends[row['hive_id']]=row['target_end'];independent+=1
        validation['horizons'][str(horizon)] = {
            'status':'insufficient','calibrated':False,'origin_count':len(rows),'independent_windows':independent,
            'reasons':['new_ensemble_intervals_not_calibrated','existing_test_is_regression_evidence'],
            'validation_mae_kg':report['scores']['validation'].get(str(horizon),{}).get('operational',{}).get('upgrade_ensemble',{}).get('mae_kg'),
            'test_mae_kg':report['scores']['test'].get(str(horizon),{}).get('operational',{}).get('upgrade_ensemble',{}).get('mae_kg')}
    artifact_root.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.pending-tirex-',dir=artifact_root) as temporary:
        folder=Path(temporary)
        old_lstm=baseline_folder/baseline['files']['lstm']
        expected=baseline.get('file_sha256',{}).get(baseline['files']['lstm'])
        if expected and hashlib.sha256(old_lstm.read_bytes()).hexdigest()!=expected:
            raise ValueError('Original trained LSTM artifact hash mismatch')
        shutil.copy2(old_lstm,folder/'lstm.keras')
        shutil.copytree(checkpoint,folder/'tirex2')
        hashes={str(p.relative_to(folder)):hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.rglob('*') if p.is_file()}
        manifest={**baseline,'schema_version':1,'version':str(version),'run_id':uuid.uuid4().hex,
            'created_at':report['created_at'],'context_hours':report['context_hours'],'max_horizon_hours':168,
            'components':list(weights),'weights':weights,'files':{'lstm':'lstm.keras','tirex2':'tirex2'},
            'file_sha256':hashes,'validation':validation,'interval_radii_kg':[],
            'evaluation_protocol':report['evaluation_protocol'],'limitations':report['limitations'],
            'parent_version':str(baseline['version']),
            'tirex2':{'model_id':'NX-AI/TiRex-2','revision':report['model_revision'],'checkpoint_sha256':report['model_sha256'],
                      'device':'cpu','runtime_package':'tirex-2==0.3.0','worker_environment':'BEEOPS_TIREX_PYTHON'},
            'lstm_origin':'Copied immutable actual trained parent artifact; no retraining',
            'benchmark_file':'../../evidence/tirex2_benchmark_20261001.json',
            'baseline_report_sha256':report['baseline_report_sha256'],'provenance_snapshot':report['data_snapshot']}
        manifest.pop('chronos',None)
        (folder/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
        folder.rename(destination)
    index_path=artifact_root/'index.json'
    index=json.loads(index_path.read_text()) if index_path.exists() else {'registry_id':baseline['registry_id'],'versions':[]}
    if not any(str(v['version'])==str(baseline['version']) for v in index['versions']):
        index['versions'].append({'version':str(baseline['version']),'manifest':f"{baseline['version']}/manifest.json"})
    index['versions'].append({'version':str(version),'manifest':f'{version}/manifest.json'})
    index['default_version']=str(version)
    pending_index=artifact_root/f'.index-{uuid.uuid4().hex}.json'
    pending_index.write_text(json.dumps(index,indent=2)+'\n');pending_index.replace(index_path)
    return manifest


def run(args):
    started = time.perf_counter()
    source_bytes = args.baseline_report.read_bytes()
    baseline = json.loads(source_bytes)
    records = baseline['origins']
    context_len = baseline['settings']['selected_context']
    data, provenance = load_real_sources(args.data_root)
    if provenance['snapshot_id'] != baseline['provenance']['snapshot_id']:
        raise ValueError('Data snapshot differs from frozen comparison evidence')
    keys = sorted(set((r['hive_id'], r['origin']) for r in records))
    contexts=[]
    for hive, origin in keys:
        frame = data[data.hive_id.eq(hive) & data.timestamp.le(pd.Timestamp(origin))].tail(context_len)
        if len(frame) != context_len or frame.timestamp.iloc[-1] != pd.Timestamp(origin) or not frame.timestamp.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all():
            raise ValueError(f'Frozen origin has missing context: {hive}, {origin}')
        contexts.append(frame[['weight_kg','temperature_c']].to_numpy(np.float32))
    print(f'Forecasting {len(keys)} frozen origins on CPU, context={context_len}, horizon=168', flush=True)
    adapter = TirexAdapter(args.checkpoint, worker_python=args.worker_python, timeout=1200)
    prediction_started=time.perf_counter()
    forecasts = adapter.predict(contexts, horizon=168)
    inference_seconds=time.perf_counter()-prediction_started
    lookup=dict(zip(keys,forecasts))
    for row in records:
        row['predictions_kg']['tirex2']=float(lookup[(row['hive_id'],row['origin'])][row['horizon_hours']-1])
    weights=choose_weights(records)
    tirex_lstm_weights=choose_weights(records,('lstm','tirex2'))
    print('Validation-only weights',weights,flush=True)
    for row in records:
        for name, selected in [('upgrade_ensemble',weights),('tirex_lstm',tirex_lstm_weights)]:
            row['predictions_kg'][name]=float(sum(row['predictions_kg'][k]*v for k,v in selected.items()))
    scores={}
    for part in ('validation','calibration','test'):
        scores[part]={}
        for horizon in (1,24,72,168):
            subset=[r for r in records if r['partition']==part and r['horizon_hours']==horizon]
            names=set.intersection(*(set(r['predictions_kg']) for r in subset)) if subset else set()
            scores[part][str(horizon)]={group:{name:summarize([r for r in subset if group=='operational' or not r['future_event_present']],name) for name in sorted(names)} for group in ('operational','normal')}
    digest=hashlib.sha256((args.checkpoint/'model.ckpt').read_bytes()).hexdigest()
    result={'schema_version':1,'status':'completed','created_at':datetime.now(timezone.utc).isoformat(),
        'model_id':'NX-AI/TiRex-2','model_revision':args.revision,'model_sha256':digest,
        'evaluation_protocol':'retrospective_regression_existing_test_already_inspected',
        'baseline_report_sha256':hashlib.sha256(source_bytes).hexdigest(),'data_snapshot':provenance['snapshot_id'],
        'context_hours':context_len,'horizons':[1,24,72,168],'weights':weights,'tirex_lstm_weights':tirex_lstm_weights,
        'weight_selection':'validation only, common weights, equal hive contributions, LSTM >= 0.10',
        'unique_origin_count':len(keys),'inference_seconds':inference_seconds,'elapsed_seconds':time.perf_counter()-started,
        'device':'cpu','past_covariates':['temperature_c'],'future_covariates':[],
        'scores':scores,'origins':records,'limitations':baseline['limitations']+[
            'This extends an already inspected test: regression comparison, not a newly sealed test.',
            'Frozen existing LSTM/Chronos predictions are reused; their training and selection are unchanged.',
            'TiRex-2 pretraining overlap with the public bee datasets is unknown.',
            'Inference includes cold loading in a separate CPU worker; no runtime MPS performance claim.',
            'No production model alias is changed by this script.']}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    np.savez_compressed(args.output.with_suffix('.npz'),forecasts=forecasts,keys=np.asarray([h+'|'+o for h,o in keys]))
    print(json.dumps({'report':str(args.output),'weights':weights,'tirex_lstm_weights':tirex_lstm_weights,'inference_seconds':inference_seconds},indent=2),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-report',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--worker-python',required=True)
    p.add_argument('--revision',required=True)
    p.add_argument('--output',type=Path,default=ROOT/'evidence/tirex2_benchmark_20261001.json')
    run(p.parse_args())


if __name__=='__main__':main()
