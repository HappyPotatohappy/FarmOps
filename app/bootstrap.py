"""Prepare common immutable models; legacy registries remain an explicit option."""
import argparse
import json
import os
from pathlib import Path
import pandas as pd
from .ml import ModelService
from .scenarios import generate
from .store import Store


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--runtime',default=os.getenv('BEEOPS_RUNTIME','runtime'))
    parser.add_argument('--source',choices=['synthetic','real'],default='synthetic')
    parser.add_argument('--legacy',action='store_true',help='Use the historical per-hive model workflow')
    args=parser.parse_args()
    runtime=Path(args.runtime).resolve(); store=Store(runtime)
    if args.source=='real':
        frame=pd.read_csv(Path(__file__).resolve().parents[1]/'data/real_hive.csv').iloc[:672]
        source='UFC Apis2 공개 실측 재생 · 시간대 미상(+00:00 표기 가정)'
    else:
        frame=pd.DataFrame(generate(672)); source='합성 교육 데이터 · seed42 계열 · 실제 벌통 아님'
    frame.attrs['source']=source
    if not args.legacy:
        from .shared_runtime import SharedModelCoordinator
        if not store.data_status()['rows']:
            store.ingest(frame.to_dict('records'),source)
            store.set_meta('demo_runtime',args.source=='synthetic')
        coordinator=SharedModelCoordinator(runtime)
        result=coordinator.ensure_ready()
        output={'model_scope':'shared','registry_id':'BeeOPS_Common_Weight',
                'data':store.data_status(),'model':coordinator.model.status(),'model_versions':result}
        (runtime/'shared_bootstrap_result.json').write_text(json.dumps(output,ensure_ascii=False,indent=2))
        print(json.dumps(output,ensure_ascii=False,indent=2))
        return
    store.ingest(frame.to_dict('records'),source)
    model=ModelService(runtime)
    result_path=runtime/'bootstrap_result.json'
    state=model.status()
    consumed=state.get('consumed_through')
    already_evaluated=bool(not state['ready'] and consumed and
                           pd.Timestamp(frame.iloc[-1]['timestamp'])<=pd.Timestamp(consumed))
    if already_evaluated:
        # A rejected initial candidate is a normal model lifecycle outcome.
        # Keep the dashboard available for new observations and don't train on
        # the same consumed initial snapshot each time the container restarts.
        result=(json.loads(result_path.read_text()) if result_path.exists() else
                state.get('last_training') or {'promoted':False,'ready':False,
                                               'gate_reasons':['initial_snapshot_already_evaluated']})
    else:
        result=model.bootstrap(frame)
    pair=None
    prepare_pair=getattr(model,'ensure_comparison_version',None)
    if model.status()['ready'] and callable(prepare_pair):
        try: pair=prepare_pair(frame)
        except Exception as exc:
            pair={'status':'failed','error':str(exc)}
            store.log('WARN','comparison model preparation failed; existing model retained',**pair)
    store.set_meta('demo_runtime',args.source=='synthetic')
    store.log('OK' if model.status()['ready'] else 'WARN','bootstrap finished',**result)
    reused_champion='existing_champion' in result.get('gate_reasons',[])
    if (not reused_champion and not already_evaluated) or not result_path.exists():
        result_path.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({'source':source,'data':store.data_status(),'model':model.status(),'result':result,'model_versions':pair},ensure_ascii=False,indent=2))
    # Readiness is exposed by /health. Gate rejection must not prevent the
    # operator from opening the UI, inspecting the candidate or uploading data.


if __name__=='__main__': main()
