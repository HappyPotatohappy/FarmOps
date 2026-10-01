#!/usr/bin/env python3
"""Verify running BeeOPS. --train explicitly stores heatwave data and retrains.

Default is read-only. Invoke --train only in the intended local demo runtime.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import time
import urllib.parse
import urllib.request


def request(base,path,payload=None):
    data=json.dumps(payload).encode() if payload is not None else None
    req=urllib.request.Request(base.rstrip('/')+path,data=data,headers={'Content-Type':'application/json'} if data else {})
    with urllib.request.urlopen(req,timeout=240) as response:
        content=response.read()
        return json.loads(content) if 'application/json' in response.headers.get('Content-Type','') else content


def wait_job(base,path,timeout=600):
    deadline=time.monotonic()+timeout;last=None
    while time.monotonic()<deadline:
        job=request(base,path);status=job['status']
        current=(status,job.get('stage'),job.get('training_progress',{}).get('completed_epochs'))
        if current!=last:print(f'  {job["job_id"]}: {current}',flush=True);last=current
        if status in ('completed','partial','failed','interrupted','skipped'):return job
        time.sleep(1)
    raise TimeoutError('Job did not finish within the verification timeout')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',default='http://127.0.0.1:8012')
    parser.add_argument('--train',action='store_true')
    parser.add_argument('--repeat',type=int,default=1,choices=(1,2))
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();base=args.url
    workspaces=request(base,'/workspaces')['workspaces'];checks=[];model_ids=set()
    for hive in workspaces:
        if hive['kind']!='public_real':continue
        identifier=hive['workspace_id'];query=urllib.parse.urlencode({'workspace_id':identifier})
        report=request(base,'/harvest/recommendation?'+query);forecast=report['forecast']
        assert forecast['status']=='ok',report
        assert len(forecast['trajectory'])==168
        assert forecast['model']['components']==['lstm','tirex2'] or set(forecast['model']['components'])=={'lstm','tirex2'}
        model_ids.add((forecast['model']['version'],forecast['model']['run_id']))
        history=request(base,'/forecast/history?limit=168&'+query)
        assert history['predictions'],history
        checks.append({'hive':identifier,'rows':hive['rows'],'forecast_hours':168,'model':forecast['model'],'first_prediction':forecast['trajectory'][0],'history_pairs':len(history['predictions'])})
        print(f'{identifier}: 168-hour forecast and historical predictions OK',flush=True)
    assert len(checks)==3 and len(model_ids)==1
    for route in ('/proposal','/model-guide','/proposal.pdf','/model-guide.pdf','/data/simulations/temperature-drift.zip'):
        assert len(request(base,route))>100
    runs=[]
    if args.train:
        for _ in range(args.repeat):
            preview=request(base,'/data/simulations/temperature/preview',{'filename':'02_heatwave.csv'})
            assert preview['can_commit'],preview
            imported=request(base,'/imports/commit',{'preview_id':preview['preview_id']})
            done=wait_job(base,'/imports/jobs/'+imported['job_id']);assert done['status']=='completed',done
            training_id=done['result']['retraining_job_id'];assert training_id
            job=wait_job(base,'/monitoring/retraining/jobs/'+training_id)
            assert job['status']=='completed',job
            assert job['training']['epochs_completed']==8
            assert set(job['training']['trained_components'])=={'lstm','ensemble_weights'}
            assert all(step['status']=='completed' for step in job['steps'])
            assert job['training']['before_weights_sha256']!=job['training']['after_weights_sha256']
            assert job['training']['tirex_before_sha256']==job['training']['tirex_after_sha256']
            runs.append({'job_id':training_id,'import_id':imported['job_id'],'parent':job['parent_version'],'version':job['new_version'],'epochs':8,'inserted':done['result']['ingest']['inserted'],'duplicates':done['result']['ingest']['duplicate'],'lstm_weights_changed':job['training']['before_weights_sha256']!=job['training']['after_weights_sha256']})
        if len(runs)==2:assert runs[0]['job_id']!=runs[1]['job_id']
    result={'status':'passed','checked_at':datetime.now(timezone.utc).isoformat(),'public_cases':checks,'runs':runs,'documentation_routes':'ok'}
    if args.output:args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
