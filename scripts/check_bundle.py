#!/usr/bin/env python3
"""Verify repository assets before running BeeOPS; does not train or mutate data."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sqlite3

from fetch_model import sha256

ROOT=Path(__file__).resolve().parents[1]


def check_bundle(root=ROOT,allow_missing_checkpoint=False):
    root=Path(root).resolve();bundle=json.loads((root/'bundle.json').read_text())
    index=json.loads((root/'data/horizon_models/index.json').read_text())
    version=str(bundle['initial_model']['version'])
    if str(index['default_version'])!=version:raise ValueError('Bundled seed index no longer matches bundle.json')
    folder=root/'data/horizon_models'/version
    manifest=json.loads((folder/'manifest.json').read_text())
    if manifest['run_id']!=bundle['initial_model']['run_id']:raise ValueError('LSTM bundle run ID mismatch')
    count=0;missing=[]
    for name,digest in manifest['file_sha256'].items():
        path=(folder/name).resolve()
        if not path.is_relative_to(folder.resolve()):raise ValueError('Artifact outside model folder')
        if name=='tirex2/model.ckpt' and not path.is_file() and allow_missing_checkpoint:
            missing.append(name);continue
        if not path.is_file():raise ValueError(f'Missing {path.relative_to(root)}; run python scripts/fetch_model.py first')
        if sha256(path)!=digest:raise ValueError(f'Artifact SHA-256 mismatch: {path.relative_to(root)}')
        count+=1
    downloads=json.loads((root/'models.lock.json').read_text())['downloads']
    if len(downloads)!=1 or downloads[0]['sha256']!=manifest['file_sha256']['tirex2/model.ckpt']:
        raise ValueError('Download manifest differs from bundled checkpoint hash')
    for name in ('app/main.py','app/static/app.js','data/trend_cases/manifest.json',
                 'data/trend_cases.zip','data/public_samples.json','data/real_hive.csv',
                 'data/simulations/temperature_practice.zip','data/compatibility_seed/shared_models/mlflow.db'):
        if not (root/name).is_file():raise ValueError(f'Required file missing: {name}')
    cases=json.loads((root/'data/trend_cases/manifest.json').read_text())
    for case in cases['cases']:
        path=(root/'data/trend_cases'/case['file']).resolve()
        if not path.is_relative_to((root/'data/trend_cases').resolve()) or not path.is_file():raise ValueError('Missing observed case CSV')
    dbpath=root/'data/compatibility_seed/shared_models/mlflow.db'
    with sqlite3.connect(dbpath.as_uri()+'?mode=ro',uri=True) as db:
        if db.execute('PRAGMA integrity_check').fetchone()[0]!='ok':raise ValueError('Compatibility seed registry is corrupt')
    return {'status':'ok','initial_version':version,'verified_artifacts':count,'deferred_downloads':missing,'observed_cases':len(cases['cases'])}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-missing-checkpoint',action='store_true',help='For source-only GitHub CI; installation must still download the checkpoint')
    args=parser.parse_args()
    print(json.dumps(check_bundle(allow_missing_checkpoint=args.allow_missing_checkpoint),ensure_ascii=False))

if __name__=='__main__':main()
