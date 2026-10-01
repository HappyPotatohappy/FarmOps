#!/usr/bin/env python3
"""Fetch only the pinned public TiRex checkpoint and verify its exact bytes.

The trained LSTM and its scaler/blend weights are already part of this repo.
A partially downloaded or mismatching file is never published as a model.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.request

ROOT=Path(__file__).resolve().parents[1]


def sha256(path):
    digest=hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda:source.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def ensure_model(root,entry):
    root=Path(root).resolve();target=(root/entry['path']).resolve()
    if not target.is_relative_to(root):raise ValueError('Model path is outside the package')
    expected_size=int(entry['bytes']);expected_hash=entry['sha256']
    if target.is_file() and target.stat().st_size==expected_size and sha256(target)==expected_hash:
        print(f'Model verified: {entry["path"]}',flush=True);return target
    target.parent.mkdir(parents=True,exist_ok=True)
    fd,temporary=tempfile.mkstemp(prefix='model-',suffix='.part',dir=target.parent)
    temp=Path(temporary);digest=hashlib.sha256();size=0;last_log=0
    try:
        request=urllib.request.Request(entry['url'],headers={'User-Agent':'BeeOPS-model-installer/1.0'})
        print(f'Downloading TiRex-2 ({expected_size/1_000_000:.1f} MB); first setup only.',flush=True)
        with os.fdopen(fd,'wb') as output,urllib.request.urlopen(request,timeout=120) as response:
            while block:=response.read(1024*1024):
                size+=len(block)
                if size>expected_size:raise ValueError('Downloaded model size exceeds expected size')
                output.write(block);digest.update(block)
                if size-last_log>=50*1024*1024:
                    print(f'  {size/1_000_000:.0f} / {expected_size/1_000_000:.0f} MB',flush=True);last_log=size
            output.flush();os.fsync(output.fileno())
        if size!=expected_size:raise ValueError(f'Downloaded model size mismatch: {size} != {expected_size}')
        if digest.hexdigest()!=expected_hash:raise ValueError('Downloaded model SHA-256 mismatch')
        temp.replace(target)
        print(f'Model downloaded and SHA-256 verified: {entry["path"]}',flush=True)
        return target
    finally:temp.unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    args=parser.parse_args();root=args.root.resolve()
    entries=json.loads((root/'models.lock.json').read_text())['downloads']
    for entry in entries:ensure_model(root,entry)


if __name__=='__main__':main()
