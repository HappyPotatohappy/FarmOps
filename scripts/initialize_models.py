#!/usr/bin/env python3
"""Copy the tracked model seed into an independent writable runtime once.

Call before launching the server. Existing operational models are validated,
never reset to the seed. Empty mounted directories publish their index last;
new ordinary directories are published by one directory rename.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

REGISTRY = 'BeeOPS_Horizon_Weight'


def _inside(root: Path, relative: str) -> Path:
    if not isinstance(relative,str) or not relative or Path(relative).is_absolute() or '..' in Path(relative).parts:
        raise ValueError(f'Model artifact path must stay inside its bundle: {relative}')
    path = (root/relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f'Model artifact escapes its bundle: {relative}')
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def validate_models(root: Path, *, all_bundles=False):
    """Read index identities and verify active (or all seed) artifact hashes."""
    root = Path(root)
    if root.is_symlink():
        raise ValueError('Model root must not be an external symbolic link')
    root = root.resolve()
    for path in root.rglob('*'):
        if path.is_symlink() and not path.resolve().is_relative_to(root):
            raise ValueError(f'External symbolic model artifact is not supported: {path}')
    index_path = _inside(root,'index.json')
    if not index_path.is_file():
        raise ValueError(f'Model index.json is missing: {root}')
    index = json.loads(index_path.read_text())
    entries = index.get('versions')
    active = str(index.get('default_version',''))
    if index.get('registry_id') != REGISTRY or not isinstance(entries,list) or not entries:
        raise ValueError('Invalid model index registry or version list')
    versions = [str(entry.get('version','')) for entry in entries if isinstance(entry,dict)]
    if (len(versions)!=len(entries) or len(set(versions))!=len(versions)
            or any(not re.fullmatch(r'[1-9][0-9]*',version) for version in versions) or active not in versions):
        raise ValueError('Model index requires unique versions and an existing active version')
    verified_files = 0
    for entry,version in zip(entries,versions):
        manifest_path = _inside(root,entry.get('manifest'))
        if not manifest_path.is_file():
            raise FileNotFoundError(f'Missing indexed model manifest: {manifest_path}')
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('registry_id')!=REGISTRY or str(manifest.get('version'))!=version:
            raise ValueError(f'Model manifest identity differs from index: {version}')
        if not (all_bundles or version==active):
            continue
        if set(manifest.get('components',[]))!={'lstm','tirex2'}:
            raise ValueError('Packaged model must contain TiRex-2 and LSTM')
        weights=manifest.get('weights',{})
        if (set(weights)!={'lstm','tirex2'} or not all(isinstance(v,(int,float)) and math.isfinite(v) and .05-1e-8<=v<=.95+1e-8 for v in weights.values())
                or not math.isclose(sum(weights.values()),1.,abs_tol=1e-8)):
            raise ValueError('Invalid active ensemble weights')
        files=manifest.get('files',{})
        if set(files)!={'lstm','tirex2'}:
            raise ValueError('Active model component files are incomplete')
        folder=manifest_path.parent
        lstm=_inside(folder,files['lstm'])
        tirex=_inside(folder,files['tirex2'])
        required=[lstm,tirex/'model.ckpt',tirex/'model-config.yaml']
        hashes=manifest.get('file_sha256')
        if not isinstance(hashes,dict) or not hashes:
            raise ValueError('Model manifest requires artifact SHA-256 hashes')
        required_names={str(path.relative_to(folder)) for path in required}
        if not required_names.issubset(hashes):
            raise ValueError('Model manifest omits a required LSTM/checkpoint/config SHA-256 hash')
        for relative,expected in hashes.items():
            artifact=_inside(folder,relative)
            if not artifact.is_file():
                raise FileNotFoundError(f'Missing model artifact: {artifact}')
            if not isinstance(expected,str) or not re.fullmatch(r'[0-9a-f]{64}',expected) or _sha256(artifact)!=expected:
                raise ValueError(f'Model artifact SHA-256 checksum mismatch: {artifact}')
            verified_files+=1
    return {'active_version':active,'verified_files':verified_files}


def _missing_seed_version(seed: Path, target: Path):
    """Return the seed's active version unless the runtime holds that exact bundle."""
    index_path=seed/'index.json'
    if not index_path.is_file():
        return None
    index=json.loads(index_path.read_text())
    active=str(index.get('default_version',''))
    manifest=next((e.get('manifest') for e in index.get('versions',[]) if str(e.get('version'))==active),None)
    if not manifest:
        return None
    seed_manifest,runtime_manifest=_inside(seed,manifest),_inside(target,manifest)
    if runtime_manifest.is_file() and runtime_manifest.read_bytes()==seed_manifest.read_bytes():
        return None
    return active


def initialize_models(seed_root: Path, artifact_root: Path):
    seed_root, artifact_root = Path(seed_root), Path(artifact_root)
    if artifact_root.is_symlink() or seed_root.is_symlink():
        raise ValueError('Model roots must not be external symbolic links')
    seed, target = seed_root.resolve(), artifact_root.resolve()
    if seed==target or (target/'index.json').is_file():
        report={'models_root':str(target),'initialized':False,**validate_models(target)}
        missing=None if seed==target else _missing_seed_version(seed,target)
        if missing:
            report['seed_version_missing']=missing
        return report
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise ValueError(f'Existing partial model directory has no valid index; retained: {target}')
    if target.is_relative_to(seed):
        raise ValueError('Runtime models must not be created inside the tracked model seed')
    validate_models(seed,all_bundles=True)
    mounted_empty=target.exists()
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary=Path(tempfile.mkdtemp(prefix='.horizon-models-',dir=target if mounted_empty else target.parent))
    staged=temporary/'models'
    try:
        shutil.copytree(seed,staged)
        checked=validate_models(staged,all_bundles=True)
        if mounted_empty:
            if any(path!=temporary for path in target.iterdir()):
                raise ValueError('Model directory changed during initialization; retained for inspection')
            # The mount's own inode must remain intact. Until index publication,
            # any interrupted partial copy is rejected by the next startup.
            for path in sorted(staged.iterdir()):
                if path.name!='index.json':
                    destination=target/path.name
                    if destination.exists():
                        raise FileExistsError(f'Model destination changed during initialization: {destination}')
                    path.rename(destination)
            (staged/'index.json').rename(target/'index.json')
        else:
            staged.rename(target)
        return {'models_root':str(target),'initialized':True,**checked}
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main():
    root=Path(__file__).resolve().parents[1]
    runtime=Path(os.environ.get('BEEOPS_RUNTIME',root/'runtime'))
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=Path,default=root/'data/horizon_models')
    parser.add_argument('--models',type=Path,default=Path(os.environ.get('BEEOPS_HORIZON_MODELS',runtime/'horizon_models')))
    arguments=parser.parse_args()
    report=initialize_models(arguments.seed,arguments.models)
    print(json.dumps(report,ensure_ascii=False))
    if report.get('seed_version_missing'):
        print(f"경고: 이미지의 시작 모델 v{report['seed_version_missing']}이(가) 기존 실행 볼륨에 없습니다. "
              f"기존 모델 v{report['active_version']}을(를) 계속 사용합니다. 새 시작 모델로 초기화하려면 "
              "'docker compose down -v' 후 다시 실행하세요(관측·재학습 이력도 함께 삭제됩니다).",file=sys.stderr)


if __name__=='__main__':
    main()
