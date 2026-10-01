#!/usr/bin/env python3
"""Initialize/rebase only this package's local registries before starting writers.

The immutable compatibility seed is copied once. Relocation reads old paths as
identifiers only; it never opens the former folder. Model bytes, observations,
training state, run IDs, versions and aliases are left unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
from urllib.parse import unquote, urlsplit


def _local_path(value: str) -> Path:
    parsed = urlsplit(value)
    if parsed.scheme not in ('', 'file') or parsed.netloc not in ('', 'localhost') or parsed.query or parsed.fragment:
        raise ValueError(f'Only local artifact paths are supported: {value}')
    path = Path(unquote(parsed.path) if parsed.scheme else value)
    if not path.is_absolute() or '..' in path.parts:
        raise ValueError(f'Artifact path must be absolute and normalized: {value}')
    return path


def _plan_database(database: Path, physical: Path, logical: Path):
    """Validate local copied artifacts and plan scoped updates without writing."""
    if physical.is_symlink() or database.is_symlink():
        raise ValueError('Runtime registries must not use symbolic links outside the package')
    physical = physical.resolve()
    logical = logical.resolve()
    changes = []
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro', uri=True) as db:
        if db.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
            raise ValueError(f'Registry integrity check failed: {database}')
        experiments = db.execute('SELECT experiment_id,name,artifact_location FROM experiments').fetchall()
        roots = {}
        for identifier, name, location in experiments:
            old = _local_path(location)
            if name == 'Default':
                relative = Path('mlruns')/str(identifier)
            elif old.name == 'artifacts':
                relative = Path('artifacts')
            else:
                raise ValueError(f'Unsupported local experiment artifact root: {location}')
            roots[str(identifier)] = (old, relative)
            new = (logical/relative).as_uri()
            if location != new:
                changes.append(('UPDATE experiments SET artifact_location=? WHERE experiment_id=?', (new,identifier)))

        def artifact(value, experiment_id):
            old = _local_path(value)
            try:
                old_root, root_relative = roots[str(experiment_id)]
                relative = old.relative_to(old_root)
            except (KeyError, ValueError) as exc:
                raise ValueError(f'Artifact is outside its registered experiment root: {value}') from exc
            copied = physical/root_relative/relative
            resolved = copied.resolve()
            if not resolved.is_relative_to(physical):
                raise ValueError(f'Artifact escapes the copied registry: {copied}')
            if not copied.exists():
                raise FileNotFoundError(f'Missing copied artifact: {copied}')
            return (logical/root_relative/relative).as_uri()

        runs = {}
        for run_id, experiment_id, uri in db.execute('SELECT run_uuid,experiment_id,artifact_uri FROM runs'):
            new = artifact(uri,experiment_id)
            runs[run_id] = experiment_id
            if uri != new:
                changes.append(('UPDATE runs SET artifact_uri=? WHERE run_uuid=?',(new,run_id)))
        for name, version, source, storage, run_id in db.execute(
                'SELECT name,version,source,storage_location,run_id FROM model_versions'):
            if run_id not in runs:
                raise ValueError(f'Model version has no local registered run: {name}/{version}')
            new_source = artifact(source,runs[run_id])
            new_storage = artifact(storage,runs[run_id]) if storage is not None else None
            if (source,storage) != (new_source,new_storage):
                changes.append(('UPDATE model_versions SET source=?,storage_location=? WHERE name=? AND version=?',
                                (new_source,new_storage,name,version)))
    return {'database':database, 'changes':changes}


def _apply(plan):
    if not plan['changes']:
        return
    with sqlite3.connect(plan['database']) as db:
        db.execute('BEGIN IMMEDIATE')
        for statement, values in plan['changes']:
            db.execute(statement,values)
        if db.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
            raise ValueError('Registry integrity check failed after relocation')


def initialize_runtime(root: Path, runtime: Path):
    """Prepare an independent runtime; call only while its server is stopped."""
    root, runtime = Path(root).resolve(), Path(runtime).resolve()
    seed = root/'data/compatibility_seed/shared_models'
    if runtime == seed or runtime.is_relative_to(seed):
        raise ValueError('The immutable compatibility seed cannot be used as runtime')
    runtime.mkdir(parents=True,exist_ok=True)
    shared = runtime/'shared_models'
    temporary = None
    initialized = not shared.exists()
    try:
        if initialized:
            if not (seed/'mlflow.db').is_file():
                raise FileNotFoundError(f'Compatibility seed registry is missing: {seed}')
            if any(path.is_symlink() for path in seed.rglob('*')):
                raise ValueError('Compatibility seed artifacts must not use symbolic links')
            temporary = Path(tempfile.mkdtemp(prefix='.shared_models-',dir=runtime))
            physical_shared = temporary/'shared_models'
            shutil.copytree(seed,physical_shared,ignore=shutil.ignore_patterns('*-wal','*-shm'))
        else:
            physical_shared = shared
            if not (shared/'mlflow.db').is_file():
                raise FileNotFoundError(f'Existing shared registry is incomplete; retained for inspection: {shared}')
        plans = [_plan_database(physical_shared/'mlflow.db',physical_shared,shared)]
        temperature = runtime/'temperature_drift'
        if (temperature/'mlflow.db').is_file():
            plans.append(_plan_database(temperature/'mlflow.db',temperature,temperature))
        # Every registry is validated first, including the moved temperature
        # registry, before changing even one operational path.
        for plan in plans:
            _apply(plan)
        if initialized:
            physical_shared.rename(shared)
        return {'runtime':str(runtime),'initialized':initialized,
                'registry_count':len(plans),'rebased_rows':sum(len(plan['changes']) for plan in plans)}
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime',type=Path,default=Path(os.environ.get('BEEOPS_RUNTIME',root/'runtime')))
    arguments = parser.parse_args()
    print(json.dumps(initialize_runtime(root,arguments.runtime),ensure_ascii=False))


if __name__ == '__main__':
    main()
