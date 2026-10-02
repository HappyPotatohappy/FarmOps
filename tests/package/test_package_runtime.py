"""Relocating the final package must keep registries local and models immutable."""
import importlib.util
import json
from pathlib import Path
import shutil
import sqlite3

import pytest

SCRIPT=Path(__file__).resolve().parents[2]/'scripts/initialize_runtime.py'


def initialize(root,runtime):
    assert SCRIPT.is_file(), 'Package runtime initializer is missing'
    spec=importlib.util.spec_from_file_location('package_initializer',SCRIPT)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.initialize_runtime(root,runtime)


def registry(path,old_root):
    path.mkdir(parents=True)
    bundle=path/'artifacts/run-one/artifacts/bundle'
    bundle.mkdir(parents=True);(bundle/'model.keras').write_bytes(b'unchanged real model bytes')
    (path/'training_state.json').write_text('{"consumed_through":"2026-07-21"}')
    with sqlite3.connect(path/'mlflow.db') as db:
        db.executescript('''CREATE TABLE experiments(experiment_id INTEGER PRIMARY KEY,name TEXT,artifact_location TEXT);
        CREATE TABLE runs(run_uuid TEXT PRIMARY KEY,experiment_id INTEGER,artifact_uri TEXT,status TEXT);
        CREATE TABLE model_versions(name TEXT,version INTEGER,source TEXT,storage_location TEXT,run_id TEXT);
        CREATE TABLE registered_model_aliases(alias TEXT,version INTEGER,name TEXT);''')
        db.execute('INSERT INTO experiments VALUES(0,?,?)',('Default',(old_root/'mlruns/0').as_uri()))
        db.execute('INSERT INTO experiments VALUES(1,?,?)',('BeeOPS_Common',(old_root/'artifacts').as_uri()))
        db.execute('INSERT INTO runs VALUES(?,?,?,?)',('run-one',1,(old_root/'artifacts/run-one/artifacts').as_uri(),'FINISHED'))
        db.execute('INSERT INTO model_versions VALUES(?,?,?,?,?)',('BeeOPS_Common_Weight',4,(old_root/'artifacts/run-one/artifacts/bundle').as_uri(),(old_root/'artifacts/run-one/artifacts/bundle').as_uri(),'run-one'))
        db.execute('INSERT INTO registered_model_aliases VALUES(?,?,?)',('champion',4,'BeeOPS_Common_Weight'))


def rows(path,table):
    with sqlite3.connect(f'file:{path.resolve()}?mode=ro',uri=True) as db:
        return db.execute('SELECT * FROM '+table).fetchall()


def tree_bytes(root):
    return {str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()}


def make_package(tmp_path):
    root=tmp_path/'package'
    registry(root/'data/compatibility_seed/shared_models',Path('/__BEEOPS_SEED__/shared_models'))
    return root


def test_initialization_copies_seed_and_rebases_only_operational_registry_paths(tmp_path):
    root=make_package(tmp_path);seed=root/'data/compatibility_seed/shared_models'
    before=tree_bytes(seed);runtime=root/'runtime'
    report=initialize(root,runtime)
    copied=runtime/'shared_models'
    assert report['initialized'] is True
    assert tree_bytes(seed)==before
    assert (copied/'artifacts/run-one/artifacts/bundle/model.keras').read_bytes()==b'unchanged real model bytes'
    assert (copied/'training_state.json').read_bytes()==before['training_state.json']
    assert rows(copied/'mlflow.db','runs')==[('run-one',1,(copied/'artifacts/run-one/artifacts').as_uri(),'FINISHED')]
    assert rows(copied/'mlflow.db','registered_model_aliases')==[('champion',4,'BeeOPS_Common_Weight')]
    assert rows(copied/'mlflow.db','model_versions')[0][2:4]==((copied/'artifacts/run-one/artifacts/bundle').as_uri(),)*2
    assert not list(runtime.glob('.shared_models-*'))


def test_repeat_initialization_is_byte_idempotent_and_preserves_runtime_data(tmp_path):
    root=make_package(tmp_path);runtime=root/'runtime';initialize(root,runtime)
    (runtime/'observations.keep').write_bytes(b'user data')
    before=tree_bytes(runtime)
    report=initialize(root,runtime)
    assert report['initialized'] is False
    assert tree_bytes(runtime)==before


def test_moving_package_rebases_shared_and_retraining_registries_without_old_source(tmp_path):
    root=make_package(tmp_path);runtime=root/'runtime';initialize(root,runtime)
    registry(runtime/'temperature_drift',runtime/'temperature_drift')
    (runtime/'operations.sqlite').write_bytes(b'opaque preserved ledger')
    state=(runtime/'shared_models/training_state.json').read_bytes()
    moved=tmp_path/'moved package';root.rename(moved)
    report=initialize(moved,moved/'runtime')
    assert report['initialized'] is False and not root.exists()
    for name in ('shared_models','temperature_drift'):
        current=moved/'runtime'/name
        assert rows(current/'mlflow.db','runs')[0][2]==(current/'artifacts/run-one/artifacts').as_uri()
        assert rows(current/'mlflow.db','registered_model_aliases')==[('champion',4,'BeeOPS_Common_Weight')]
        assert (current/'artifacts/run-one/artifacts/bundle/model.keras').read_bytes()==b'unchanged real model bytes'
    assert (moved/'runtime/shared_models/training_state.json').read_bytes()==state
    assert (moved/'runtime/operations.sqlite').read_bytes()==b'opaque preserved ledger'


def test_missing_seed_artifact_rejects_without_publishing_partial_runtime(tmp_path):
    root=make_package(tmp_path);seed=root/'data/compatibility_seed/shared_models'
    shutil.rmtree(seed/'artifacts/run-one/artifacts/bundle')
    before=tree_bytes(seed)
    with pytest.raises((ValueError,FileNotFoundError),match='artifact|Artifact'):
        initialize(root,root/'runtime')
    assert not (root/'runtime/shared_models').exists()
    assert tree_bytes(seed)==before
    assert not list((root/'runtime').glob('.shared_models-*'))


def test_remote_artifact_source_rejected_without_modifying_existing_database(tmp_path):
    root=make_package(tmp_path);runtime=root/'runtime';initialize(root,runtime)
    dbpath=runtime/'shared_models/mlflow.db'
    with sqlite3.connect(dbpath) as db:db.execute("UPDATE model_versions SET source='https://example.invalid/model'")
    before=dbpath.read_bytes()
    with pytest.raises(ValueError,match='local|remote|Unsupported'):
        initialize(root,runtime)
    assert dbpath.read_bytes()==before


def test_invalid_second_registry_is_validated_before_first_registry_is_rebased(tmp_path):
    root=make_package(tmp_path);runtime=root/'runtime';initialize(root,runtime)
    registry(runtime/'temperature_drift',runtime/'temperature_drift')
    shutil.rmtree(runtime/'temperature_drift/artifacts/run-one/artifacts/bundle')
    moved=tmp_path/'moved';root.rename(moved)
    dbpath=moved/'runtime/shared_models/mlflow.db';before=dbpath.read_bytes()
    with pytest.raises((ValueError,FileNotFoundError),match='artifact|Artifact'):
        initialize(moved,moved/'runtime')
    assert dbpath.read_bytes()==before


def test_existing_registry_symlink_cannot_rewrite_an_external_source(tmp_path):
    root=make_package(tmp_path);runtime=root/'runtime';runtime.mkdir()
    external=tmp_path/'external';registry(external,Path('/former/source'))
    (runtime/'shared_models').symlink_to(external,target_is_directory=True)
    before=tree_bytes(external)
    with pytest.raises(ValueError,match='symbolic|outside'):
        initialize(root,runtime)
    assert tree_bytes(external)==before


def test_actual_bundled_seed_initializes_all_four_registry_versions(tmp_path):
    root=SCRIPT.parents[1];seed=root/'data/compatibility_seed/shared_models'
    before=tree_bytes(seed)
    runtime=tmp_path/'real seed runtime'
    initialize(root,runtime)
    assert len(rows(runtime/'shared_models/mlflow.db','model_versions'))==4
    assert rows(runtime/'shared_models/mlflow.db','registered_model_aliases')==[('champion',4,'BeeOPS_Common_Weight')]
    assert tree_bytes(seed)==before


def test_failed_run_without_artifact_directory_does_not_block_startup(tmp_path):
    # A retrain that dies after create_run but before its first artifact leaves no directory.
    root=make_package(tmp_path);runtime=root/'runtime';initialize(root,runtime)
    registry(runtime/'temperature_drift',runtime/'temperature_drift')
    with sqlite3.connect(runtime/'temperature_drift/mlflow.db') as db:
        db.execute('INSERT INTO runs VALUES(?,?,?,?)',('run-empty',1,(runtime/'temperature_drift/artifacts/run-empty/artifacts').as_uri(),'FAILED'))
    moved=tmp_path/'moved';root.rename(moved)
    initialize(moved,moved/'runtime')
    uris=dict((r[0],r[2]) for r in rows(moved/'runtime/temperature_drift/mlflow.db','runs'))
    assert uris['run-empty']==(moved/'runtime/temperature_drift/artifacts/run-empty/artifacts').resolve().as_uri()
