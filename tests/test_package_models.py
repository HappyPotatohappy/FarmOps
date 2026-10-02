"""Runtime model copies must never overwrite the tracked seed or newer training."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/initialize_models.py'


def initialize(seed,destination):
    assert SCRIPT.is_file(),'Runtime model initializer is missing'
    spec=importlib.util.spec_from_file_location('package_models_initializer',SCRIPT)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.initialize_models(seed,destination)


def bundle(root,version='9'):
    folder=root/version
    files={'lstm.keras':b'lstm learned weights '+version.encode(),
           'tirex2/model.ckpt':b'frozen foundation checkpoint',
           'tirex2/model-config.yaml':b'quantiles: 9\n',
           'training_snapshot.json':b'{"actual_rows":true}'}
    for relative,content in files.items():
        path=folder/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content)
    manifest={'registry_id':'BeeOPS_Horizon_Weight','version':version,'run_id':'run-'+version,
              'components':['lstm','tirex2'],'weights':{'lstm':.45,'tirex2':.55},
              'files':{'lstm':'lstm.keras','tirex2':'tirex2'},
              'file_sha256':{name:hashlib.sha256(content).hexdigest() for name,content in files.items()}}
    (folder/'manifest.json').write_text(json.dumps(manifest))
    index={'registry_id':'BeeOPS_Horizon_Weight','default_version':version,
           'versions':[{'version':version,'manifest':version+'/manifest.json'}]}
    (root/'index.json').write_text(json.dumps(index))


def contents(root):
    return {str(path.relative_to(root)):path.read_bytes() for path in root.rglob('*') if path.is_file()}


def test_seed_is_copied_atomically_without_modifying_original(tmp_path):
    seed=tmp_path/'seed';bundle(seed);before=contents(seed)
    target=tmp_path/'runtime/horizon_models'
    report=initialize(seed,target)
    assert report['initialized'] is True and report['active_version']=='9'
    assert contents(target)==before and contents(seed)==before
    assert not list(target.parent.glob('.horizon-models-*'))


def test_existing_newer_default_is_preserved_even_when_seed_is_unavailable(tmp_path):
    seed=tmp_path/'seed';bundle(seed)
    target=tmp_path/'runtime/horizon_models';initialize(seed,target);bundle(target,'10')
    before=contents(target);seed.rename(tmp_path/'seed removed')
    report=initialize(seed,target)
    assert report['initialized'] is False and report['active_version']=='10'
    assert contents(target)==before


def test_empty_mount_is_initialized_without_replacing_directory(tmp_path):
    seed=tmp_path/'seed';bundle(seed)
    target=tmp_path/'mounted';target.mkdir();inode=target.stat().st_ino
    report=initialize(seed,target)
    assert report['initialized'] is True and contents(target)==contents(seed)
    assert target.stat().st_ino==inode


def test_partial_nonempty_destination_is_rejected_without_clearing_it(tmp_path):
    seed=tmp_path/'seed';bundle(seed)
    target=tmp_path/'partial';target.mkdir();(target/'keep.txt').write_text('retained')
    before=contents(target)
    with pytest.raises(ValueError,match='partial|index'):
        initialize(seed,target)
    assert contents(target)==before


def test_corrupt_seed_checkpoint_is_rejected_before_publish(tmp_path):
    seed=tmp_path/'seed';bundle(seed);(seed/'9/tirex2/model.ckpt').write_bytes(b'corrupt')
    target=tmp_path/'new'
    with pytest.raises(ValueError,match='hash|checksum|SHA'):
        initialize(seed,target)
    assert not target.exists()


def test_existing_corrupted_active_model_is_rejected_and_never_reset_to_seed(tmp_path):
    seed=tmp_path/'seed';bundle(seed)
    target=tmp_path/'runtime';bundle(target,'10');(target/'10/lstm.keras').write_bytes(b'damaged')
    before=contents(target)
    with pytest.raises(ValueError,match='hash|checksum|SHA'):
        initialize(seed,target)
    assert contents(target)==before


def test_same_seed_and_model_root_is_validated_without_writing(tmp_path):
    seed=tmp_path/'seed';bundle(seed);before=contents(seed)
    report=initialize(seed,seed)
    assert report['initialized'] is False and contents(seed)==before


def test_required_checkpoint_hash_cannot_be_omitted(tmp_path):
    seed=tmp_path/'seed';bundle(seed)
    path=seed/'9/manifest.json';value=json.loads(path.read_text());del value['file_sha256']['tirex2/model.ckpt'];path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='hash|checksum|SHA'):
        initialize(seed,tmp_path/'new')


def test_external_checkpoint_symlink_cannot_create_source_dependency(tmp_path):
    seed=tmp_path/'seed';bundle(seed)
    original=seed/'9/tirex2/model.ckpt';external=tmp_path/'outside';original.rename(external);original.symlink_to(external)
    before=external.read_bytes()
    with pytest.raises(ValueError,match='symbolic|outside|escape'):
        initialize(seed,tmp_path/'new')
    assert external.read_bytes()==before


def test_packaged_real_seed_validates_in_place_without_copy(tmp_path):
    root=SCRIPT.parents[1]/'data/horizon_models'
    index_before=(root/'index.json').read_bytes()
    report=initialize(root,root)
    assert report['initialized'] is False and report['active_version']=='9'
    assert (root/'index.json').read_bytes()==index_before


def test_existing_runtime_reports_newer_seed_without_replacing_models(tmp_path):
    seed=tmp_path/'seed';bundle(seed)
    target=tmp_path/'runtime/horizon_models';initialize(seed,target)
    before=contents(target)
    bundle(seed,'11')
    report=initialize(seed,target)
    assert report['initialized'] is False and report['active_version']=='9'
    assert report['seed_version_missing']=='11'
    assert contents(target)==before


def test_existing_runtime_from_same_seed_reports_no_mismatch(tmp_path):
    seed=tmp_path/'seed';bundle(seed)
    target=tmp_path/'runtime/horizon_models';initialize(seed,target)
    bundle(target,'10')
    assert 'seed_version_missing' not in initialize(seed,target)
