import hashlib
from pathlib import Path
import pytest
from scripts.fetch_model import ensure_model


def metadata(tmp_path, content=b'checkpoint'):
    source=tmp_path/'official.ckpt';source.write_bytes(content)
    return {'path':'data/horizon_models/9/tirex2/model.ckpt','url':source.as_uri(),
            'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()}


def test_download_missing_model_then_reuse_without_network(tmp_path):
    entry=metadata(tmp_path);target=ensure_model(tmp_path,entry)
    assert target.read_bytes()==b'checkpoint'
    (tmp_path/'official.ckpt').unlink()
    assert ensure_model(tmp_path,entry)==target


def test_bad_download_preserves_existing_file_and_cleans_partial(tmp_path):
    entry=metadata(tmp_path);entry['sha256']='0'*64
    target=tmp_path/entry['path'];target.parent.mkdir(parents=True);target.write_bytes(b'previous valid file')
    with pytest.raises(ValueError,match='SHA-256'):ensure_model(tmp_path,entry)
    assert target.read_bytes()==b'previous valid file'
    assert not list(target.parent.glob('*.part'))


def test_pointer_file_is_replaced_with_verified_model(tmp_path):
    entry=metadata(tmp_path);target=tmp_path/entry['path'];target.parent.mkdir(parents=True)
    target.write_text('version https://git-lfs.github.com/spec/v1\n')
    ensure_model(tmp_path,entry)
    assert target.read_bytes()==b'checkpoint'


def test_path_escape_and_mismatch_are_rejected(tmp_path):
    entry=metadata(tmp_path);entry['path']='../outside.ckpt'
    with pytest.raises(ValueError,match='outside'):ensure_model(tmp_path,entry)
    entry=metadata(tmp_path);entry['bytes']=100
    with pytest.raises(ValueError,match='size'):ensure_model(tmp_path,entry)
