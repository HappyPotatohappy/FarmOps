"""Checked-in, licensed samples; request IDs never become filesystem paths."""
import json
from pathlib import Path


DATA_ROOT = Path(__file__).resolve().parents[1] / 'data'


def sample_catalog():
    catalog = json.loads((DATA_ROOT / 'public_samples.json').read_text())
    return {'default_sample_id': catalog['default_sample_id'],
            'samples': [{key: value for key, value in sample.items() if key != 'file'}
                        for sample in catalog['samples']]}


def sample_file(sample_id):
    catalog = json.loads((DATA_ROOT / 'public_samples.json').read_text())
    item = next((sample for sample in catalog['samples'] if sample['id'] == sample_id), None)
    if item is None:
        raise KeyError(sample_id)
    path = (DATA_ROOT / item['file']).resolve()
    if not path.is_relative_to(DATA_ROOT.resolve()) or not path.is_file():
        raise RuntimeError('Configured public sample is missing or outside the data directory')
    return path
