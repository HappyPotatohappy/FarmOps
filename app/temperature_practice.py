"""Trusted built-in practice files; never grant recovery from an upload name."""
import csv
import hashlib
import io
import json
from pathlib import Path

from .contracts import validate_series

ROOT = Path(__file__).resolve().parents[1]/'data/simulations/temperature_practice'
PHASES = {'01_baseline.csv':'baseline', '02_heatwave.csv':'heatwave', '03_followup.csv':'followup'}


def practice_rows(filename):
    if filename not in PHASES:
        raise ValueError('Unknown built-in temperature practice phase')
    content = (ROOT/filename).read_bytes()
    manifest = json.loads((ROOT/'manifest.json').read_text())
    if hashlib.sha256(content).hexdigest() != manifest['files'][filename]['sha256']:
        raise ValueError('Built-in temperature practice file checksum mismatch')
    return validate_series(list(csv.DictReader(io.StringIO(content.decode('utf-8-sig')))))


def heatwave_rows():
    return practice_rows('01_baseline.csv')+practice_rows('02_heatwave.csv')


def validate_practice_recovery(rows, existing):
    """Only the fixed 504-row exercise can prepend its missing real file rows.

    Every existing observation must still match one of the fixed 576 practice
    rows. This deliberately refuses a ledger with arbitrary user additions.
    """
    from .store import same_observation_values
    expected = heatwave_rows()
    if len(rows) != len(expected) or any(
            row['timestamp'] != canonical['timestamp'] or row['event'] != canonical['event']
            or not same_observation_values(row, canonical) for row, canonical in zip(rows, expected)):
        raise ValueError('Recovery requires the canonical built-in temperature practice rows')
    canonical = {row['timestamp']:row for row in expected+practice_rows('03_followup.csv')}
    for row in existing:
        original = canonical.get(row['timestamp'])
        if original is None or row['event'] != original['event'] or not same_observation_values(row, original):
            raise ValueError('Existing data differs from the built-in temperature practice; original retained')


def practice_preview_input(filename):
    selected = heatwave_rows() if filename == '02_heatwave.csv' else practice_rows(filename)
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=['timestamp','hive_id','weight_kg','temperature_c','event'])
    writer.writeheader()
    writer.writerows(selected)
    phase = PHASES[filename]
    return stream.getvalue().encode(), {
        'phase':phase, 'filename':filename, 'data_kind':'synthetic',
        'includes_baseline':phase == 'heatwave',
        'baseline_rows':336 if phase == 'heatwave' else 0,
        'heatwave_rows':168 if phase == 'heatwave' else 0,
        'reference_hours':168,
    }
