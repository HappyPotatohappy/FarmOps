#!/usr/bin/env python3
"""Prepare additional UFC observations and provenance as JSON, without training.

From beeops/: ../.venv/bin/python scripts/prepare_additional_ufc_data.py
Default output: /tmp/beeops-ufc-prepared.json (no final CSV is written).
The source clock has no timezone. +00:00 is bookkeeping, not verified UTC.
Selections use only fixed physical bounds, numeric jump flags and coverage;
they are not selected by model performance or claimed to be representative.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.contracts import validate_series


RAW = ROOT / "data/raw"
METADATA = RAW / "zenodo_20399470_metadata.json"
RECORD = "https://zenodo.org/records/20399470"
DATASETS = {
    "meliponini3": ("DadosMeliponas.csv", 3, "ufc_meliponini_s3", "Meliponini (stingless bees)"),
    "apis1": ("dadosColmeia1Apis.csv", 1, "ufc_apis_1", "Apis mellifera"),
    "meliponini1": ("DadosMeliponas.csv", 1, "ufc_meliponini_s1", "Meliponini (stingless bees)"),
}
# Pinned from the official Zenodo record, independently rechecked via its API.
SOURCE_MD5 = {
    "DadosMeliponas.csv": "e1315f0db0f3505362d2feacb7ac9c67",
    "dadosColmeia1Apis.csv": "84206d6ca170906019874e0ed5f27650",
}


def hourly_sensor(raw: pd.DataFrame, sensor_id: int) -> tuple[pd.DataFrame, dict]:
    required = {"time", "id_sensor", "weight", "ext_temperature", "ext_humidity"}
    if not required.issubset(raw.columns):
        raise ValueError(f"Missing source columns: {sorted(required - set(raw.columns))}")
    source = raw.loc[raw["id_sensor"].eq(sensor_id)].copy()
    if source.empty:
        raise ValueError(f"No source rows for sensor {sensor_id}")
    source["time"] = pd.to_datetime(source["time"], format="mixed", errors="raise")
    if source["time"].isna().any() or source["time"].dt.tz is not None:
        raise ValueError("Expected non-null naive source timestamps")
    if source["time"].duplicated().any():
        raise ValueError("Duplicate source timestamps need explicit resolution")
    physical = (
        np.isfinite(source["weight"])
        & source["weight"].between(0, 300, inclusive="right")
        & np.isfinite(source["ext_temperature"])
        & source["ext_temperature"].between(-50, 60)
    )
    sentinel = source["ext_temperature"].eq(0) & source["ext_humidity"].eq(0)
    valid = source.loc[physical & ~sentinel].sort_values("time")
    bins = valid.set_index("time").resample("h", closed="left", label="right")
    hourly = bins[["weight", "ext_temperature"]].median().dropna()
    hourly["raw_samples"] = bins.size().reindex(hourly.index)
    if hourly.empty:
        raise ValueError("No occupied valid hourly bins remain")
    stats = {
        "sensor_raw_rows": len(source),
        "valid_raw_rows": len(valid),
        "rejected_total_rows": int((~physical | sentinel).sum()),
        "rejected_joint_zero_ambient_rows": int(sentinel.sum()),
        "rejected_physical_bounds_rows": int((~physical).sum()),
        "rejected_overlap_rows": int((~physical & sentinel).sum()),
        "occupied_hourly_bins": len(hourly),
        "raw_start": source["time"].min().isoformat(),
        "raw_end": source["time"].max().isoformat(),
        "raw_duplicate_timestamps": 0,
    }
    return hourly, stats


def select_clean_block(hourly: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Keep the longest contiguous block without >1 kg adjacent-hour jumps.

    A gap starts a new segment, so no jump is inferred across missing hours.
    A flagged observation is excluded and splits clean blocks; no rows are
    joined across the exclusion. The earliest block wins a length tie.
    """
    segment_ids = hourly.index.to_series().diff().ne(pd.Timedelta(hours=1)).cumsum()
    blocks, segments = [], []
    alert_count = 0
    for _, segment in hourly.groupby(segment_ids):
        alert = segment["weight"].diff().abs().gt(1)
        alert_count += int(alert.sum())
        clean = segment.loc[~alert]
        clean_ids = clean.index.to_series().diff().ne(pd.Timedelta(hours=1)).cumsum()
        parts = [part for _, part in clean.groupby(clean_ids)]
        blocks.extend(parts)
        segments.append({
            "start_hour_end": segment.index[0].isoformat(),
            "end_hour_end": segment.index[-1].isoformat(),
            "rows": len(segment), "derived_alert_rows": int(alert.sum()),
            "longest_clean_block_rows": max(map(len, parts)),
        })
    selected = max(blocks, key=len)
    return selected, {
        "method": "Longest contiguous clean hourly block; earliest start breaks ties",
        "model_metrics_used": False,
        "continuous_segments": len(segments),
        "longest_continuous_segment_rows": max(s["rows"] for s in segments),
        "derived_alert_hourly_rows": alert_count,
        "clean_blocks": len(blocks),
        "selected_rows": len(selected),
        "segments": segments,
        "scope_limitation": "Coverage-selected clean segment; not a full source history or an unbiased evaluation sample",
    }


def prepare_dataset(key: str, raw_dir: Path = RAW, metadata_path: Path = METADATA) -> dict:
    filename, sensor_id, hive_id, bee_category = DATASETS[key]
    metadata_bytes = metadata_path.read_bytes()
    metadata = json.loads(metadata_bytes)
    if metadata.get("id") != 20399470 or metadata["metadata"]["license"]["id"] != "cc-by-4.0":
        raise ValueError("Unexpected source record or license")
    file_info = next(item for item in metadata["files"] if item["key"] == filename)
    raw_bytes = (raw_dir / filename).read_bytes()
    md5 = hashlib.md5(raw_bytes).hexdigest()
    if md5 != SOURCE_MD5[filename] or file_info["checksum"] != f"md5:{md5}":
        raise ValueError(f"Source checksum mismatch for {filename}; original Zenodo bytes are required")
    if len(raw_bytes) != file_info["size"]:
        raise ValueError(f"Source size mismatch for {filename}")
    raw = pd.read_csv(io.BytesIO(raw_bytes))
    hourly, filter_stats = hourly_sensor(raw, sensor_id)
    selected, selection_stats = select_clean_block(hourly)
    rows = [{
        "timestamp": timestamp.tz_localize("UTC").isoformat(),
        "hive_id": hive_id,
        "weight_kg": float(row["weight"]),
        "temperature_c": float(row["ext_temperature"]),
        "event": "normal",
    } for timestamp, row in selected.iterrows()]
    # The upload contract requires 24 rows for an input sequence. This export
    # additionally requires 168 clean rows, the initial training data minimum.
    validate_series(rows, minimum=24)
    if len(rows) < 168:
        raise ValueError(f"Selected {hive_id} block has fewer than 168 clean hourly rows")
    manifest = {
        "hive_id": hive_id,
        "label": f"브라질 UFC · {'무침벌 센서 ' if bee_category.startswith('Meliponini') else 'Apis 벌통 '}{sensor_id}",
        "source_url": RECORD,
        "license": "CC-BY-4.0",
        "bee_category": bee_category,
        "source": {
            "record_url": RECORD, "doi": metadata["doi"],
            "title": metadata["metadata"]["title"],
            "creators": metadata["metadata"]["creators"],
            "license": "CC-BY-4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "filename": filename, "sensor_id": sensor_id,
            "download_url": f"https://zenodo.org/api/records/20399470/files/{filename}/content",
            "bytes": len(raw_bytes), "file_raw_rows": len(raw),
            "md5": md5, "md5_verified": True,
            "sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
        },
        "transformation": {
            "weight_kg": "Source weight, already kilograms; no rescaling",
            "temperature_c": "Source ext_temperature, external ambient degrees Celsius; not internal temperature",
            "physical_filter": "Finite 0 < weight <= 300 kg and -50 <= ext_temperature <= 60 C",
            "sentinel_filter": "Exclude rows where ext_temperature == 0 and ext_humidity == 0",
            "aggregation": "Median in [h, h+1), left-closed, labelled by completed hour h+1",
            "imputation": "None; empty hourly bins remain gaps",
            "event_rule": "Within each continuous segment abs(adjacent hourly weight difference) > 1 kg is colony_alert; a numeric flag, not a diagnosis",
            "event_exclusion": "Flagged hourly observations split clean blocks and are excluded before coverage selection",
        },
        "timestamp_policy": {
            "source_timezone": "Unspecified; source timestamps are naive",
            "output_offset": "+00:00",
            "assumption": "UTC offset attached only for deterministic bookkeeping; no source clock conversion or verified UTC claim",
        },
        "filters": filter_stats,
        "selection": selection_stats,
        "selected": {
            "rows": len(rows), "raw_rows": int(selected["raw_samples"].sum()),
            "start_hour_end": rows[0]["timestamp"], "end_hour_end": rows[-1]["timestamp"],
            "weight_kg_min": float(selected["weight"].min()),
            "weight_kg_max": float(selected["weight"].max()),
            "temperature_c_min": float(selected["ext_temperature"].min()),
            "temperature_c_max": float(selected["ext_temperature"].max()),
            "event_counts": {"normal": len(rows), "colony_alert": 0},
            "contract_validated": True, "minimum_clean_rows": 168,
        },
    }
    return {"manifest": manifest, "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/tmp/beeops-ufc-prepared.json"))
    parser.add_argument("--include-sensor1", action="store_true", help="Also prepare the 700-hour Meliponini sensor 1 block")
    args = parser.parse_args()
    try:
        if args.output.suffix.lower() != ".json":
            raise ValueError("Only a JSON intermediate output is permitted")
        if args.output.resolve().is_relative_to(RAW.resolve()):
            raise ValueError("Refusing to write into the raw source directory")
        keys = ["meliponini3", "apis1"] + (["meliponini1"] if args.include_sensor1 else [])
        result = {"schema_version": 1, "datasets": [prepare_dataset(key) for key in keys]}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({"output": str(args.output.resolve()), "datasets": [
            {"hive_id": item["manifest"]["hive_id"], **item["manifest"]["selected"]}
            for item in result["datasets"]
        ]}, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, KeyError, StopIteration) as exc:
        print(f"prepare_additional_ufc_data: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
