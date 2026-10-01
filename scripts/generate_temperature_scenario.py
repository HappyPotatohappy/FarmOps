#!/usr/bin/env python3
"""Build deterministic SYNTHETIC hourly CSV fixtures; never use real hive data.

The toy temperature/weight relationship exists to exercise an operations workflow.
It is not a biological model, and this generator never computes model accuracy.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import random
import re
from statistics import mean
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data/simulations/temperature_drift"
FIELDS = ["timestamp", "hive_id", "weight_kg", "temperature_c", "event"]
SEED = 20261001
START = datetime(2026, 7, 1, tzinfo=timezone.utc)
BASELINE_HOURS = 336
HEAT_HOURS = 168
FOLLOWUP_HOURS = 72
RAMP_HOURS = 12
HEAT_SHIFT_C = 8.0


USAGE = """# 외부 온도 변화 실습 데이터

**SYNTHETIC · 합성 데이터**. 온도 변화 감지, 실제 앙상블 재학습, 후속 관측 연결을 재현하기 위한 자료입니다. 공개·현장 실측이나 검증된 생물학 모델이 아니며 모델 정확도·재학습 성능 향상을 미리 정하지 않았습니다.

## 기본 고온 실습: 첫 실행과 반복

최종본을 실행하고 `http://localhost:8012`의 **데이터 → 온도 변화 실습 → 02 · 고온 기간 → 파일 불러오기**를 선택합니다. 기본 버튼은 서버의 검증된 `TEMP-DRIFT-PRACTICE` 자료를 사용합니다. **정상 기준 336행 + 고온 168행 = 504행**을 미리보기로 함께 준비하므로 기준 파일을 먼저 저장하지 않아도 됩니다. 고온 CSV 파일 자체는 168행입니다.

1. 대상 벌통과 신규·중복·충돌 행 수를 확인한 뒤 **저장하고 재학습**을 누릅니다.
2. 접수되면 해당 TEMP 벌통이 바로 선택되고 대시보드로 이동합니다. 관측 저장·준비 상태에서 실제 대기·학습·등록·적용 단계로 이어집니다.
3. 실제 완료한 학습 회차와 최종 작업 결과를 확인합니다. 8/8회 뒤에도 결합 비율 학습·평가·등록이 남을 수 있습니다. 실패하면 실제 오류를 표시하며 성공이나 적용을 미리 보장하지 않습니다.
4. 결과는 **확인**을 누를 때까지 남고 화면 이동·새로고침 뒤에도 복원됩니다. 빠르게 끝났다는 이유로 알림이 먼저 사라지지 않습니다.
5. 완료 후 **고온 파일을 다시 불러와 새 미리보기를 저장**하면 현재 모델에서 실제 재학습을 다시 수행합니다. 기존 504행은 중복으로 보존합니다. 같은 미리보기 저장 요청의 재전송은 같은 작업을 반환하며, 동일 실습이 이미 대기·실행 중이면 그 실제 작업에 연결합니다.

반복 실행은 고정 합성 자료를 다시 사용하는 명시적 실습입니다. LSTM 파라미터와 앙상블 결합 비율을 학습하고 TiRex-2 체크포인트는 유지합니다. 같은 실행 안에서 학습·비율 보정·최종 평가 구간을 나누어도 과거 실행이 같은 자료를 사용했을 수 있습니다. 결과의 `data_reused=true`, `independent_future_evaluation=false`는 이를 뜻합니다. 평가 수치를 독립적인 새 미래 정확도나 7일 성능의 증거로 설명하지 않습니다.

## CSV를 직접 올릴 때와 파일 구성

이 묶음의 실험군 ID는 `TEMP-DRIFT-DEMO`입니다. 아래 순서는 **CSV 파일 선택을 통한 일반 업로드**에 적용합니다. 이 경우 먼저 기준 파일을 저장하고 다음 고온 파일을 추가합니다. 일반 파일 업로드나 파일 이름 변경만으로 기본 버튼의 기준 복구·명시적 반복 재학습 예외가 적용되지는 않습니다. 다른 ID로 생성한 자료도 일반 업로드를 사용합니다.

| 순서 | 파일 | 행 수 | UTC 기간 | 내용 |
|---|---|---:|---|---|
| 1 | `01_baseline.csv` | 336 | 2026-07-01 00:00 ~ 07-14 23:00 | 약 24°C의 일주기 정상 기준 |
| 2 | `02_heatwave.csv` | 168 | 2026-07-15 00:00 ~ 07-21 23:00 | 처음 12시간에 +8°C까지 상승 후 유지 |
| 3 | `03_followup.csv` | 72 | 2026-07-22 00:00 ~ 07-24 23:00 | 고온 상태가 유지되는 후속 관측 |

기준 파일을 따로 저장한 뒤 기본 고온 버튼을 사용해도 이미 있는 기준 행은 중복 처리합니다. 기본 버튼의 누락 기준 복구는 파일 해시와 기존 관측을 검증한 동일 실습에 한정됩니다. 충돌하는 값을 덮어쓰거나 다른 관측 원장을 되돌리지 않습니다.

후속 파일은 재학습 결과와 그 이후 예측 생성을 확인한 뒤 저장합니다. 저장된 예측의 목표 시각에 맞는 합성 관측을 실제값 역할로 연결할 수 있지만, 72행을 추가했다고 72개의 사전 예측·평가 쌍이 자동 생성되지는 않습니다. 대시보드의 과거 비교는 현재 모델로 다시 계산한 예측이며 당시 미리 생성한 예측의 운영 성능과 구분합니다. 합성 관측은 원장에서 실제값 역할을 하더라도 현장 실측이 되지 않습니다.

## 정상 대조군

별도 벌통 `TEMP-CONTROL-DEMO`에 `control/01_baseline.csv` 336행, `control/02_normal.csv` 168행을 일반 CSV 업로드로 차례로 저장합니다. 고온 가산 외에는 같은 시각·일주기·난수 배경을 사용합니다. 기준 구간의 온도·무게는 실험군과 같습니다. 정상 대조군은 고온 실습의 명시적 재실행 버튼과 구분해 일반 온도 감시 결과를 확인합니다.

## 생성 가정

- 시간은 합성 시나리오의 UTC 정시이며 정확히 1시간 간격입니다. 관측을 보간하거나 기존 파일에서 복사하지 않았습니다.
- 온도는 `24 + 3 × sin(2π × (hour − 9) / 24) + 고온 가산값 + 잡음`입니다. 온도 잡음은 ±0.18°C 균등분포입니다.
- 고온 가산값은 기준 구간 이후 12시간의 smoothstep 전환으로 0에서 8°C까지 올라가고 후속 구간까지 유지됩니다.
- 고온 부하는 `max(현재 온도 − 28, 0)`의 지수 이동 평균입니다. 매시간 현재값에 1/6, 직전 부하에 5/6의 비중을 적용합니다. 미래값은 사용하지 않습니다.
- 무게는 50kg에서 시작하는 누적 질량과 진폭 0.25kg의 24시간 주기를 합합니다. 누적 질량의 시간당 변화는 `0.010 − 0.007 × 고온 부하 + 잡음`kg입니다. 무게 잡음은 ±0.003kg 균등분포입니다. 고온에서는 정상보다 일별 순증가가 줄고 감소할 수 있습니다. 이 계수는 실습용 가정입니다.
- 난수 seed는 `20261001`입니다. 무게는 소수점 6자리, 온도는 4자리로 기록합니다.
- 모든 `event`는 `normal`입니다. 채밀·급이·질병·센서 이상 사건을 임의 표시하지 않았습니다. 온도 변화는 수치 분포로 확인합니다.
- 정확히 `timestamp,hive_id,weight_kg,temperature_c,event` 다섯 열을 사용합니다. 단계 사이 중복·누락·값 덮어쓰기는 없습니다. 같은 파일 재업로드는 기존 업로드 규칙에 따라 중복으로 처리됩니다.

파일별 SHA-256, 범위와 생성 설정은 `manifest.json`에 있습니다. ZIP에는 이 안내와 CSV 5개 및 manifest가 들어 있습니다.

## 재생성

최종본 루트에서 다음 명령을 사용합니다. `python`이 없으면 `python3`를 사용합니다. 이 명령은 현재 묶음의 벌통 ID와 출력 위치를 명시합니다.

```sh
python scripts/generate_temperature_scenario.py --output-dir data/simulations/temperature_practice --drift-hive TEMP-DRIFT-DEMO --control-hive TEMP-CONTROL-DEMO
```

비교용 새 폴더를 만들려면 `--output-dir`만 빈 경로로 바꿉니다. 같은 내용의 파일은 유지하고 다른 내용의 기존 파일은 덮어쓰지 않고 중단합니다. ZIP은 출력 폴더 옆에 `<폴더명>.zip`으로 생성합니다. 생성기는 관측 원장이나 모델을 수정하지 않습니다. 기본 버튼의 신뢰한 실습 경로는 최종본의 고정 PRACTICE 자료만 사용하며 다른 출력 폴더를 자동 등록하지 않습니다.

상세 흐름은 프로젝트의 `docs/온도_드리프트_실습.md`에서 확인합니다.
"""


def generate_rows(*, heatwave: bool, hours: int, hive_id: str) -> list[dict]:
    """Advance state chronologically using current/past temperature only."""
    rng = random.Random(SEED)
    mass = 50.0
    heat_load = 0.0
    rows = []
    for index in range(hours):
        timestamp = START + timedelta(hours=index)
        hour = timestamp.hour
        progress = min(1.0, max(0.0, (index - BASELINE_HOURS + 1) / RAMP_HOURS)) if heatwave else 0.0
        offset = HEAT_SHIFT_C * progress * progress * (3 - 2 * progress)
        temperature = 24 + 3 * math.sin(2 * math.pi * (hour - 9) / 24) + offset + rng.uniform(-.18, .18)
        weight_noise = rng.uniform(-.003, .003)
        heat_load += (max(temperature - 28, 0) - heat_load) / 6
        if index:
            mass += .010 - .007 * heat_load + weight_noise
        weight = mass + .25 * math.cos(2 * math.pi * (hour - 3) / 24)
        if not (math.isfinite(weight) and 0 < weight <= 300 and math.isfinite(temperature) and -50 <= temperature <= 60):
            raise ValueError("Synthetic generation exceeded the hourly observation contract")
        rows.append({
            "timestamp": timestamp.isoformat(),
            "hive_id": hive_id,
            "weight_kg": round(weight, 6),
            "temperature_c": round(temperature, 4),
            "event": "normal",
        })
    return rows


def csv_payload(rows: list[dict]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def file_metadata(rows: list[dict], payload: bytes) -> dict:
    temperatures = [row["temperature_c"] for row in rows]
    return {
        "hive_id": rows[0]["hive_id"],
        "rows": len(rows),
        "start": rows[0]["timestamp"],
        "end": rows[-1]["timestamp"],
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "temperature_mean_c": round(mean(temperatures), 6),
        "temperature_min_c": min(temperatures),
        "temperature_max_c": max(temperatures),
        "weight_start_kg": rows[0]["weight_kg"],
        "weight_end_kg": rows[-1]["weight_kg"],
        "net_weight_change_kg": round(rows[-1]["weight_kg"] - rows[0]["weight_kg"], 6),
        "mean_24h_weight_change_kg": round(mean(rows[i]["weight_kg"] - rows[i - 24]["weight_kg"] for i in range(24, len(rows))), 6),
    }


def bundle_payloads(*, drift_hive: str = "TEMP-DRIFT-DEMO", control_hive: str = "TEMP-CONTROL-DEMO") -> tuple[dict[str, bytes], dict]:
    if any(not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", hive) for hive in (drift_hive, control_hive)):
        raise ValueError("Synthetic hive IDs must match [A-Za-z0-9_-]{1,40}")
    if drift_hive == control_hive:
        raise ValueError("Drift and control hive IDs must be different")
    heat_end = BASELINE_HOURS + HEAT_HOURS
    drift = generate_rows(heatwave=True, hours=heat_end + FOLLOWUP_HOURS, hive_id=drift_hive)
    control = generate_rows(heatwave=False, hours=heat_end, hive_id=control_hive)
    stages = {
        "01_baseline.csv": drift[:BASELINE_HOURS],
        "02_heatwave.csv": drift[BASELINE_HOURS:heat_end],
        "03_followup.csv": drift[heat_end:],
        "control/01_baseline.csv": control[:BASELINE_HOURS],
        "control/02_normal.csv": control[BASELINE_HOURS:],
    }
    payloads = {name: csv_payload(rows) for name, rows in stages.items()}
    manifest = {
        "schema_version": 1,
        "scenario_id": "synthetic_temperature_drift_v1",
        "data_kind": "synthetic",
        "source": "SYNTHETIC: deterministic generator; no observed/public hive data",
        "purpose": "Temperature drift, retraining workflow, and delayed actual-join demonstration",
        "generator": "scripts/generate_temperature_scenario.py",
        "seed": SEED,
        "timezone": "UTC (chosen synthetic timeline)",
        "columns": FIELDS,
        "method": {
            "baseline_hours": BASELINE_HOURS,
            "heat_hours": HEAT_HOURS,
            "followup_hours": FOLLOWUP_HOURS,
            "temperature_center_c": 24,
            "temperature_daily_amplitude_c": 3,
            "temperature_noise_uniform_c": [-.18, .18],
            "heat_shift_c": HEAT_SHIFT_C,
            "heat_ramp_hours": RAMP_HOURS,
            "heat_ramp": "smoothstep: 3*x^2 - 2*x^3",
            "heat_load_threshold_c": 28,
            "heat_load_ema_alpha": 1 / 6,
            "starting_mass_kg": 50,
            "weight_daily_amplitude_kg": .25,
            "normal_mass_gain_kg_per_hour": .010,
            "heat_mass_penalty_kg_per_hour_per_c": .007,
            "mass_noise_uniform_kg_per_hour": [-.003, .003],
            "causal": True,
            "interpolation": False,
            "event_labels": "normal only; no synthetic operational/biological event labels",
        },
        "model_metrics": None,
        "limitations": [
            "Toy engineering scenario, not a validated biological or disease model",
            "No forecast accuracy or retraining improvement is claimed",
            "Follow-up observations remain synthetic even when joined as actuals in the demonstration ledger",
            "Upload stages separately in order; create predictions before uploading their follow-up targets",
        ],
        "files": {name: file_metadata(rows, payloads[name]) for name, rows in stages.items()},
    }
    payloads["manifest.json"] = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    payloads["usage.md"] = USAGE.replace("TEMP-DRIFT-DEMO", drift_hive).replace("TEMP-CONTROL-DEMO", control_hive).encode("utf-8")
    return payloads, manifest


def build_bundle(output_dir: Path | str = DEFAULT_OUTPUT, *, drift_hive: str = "TEMP-DRIFT-DEMO", control_hive: str = "TEMP-CONTROL-DEMO") -> dict:
    """Write new files, retain byte-identical files, reject any other overwrite."""
    output_dir = Path(output_dir)
    payloads, manifest = bundle_payloads(drift_hive=drift_hive, control_hive=control_hive)
    zipped = io.BytesIO()
    with zipfile.ZipFile(zipped, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in payloads.items():
            entry = zipfile.ZipInfo(name, date_time=(2026, 7, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.create_system = 3
            entry.external_attr = 0o100644 << 16
            archive.writestr(entry, payload)
    planned = {output_dir / name: payload for name, payload in payloads.items()}
    archive_path = output_dir.parent / f"{output_dir.name}.zip"
    planned[archive_path] = zipped.getvalue()
    # Check the entire bundle before creating anything so a conflict preserves it.
    for path, payload in planned.items():
        if path.exists() and (not path.is_file() or path.read_bytes() != payload):
            raise FileExistsError(f"Refusing to overwrite existing different content: {path}")
    for path, payload in planned.items():
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(payload)
    return {"output_dir": str(output_dir), "archive": str(archive_path), "manifest": manifest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--drift-hive", default="TEMP-DRIFT-DEMO")
    parser.add_argument("--control-hive", default="TEMP-CONTROL-DEMO")
    args = parser.parse_args()
    try:
        result = build_bundle(args.output_dir, drift_hive=args.drift_hive, control_hive=args.control_hive)
    except (FileExistsError, ValueError) as exc:
        parser.exit(2, f"{exc}\n")
    print(json.dumps({"data_kind": "synthetic", "output_dir": result["output_dir"], "archive": result["archive"], "csv_files": len(result["manifest"]["files"])}))


if __name__ == "__main__":
    main()
