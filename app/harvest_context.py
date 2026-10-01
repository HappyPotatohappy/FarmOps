"""Optional observations and user work hours; never change a harvest forecast.

Recorded-clock reports carry a placeholder UTC offset. A selected IANA zone
reinterprets that clock; it does not convert the placeholder as an instant.
Work-hour intersections are evaluated on actual UTC minutes so daylight saving
transitions neither invent nonexistent minutes nor discard repeated minutes.
"""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from math import isfinite
import re
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


SOURCES = (
    {"id": "fao_honey", "title": "FAO · Value-added products from beekeeping, 2.6.8–2.6.9",
     "url": "https://www.fao.org/4/w0076e/w0076e05.htm", "published": "1996"},
    {"id": "osu_harvest", "title": "Oklahoma State University · Honey Harvest Methods",
     "url": "https://extension.okstate.edu/fact-sheets/beekeeping-honey-harvest-methods-costs-and-breakeven-calculations",
     "published": "2022-08"},
    {"id": "rda_reserves", "title": "농촌진흥청 · 양봉산업 안정화를 위한 현장교육 교재, 18–20쪽",
     "url": "https://agri.jeju.go.kr/files/board/eba3e7e7-0208-41bb-b1ca-9fbe628abacd.pdf", "published": "2022"},
    {"id": "aces_feeding", "title": "Alabama Extension · Supplemental Feeding for Honey Bees",
     "url": "https://www.aces.edu/blog/topics/bees-pollinators/supplemental-feeding-for-honey-bees/",
     "published": "2025-01-23"},
    {"id": "unido_weather", "title": "UNIDO · Quality Management when Harvesting and Processing, 12–13쪽",
     "url": "https://downloads.unido.org/ot/39/43/39436064/Guide_2_v_1.2-lr.pdf", "published": "2025-10-13"},
)
NUMERIC_FIELDS = ("honey_moisture_pct", "capped_ratio_pct")
BOOLEAN_FIELDS = ("reserves_confirmed", "feeding_separated", "treatment_checked")
PREFERENCE_FIELDS = {*NUMERIC_FIELDS, *BOOLEAN_FIELDS, "recorded_timezone", "working_hours", "measured_at"}
ABSOLUTE_TIME_BASES = {"utc", "UTC", "utc_timestamp", "absolute_timestamp"}


def _timestamp(value):
    if not isinstance(value, str):
        raise ValueError("시각은 UTC 오프셋을 포함하는 ISO 8601 문자열이어야 합니다.")
    try:
        result = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError("올바른 ISO 8601 시각을 입력하세요.") from error
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("검사 시각에는 UTC 오프셋이 필요합니다.")
    return result


def _validate(preferences):
    if not isinstance(preferences, dict) or set(preferences) - PREFERENCE_FIELDS:
        raise ValueError("지원하는 채밀 확인 설정을 객체로 입력하세요.")
    for name in NUMERIC_FIELDS:
        value = preferences.get(name)
        if value is None:
            continue
        try:
            valid = (isinstance(value, (int, float)) and not isinstance(value, bool)
                     and isfinite(value) and 0 <= value <= 100)
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError(f"{name}은 0~100 사이의 유한한 숫자여야 합니다.")
    for name in BOOLEAN_FIELDS:
        if preferences.get(name) is not None and not isinstance(preferences[name], bool):
            raise ValueError(f"{name}은 true, false 또는 null이어야 합니다.")
    zone = None
    if preferences.get("recorded_timezone") is not None:
        try:
            if not isinstance(preferences["recorded_timezone"], str):
                raise ValueError()
            zone = ZoneInfo(preferences["recorded_timezone"])
        except (ValueError, TypeError, ZoneInfoNotFoundError) as error:
            raise ValueError("원본 데이터의 실제 IANA 시간대를 선택하세요.") from error
    hours = preferences.get("working_hours")
    if hours is not None:
        if not isinstance(hours, dict) or set(hours) != {"start", "end"}:
            raise ValueError("작업시간은 start와 end를 포함해야 합니다.")
        for value in hours.values():
            if not isinstance(value, str) or not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", value):
                raise ValueError("작업시간을 HH:MM 형식으로 입력하세요.")
        if hours["start"] == hours["end"]:
            raise ValueError("작업 시작과 종료 시각은 달라야 합니다.")
    measured_at = preferences.get("measured_at")
    if measured_at is not None:
        _timestamp(measured_at)
    return zone, hours


def _recorded_instant(value, zone, absolute):
    parsed = _timestamp(value)
    if absolute:
        return parsed.astimezone(timezone.utc)
    wall = parsed.replace(tzinfo=None)
    candidates = set()
    for fold in (0, 1):
        instant = wall.replace(tzinfo=zone, fold=fold).astimezone(timezone.utc)
        if instant.astimezone(zone).replace(tzinfo=None) == wall:
            candidates.add(instant)
    if len(candidates) != 1:
        raise ValueError("원본 기록 시각이 일광절약시간 전환으로 중복되거나 존재하지 않습니다.")
    return candidates.pop()


def _work_windows(start, end, zone, hours):
    def in_hours(instant):
        clock = instant.astimezone(zone).strftime("%H:%M")
        if hours["start"] < hours["end"]:
            return hours["start"] <= clock < hours["end"]
        return clock >= hours["start"] or clock < hours["end"]

    def window(left, right):
        return {"start": left.astimezone(zone).isoformat(), "end": right.astimezone(zone).isoformat(),
                "timezone": zone.key, "duration_minutes": (right-left).total_seconds()/60}

    if start == end:
        return [window(start, end)] if in_hours(start) else []
    windows, opened, cursor = [], None, start
    while cursor < end:
        eligible = in_hours(cursor)
        if eligible and opened is None:
            opened = cursor
        elif not eligible and opened is not None:
            windows.append(window(opened, cursor)); opened = None
        # All report boundaries are minute-aligned; handle fractional first
        # minutes defensively without skipping a local HH:MM transition.
        cursor = min(cursor.replace(second=0, microsecond=0) + timedelta(minutes=1), end)
    if opened is not None:
        windows.append(window(opened, end))
    return windows


def _field_checks(report, preferences):
    evidence = report.get("evidence")
    species = evidence.get("species") if isinstance(evidence, dict) else None
    applicable = species == "apis"
    checks = []
    definitions = (
        ("honey_moisture_pct", "꿀 수분 실측", {"operator": "<=", "value": 18, "unit": "%", "source_id": "fao_honey"},
         "18%는 품질 관리 참고 목표이며 법정 기준이나 품질 보증이 아닙니다. 굴절계로 측정하세요."),
        ("capped_ratio_pct", "봉개율 확인", {"operator": ">=", "value": 80, "unit": "%", "source_id": "osu_harvest"},
         "80%는 해외 양봉 안내의 참고값입니다. 봉개율만으로 수분이나 숙성이 확정되지 않습니다."),
        ("reserves_confirmed", "채밀 후 비축 먹이 확인", {"source_id": "rda_reserves"},
         "지역·계절·군세에 맞춰 남길 먹이를 현장에서 확인하세요. 벌통 전체 무게는 꿀의 양과 다릅니다."),
        ("feeding_separated", "급이 소비 분리 확인", {"source_id": "aces_feeding"},
         "급이 이력과 설탕액이 저장된 소비의 분리 여부를 확인하세요."),
        ("treatment_checked", "약제 사용·채밀 제한 확인", {"source_id": "fao_honey"},
         "사용한 제품의 설명서에 따른 채밀 제한과 사용 기록을 확인하세요. 공통 휴약일수는 적용하지 않습니다."),
    )
    for key, label, reference, note in definitions:
        value = preferences.get(key)
        status = "unknown"
        if value is not None:
            if key in BOOLEAN_FIELDS:
                status = "confirmed" if value else "attention"
            elif not applicable:
                status = "observed"
            else:
                met = value <= 18 if key == "honey_moisture_pct" else value >= 80
                status = "reference_met" if met else "reference_not_met"
        checks.append({"key": key, "label": label, "status": status, "value": value,
                       "reference": reference, "note": note, "measured_at": preferences.get("measured_at")})
    return checks, applicable


def build_harvest_context(report: dict, preferences: dict) -> dict:
    """Return an independent supplement. Never mutate or gate the input report.

    Unknown preferences remain unknown. Any malformed preference raises
    ValueError; malformed source timestamps produce explanatory empty schedules.
    ``recorded_timezone`` explicitly identifies the clock's original IANA zone
    and is also used for the user's optional working hours and display.
    """
    zone, hours = _validate(preferences)
    if not isinstance(report, dict):
        raise ValueError("채밀 추천 보고서는 객체여야 합니다.")
    field_checks, applicable = _field_checks(report, preferences)
    timing = {"status": "timezone_required", "recorded_timezone": zone.key if zone else None,
              "as_of": None, "measured_at": preferences.get("measured_at"),
              "interpretation": "원본 기록 시각의 실제 시간대는 명시적으로 선택해야 합니다."}
    notes = ["작업 시간 후보는 예측 구간과 사용자가 정한 작업시간의 교집합이며, 생물학적 채밀 적기 판정이 아닙니다.",
             "현장 확인 항목은 무게 예측이나 기존 채밀 추천 구간을 차단하지 않습니다.",
             "날씨 예보는 이 계산에 포함하지 않았습니다. 작업 전 비·고습 여부를 확인하세요."]
    measured_at = preferences.get("measured_at")
    notes.append(f"검사 기록 시각: {measured_at}" if measured_at else "검사 시각 미기록: 현장 확인값의 검사 시각을 함께 기록하세요.")
    if report.get("historical"):
        notes.append("과거 관측 데이터의 예측입니다. 입력한 현장 확인값이 당시 상태를 소급 증명하지는 않습니다.")
    if not applicable:
        notes.append("벌종 적용 범위: 수분·봉개율 참고값은 일반 양봉 자료 기반입니다. 이 벌종에는 자동 적합 판정을 적용하지 않았습니다.")
    result = {"field_checks": field_checks, "work_windows": [], "timing_basis": timing,
              "sources": deepcopy(list(SOURCES)), "notes": notes}
    if zone is None:
        notes.append("원본 데이터의 실제 시간대를 선택하면 같은 지역 시각으로 작업 시간을 표시합니다.")
        return result
    absolute = report.get("time_basis") in ABSOLUTE_TIME_BASES
    timing["interpretation"] = ("명시된 절대 시각을 선택 지역 시각으로 변환했습니다." if absolute else
        "기록에 붙은 +00:00을 UTC로 추정하지 않고 원본 시계 시각을 선택 시간대로 해석했습니다.")
    try:
        as_of = _recorded_instant(report.get("as_of"), zone, absolute)
        timing["as_of"] = as_of.astimezone(zone).isoformat()
        notes.append(f"관측 기준 시각: {timing['as_of']} ({zone.key})")
    except (ValueError, TypeError, OverflowError):
        timing["status"] = "source_time_unresolved"
        notes.append("관측 기준 시각을 고유한 실제 시각으로 해석할 수 없어 작업 시간을 만들지 않았습니다.")
        return result
    candidate = report.get("candidate_window")
    if not isinstance(candidate, dict) or not candidate.get("start") or not candidate.get("end"):
        timing["status"] = "candidate_unavailable"
        notes.append("예측 후보 구간이 없어 작업 시간 후보를 계산하지 않았습니다.")
        return result
    try:
        start = _recorded_instant(candidate["start"], zone, absolute)
        end = _recorded_instant(candidate["end"], zone, absolute)
        # Forecasts currently support at most 168 hours. Permit a DST transition
        # without accepting an unbounded interval from a malformed report.
        if end < start or end-start > timedelta(hours=192):
            raise ValueError("예측 후보 구간의 범위가 올바르지 않습니다.")
    except (ValueError, TypeError, OverflowError):
        timing["status"] = "source_time_unresolved"
        notes.append("후보 구간의 시각이 잘못되었거나 일광절약시간으로 모호합니다. 기존 예측 구간은 유지됩니다.")
        return result
    if hours is None:
        timing["status"] = "working_hours_required"
        notes.append("작업 시작·종료 시각을 입력하면 예측 후보 구간 안의 작업 시간을 계산합니다.")
        return result
    result["work_windows"] = _work_windows(start, end, zone, hours)
    timing["status"] = "ready" if result["work_windows"] else "no_overlap"
    if not result["work_windows"]:
        notes.append("입력한 작업시간과 예측 후보 구간이 겹치지 않습니다. 기존 예측 추천은 유지됩니다.")
    if start == end:
        notes.append("예측 후보는 한 시점입니다. 이를 넘어서는 작업 기간은 추정하지 않았습니다.")
    return result
