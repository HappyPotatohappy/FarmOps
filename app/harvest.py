"""Recommend the highest forecast-weight period without eligibility gates.

The v3 engineering heuristic groups points into observation-relative 24-hour
buckets (a one-hour forecast has one partial bucket), takes their medians, and
joins contiguous buckets within max(0.05 kg, 10% of the daily forecast range) of
the peak. A component wider than three buckets is reduced to the highest-mean
three-bucket window containing the peak, with earlier starts breaking ties.
These are recorded-clock intervals, not local calendar days. The rule ranks
forecast weight; it does not infer honey yield, maturity, or reserves.
"""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from math import isfinite
from statistics import fmean, median


RULE_VERSION = "forecast_peak_window_v3"
IDENTITY_KEYS = (
    "workspace_id", "hive_id", "snapshot_id", "forecast_id", "as_of",
    "generated_at", "horizon_hours", "requested_horizon_hours", "time_basis",
    "historical", "model", "validation",
)
DAY_HOURS = 24
MAX_WINDOW_DAYS = 3


def _finite(value):
    try:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value)
    except OverflowError:
        return False


def _hour(value):
    if not isinstance(value, str):
        raise ValueError("Missing timestamp")
    value = datetime.fromisoformat(value)
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Timezone required")
    value = value.astimezone(timezone.utc)
    if value.minute or value.second or value.microsecond:
        raise ValueError("Completed hourly timestamp required")
    return value


def _weight(value):
    return _finite(value) and 0 < value <= 300


def _observed_summary(observations, hive_id, as_of):
    """Optional descriptive evidence; missing history never gates a forecast."""
    summary = {"recent_growth_kg": None, "observed_daily_weight_kg": [], "clean_history_hours": 0}
    if not isinstance(observations, list) or len(observations) < 72:
        return summary
    recent = observations[-72:]
    for index, point in enumerate(recent):
        try:
            valid = (isinstance(point, dict) and point.get("hive_id") == hive_id
                     and _weight(point.get("weight_kg"))
                     and _hour(point.get("timestamp")) == as_of - timedelta(hours=71-index))
        except (ValueError, TypeError, OverflowError):
            valid = False
        if not valid:
            return summary
    daily = [median(p["weight_kg"] for p in recent[start:start+24]) for start in (0, 24, 48)]
    summary.update(recent_growth_kg=daily[-1]-daily[0], observed_daily_weight_kg=daily)
    for point in reversed(recent):
        if point.get("event") != "normal":
            break
        summary["clean_history_hours"] += 1
    return summary


def recommend_harvest(forecast: dict, observations: list[dict], species: str) -> dict:
    """Return a recommendation for every valid forecast, preserving its identity.

    Validation, calibration, species, observed trends and events are descriptive
    only. The only unavailable states reflect an absent or malformed forecast.
    Actual honey maturity and colony reserves remain field checks.
    """
    forecast = forecast if isinstance(forecast, dict) else {}
    result = {key: deepcopy(forecast.get(key)) for key in IDENTITY_KEYS}
    evidence = {
        "species": species, "aggregation_hours": DAY_HOURS,
        "required_history_hours": 0, "minimum_forecast_days": 0,
        "recent_growth_kg": None, "forecast_peak_kg": None, "forecast_decline_kg": None,
        "tolerance_kg": None, "daily_forecast": [], "noise_kg": None,
        "interval_half_width_kg": None, "observed_daily_weight_kg": [], "clean_history_hours": 0,
    }
    result.update(status="forecast_unavailable", candidate_window=None, reasons=[], evidence=evidence,
                  field_checks={"maturity": "unknown", "reserves": "unknown"},
                  actionability="none", rule_version=RULE_VERSION)

    def unavailable(reason):
        result["reasons"] = [reason]
        return result

    if forecast.get("status") != "ok":
        return unavailable("무게 예측을 불러오면 해당 예측에서 채밀 추천 기간을 계산합니다.")
    model, horizon = forecast.get("model"), forecast.get("horizon_hours")
    if (any(not isinstance(forecast.get(key), str) or not forecast[key]
            for key in ("workspace_id", "hive_id", "snapshot_id", "forecast_id"))
            or not isinstance(model, dict)
            or any(not model.get(key) for key in ("registry_id", "version", "run_id"))
            or not isinstance(horizon, int) or isinstance(horizon, bool) or not 1 <= horizon <= 168):
        return unavailable("예측의 벌통·모델·기간 정보를 확인할 수 없습니다. 예측을 다시 불러오세요.")
    try:
        as_of = _hour(forecast.get("as_of"))
    except (ValueError, TypeError, OverflowError):
        return unavailable("예측 기준시각을 확인할 수 없습니다. 예측을 다시 불러오세요.")
    trajectory = forecast.get("trajectory")
    if not isinstance(trajectory, list) or len(trajectory) != horizon:
        return unavailable("예측 기간의 시간별 무게가 완전하지 않습니다. 예측을 다시 불러오세요.")
    half_widths = []
    for index, point in enumerate(trajectory):
        try:
            valid = (isinstance(point, dict)
                     and _hour(point.get("timestamp")) == as_of + timedelta(hours=index+1)
                     and _weight(point.get("weight_kg")))
            if valid:
                lower, upper = point.get("lower_kg"), point.get("upper_kg")
                if lower is not None or upper is not None:
                    valid = (_finite(lower) and _finite(upper)
                             and lower <= point["weight_kg"] <= upper)
                    if valid:
                        half_widths.append(max(point["weight_kg"]-lower, upper-point["weight_kg"]))
        except (ValueError, TypeError, OverflowError):
            valid = False
        if not valid:
            return unavailable("시간별 예측 시각 또는 무게 값이 올바르지 않습니다. 예측을 다시 불러오세요.")

    evidence.update(_observed_summary(observations, forecast["hive_id"], as_of))
    validation = forecast.get("validation")
    if isinstance(validation, dict) and _finite(validation.get("noise_kg")):
        evidence["noise_kg"] = validation["noise_kg"]
    if half_widths:
        evidence["interval_half_width_kg"] = max(half_widths)
    for start in range(0, horizon, DAY_HOURS):
        day = trajectory[start:start+DAY_HOURS]
        evidence["daily_forecast"].append({
            "start": day[0]["timestamp"], "end": day[-1]["timestamp"],
            "weight_kg": median(p["weight_kg"] for p in day), "hours": len(day),
        })
    daily = evidence["daily_forecast"]
    weights = [day["weight_kg"] for day in daily]
    peak, minimum = max(weights), min(weights)
    tolerance = max(.05, .1*(peak-minimum))
    peak_index = weights.index(peak)
    left = right = peak_index
    while left > 0 and peak-weights[left-1] <= tolerance+1e-9:
        left -= 1
    while right+1 < len(weights) and peak-weights[right+1] <= tolerance+1e-9:
        right += 1
    window_days = min(MAX_WINDOW_DAYS, right-left+1)
    starts = range(max(left, peak_index-window_days+1), min(peak_index, right-window_days+1)+1)
    left = max(starts, key=lambda start: (fmean(weights[start:start+window_days]), -start))
    right = left+window_days-1
    evidence.update(forecast_peak_kg=peak, forecast_decline_kg=peak-weights[-1],
                    tolerance_kg=tolerance, forecast_daily_range_kg=peak-minimum)
    result.update(status="inspection_window", actionability="requires_field_check",
                  candidate_window={"start": daily[left]["start"], "end": daily[right]["end"]})
    if len(daily) > 1 and peak-minimum <= tolerance:
        reason = "예측 무게가 비슷하게 유지되는 구간을 채밀 추천 기간으로 표시합니다."
    elif right == len(daily)-1 and len(daily) > 1 and weights[-1]-weights[0] > tolerance:
        reason = "예측 무게가 높은 마지막 구간을 채밀 추천 기간으로 표시합니다."
    else:
        reason = "예측 무게가 가장 높은 구간을 채밀 추천 기간으로 표시합니다."
    result["reasons"] = [reason, "채밀 전 꿀의 숙성 상태와 군체에 남길 비축량을 현장에서 확인하세요."]
    return result
