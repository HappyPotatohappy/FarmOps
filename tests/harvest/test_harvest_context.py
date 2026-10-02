"""Field information supplements a forecast; it never becomes a forecast gate."""
from copy import deepcopy
from datetime import datetime

import pytest

from app.harvest_context import build_harvest_context


def report(start="2025-05-01T06:00:00+00:00", end="2025-05-02T12:00:00+00:00", species="apis"):
    return {
        "status": "inspection_window", "candidate_window": {"start": start, "end": end},
        "time_basis": "recorded_timestamp", "as_of": "2025-05-01T00:00:00+00:00",
        "historical": True, "evidence": {"species": species},
    }


def checks(result):
    return {check["key"]: check for check in result["field_checks"]}


def test_absent_preferences_preserve_candidate_and_keep_field_checks_unknown():
    original = report(); before = deepcopy(original)
    result = build_harvest_context(original, {})
    assert original == before
    assert result["work_windows"] == []
    assert result["timing_basis"]["status"] == "timezone_required"
    assert {check["status"] for check in result["field_checks"]} == {"unknown"}
    assert "candidate_window" not in result
    assert result["sources"] and result["notes"]


def test_explicit_timezone_reinterprets_recorded_clock_without_nine_hour_shift():
    result = build_harvest_context(report(), {
        "recorded_timezone": "Asia/Seoul", "working_hours": {"start": "08:30", "end": "11:00"},
    })
    assert result["timing_basis"]["as_of"] == "2025-05-01T00:00:00+09:00"
    assert result["work_windows"] == [
        {"start": "2025-05-01T08:30:00+09:00", "end": "2025-05-01T11:00:00+09:00",
         "timezone": "Asia/Seoul", "duration_minutes": 150},
        {"start": "2025-05-02T08:30:00+09:00", "end": "2025-05-02T11:00:00+09:00",
         "timezone": "Asia/Seoul", "duration_minutes": 150},
    ]


def test_working_hours_are_not_invented_when_only_timezone_is_selected():
    result = build_harvest_context(report(), {"recorded_timezone": "Asia/Seoul"})
    assert result["work_windows"] == []
    assert result["timing_basis"]["status"] == "working_hours_required"


def test_overnight_hours_clip_to_original_candidate_window():
    original = report("2025-05-01T23:00:00+00:00", "2025-05-02T02:00:00+00:00")
    result = build_harvest_context(original, {"recorded_timezone": "Asia/Seoul",
        "working_hours": {"start": "22:00", "end": "03:00"}})
    assert result["work_windows"] == [{"start": "2025-05-01T23:00:00+09:00",
        "end": "2025-05-02T02:00:00+09:00", "timezone": "Asia/Seoul", "duration_minutes": 180}]


def test_nonoverlapping_work_hours_leave_candidate_intact():
    original = report("2025-05-01T06:00:00+00:00", "2025-05-01T07:00:00+00:00")
    before = deepcopy(original)
    result = build_harvest_context(original, {"recorded_timezone": "Asia/Seoul",
        "working_hours": {"start": "08:00", "end": "09:00"}})
    assert result["work_windows"] == []
    assert result["timing_basis"]["status"] == "no_overlap"
    assert original == before


def test_one_hour_forecast_point_does_not_claim_a_longer_work_window():
    result = build_harvest_context(report("2025-05-01T09:00:00+00:00", "2025-05-01T09:00:00+00:00"),
        {"recorded_timezone": "Asia/Seoul", "working_hours": {"start": "08:00", "end": "10:00"}})
    assert result["work_windows"][0]["duration_minutes"] == 0
    assert result["work_windows"][0]["start"] == result["work_windows"][0]["end"]


def test_dst_spring_gap_and_autumn_repeat_use_actual_elapsed_time():
    preferences = {"recorded_timezone": "America/New_York",
                   "working_hours": {"start": "01:30", "end": "03:30"}}
    spring = build_harvest_context(report("2025-03-09T00:00:00+00:00", "2025-03-09T05:00:00+00:00"), preferences)
    assert spring["work_windows"] == [{"start": "2025-03-09T01:30:00-05:00",
        "end": "2025-03-09T03:30:00-04:00", "timezone": "America/New_York", "duration_minutes": 60}]
    autumn = build_harvest_context(report("2025-11-02T00:00:00+00:00", "2025-11-02T04:00:00+00:00"), preferences)
    assert sum(item["duration_minutes"] for item in autumn["work_windows"]) == 150
    for item in autumn["work_windows"]:
        assert datetime.fromisoformat(item["end"]) >= datetime.fromisoformat(item["start"])


@pytest.mark.parametrize("start,end", [
    ("2025-03-09T02:30:00+00:00", "2025-03-09T05:00:00+00:00"),
    ("2025-11-02T01:30:00+00:00", "2025-11-02T04:00:00+00:00"),
])
def test_ambiguous_or_nonexistent_source_clock_does_not_guess_an_instant(start, end):
    result = build_harvest_context(report(start, end), {"recorded_timezone": "America/New_York",
        "working_hours": {"start": "01:00", "end": "05:00"}})
    assert result["work_windows"] == []
    assert result["timing_basis"]["status"] == "source_time_unresolved"


def test_inspection_results_remain_descriptive_even_when_reference_not_met():
    original = report(); before = deepcopy(original)
    result = build_harvest_context(original, {"honey_moisture_pct": 22, "capped_ratio_pct": 40,
        "reserves_confirmed": False, "feeding_separated": False, "treatment_checked": True,
        "measured_at": "2025-05-01T09:10:00+09:00"})
    values = checks(result)
    assert values["honey_moisture_pct"]["status"] == "reference_not_met"
    assert values["capped_ratio_pct"]["status"] == "reference_not_met"
    assert values["reserves_confirmed"]["status"] == "attention"
    assert values["feeding_separated"]["status"] == "attention"
    assert values["treatment_checked"]["status"] == "confirmed"
    assert result["timing_basis"]["measured_at"] == "2025-05-01T09:10:00+09:00"
    assert original == before


def test_reference_boundaries_and_non_apis_applicability_are_explicit():
    preferences = {"honey_moisture_pct": 18, "capped_ratio_pct": 80}
    result = build_harvest_context(report(), preferences)
    assert checks(result)["honey_moisture_pct"]["status"] == "reference_met"
    assert checks(result)["capped_ratio_pct"]["status"] == "reference_met"
    non_apis = build_harvest_context(report(species="meliponini"), preferences)
    assert checks(non_apis)["honey_moisture_pct"]["status"] == "observed"
    assert checks(non_apis)["capped_ratio_pct"]["status"] == "observed"
    assert any("벌종" in note for note in non_apis["notes"])


@pytest.mark.parametrize("preferences", [
    None, [], {"unrecognized": True}, {"timezone": "Asia/Seoul"},
    {"recorded_timezone": "not/a/timezone"}, {"recorded_timezone": ""},
    {"working_hours": {"start": "24:00", "end": "08:00"}},
    {"working_hours": {"start": "8:00", "end": "09:00"}},
    {"working_hours": {"start": "08:00", "end": "08:00"}},
    {"working_hours": {"start": "08:00"}}, {"working_hours": "08:00-09:00"},
    {"honey_moisture_pct": float("nan")}, {"honey_moisture_pct": True},
    {"honey_moisture_pct": -1}, {"capped_ratio_pct": 101},
    {"reserves_confirmed": "yes"}, {"feeding_separated": 1},
    {"treatment_checked": 0}, {"measured_at": "2025-05-01T00:00:00"},
    {"measured_at": "not-a-date"},
])
def test_malformed_preferences_fail_explicitly(preferences):
    with pytest.raises(ValueError):
        build_harvest_context(report(), preferences)


def test_missing_or_invalid_candidate_never_fabricates_work_times():
    for candidate in (None, {}, {"start": "bad", "end": "bad"},
                      {"start": "2025-05-02T00:00:00Z", "end": "2025-05-01T00:00:00Z"}):
        original = report(); original["candidate_window"] = candidate
        result = build_harvest_context(original, {"recorded_timezone": "Asia/Seoul",
            "working_hours": {"start": "08:00", "end": "10:00"}})
        assert result["work_windows"] == []
        assert result["timing_basis"]["status"] in {"candidate_unavailable", "source_time_unresolved"}


def test_null_optional_preferences_are_unknown_and_do_not_mutate_inputs():
    preferences = {"recorded_timezone": None, "working_hours": None, "measured_at": None,
        "honey_moisture_pct": None, "capped_ratio_pct": None, "reserves_confirmed": None,
        "feeding_separated": None, "treatment_checked": None}
    before = deepcopy(preferences)
    result = build_harvest_context(report(), preferences)
    assert preferences == before
    assert all(item["status"] == "unknown" for item in result["field_checks"])



def test_timezone_area_name_is_rejected_as_value_error():
    with pytest.raises(ValueError):
        build_harvest_context(report(), {"recorded_timezone": "Europe"})
