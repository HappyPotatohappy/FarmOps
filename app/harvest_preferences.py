"""Optional operator records, stored separately from model and prediction state."""
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator, model_validator
from .store import utcnow


class WorkingHours(BaseModel):
    model_config = ConfigDict(extra='forbid')
    start: str = Field(pattern=r'^(?:[01][0-9]|2[0-3]):[0-5][0-9]$')
    end: str = Field(pattern=r'^(?:[01][0-9]|2[0-3]):[0-5][0-9]$')

    @model_validator(mode='after')
    def distinct(self):
        if self.start == self.end:
            raise ValueError('작업 시작·종료 시간을 다르게 입력해 주세요.')
        return self


class HarvestPreferences(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    recorded_timezone: str | None = Field(default=None, max_length=100)
    working_hours: WorkingHours | None = None
    honey_moisture_pct: float | None = Field(default=None, ge=0, le=100, strict=True)
    capped_ratio_pct: float | None = Field(default=None, ge=0, le=100, strict=True)
    reserves_confirmed: StrictBool | None = None
    feeding_separated: StrictBool | None = None
    treatment_checked: StrictBool | None = None
    measured_at: str | None = Field(default=None, max_length=60)

    @field_validator('recorded_timezone')
    @classmethod
    def valid_timezone(cls, value):
        if value is None:
            return value
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError, OSError) as exc:
            raise ValueError('실제 원본 데이터의 IANA 시간대를 입력해 주세요. 예: Asia/Seoul') from exc
        return value

    @field_validator('measured_at')
    @classmethod
    def aware_measurement(cls, value):
        if value is None:
            return value
        stamp = datetime.fromisoformat(value)
        if stamp.utcoffset() is None:
            raise ValueError('현장 검사 시각에 시간대 offset이 필요합니다.')
        return stamp.isoformat()


def read_preferences(store, workspace_id):
    saved = store.get_meta('harvest_preferences_v1', {})
    preferences = HarvestPreferences.model_validate(saved.get('preferences', {})).model_dump()
    return {'workspace_id':workspace_id, 'preferences':preferences, 'updated_at':saved.get('updated_at')}


def save_preferences(store, workspace_id, preferences):
    result = {'workspace_id':workspace_id, 'preferences':preferences.model_dump(), 'updated_at':utcnow()}
    store.set_meta('harvest_preferences_v1', result)
    return result
