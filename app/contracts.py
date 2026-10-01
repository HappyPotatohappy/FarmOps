from datetime import datetime, timedelta, timezone
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Event = Literal['normal','harvest','feeding','inspection','sensor_fault','colony_alert']


class Observation(BaseModel):
    model_config = ConfigDict(extra='forbid')
    timestamp: datetime
    hive_id: str = Field(pattern=r'^[A-Za-z0-9_-]{1,40}$')
    weight_kg: Annotated[float, Field(gt=0, le=300, allow_inf_nan=False)]
    temperature_c: Annotated[float, Field(ge=-50, le=60, allow_inf_nan=False)]
    event: Event = 'normal'

    @field_validator('timestamp')
    @classmethod
    def hourly_aware(cls, value):
        if value.tzinfo is None: raise ValueError('timestamp requires an explicit timezone')
        value=value.astimezone(timezone.utc)
        if value.minute or value.second or value.microsecond:
            raise ValueError('timestamp must be an exact hour (completed-hour observation)')
        return value


def validate_series(rows, minimum=1, maximum=10000):
    if not minimum <= len(rows) <= maximum:
        raise ValueError(f'Expected {minimum}..{maximum} hourly observations')
    points=[r if isinstance(r,Observation) else Observation(**r) for r in rows]
    hive=points[0].hive_id
    for i,p in enumerate(points):
        if p.hive_id != hive: raise ValueError('Only one hive_id is allowed')
        if i and p.timestamp-points[i-1].timestamp != timedelta(hours=1):
            raise ValueError('Observations must be ordered, unique, and exactly one hour apart')
    return [{**p.model_dump(mode='json'),'timestamp':p.timestamp.isoformat()} for p in points]


class PredictRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    sequence: list[Observation] = Field(min_length=24,max_length=24)

    @model_validator(mode='after')
    def contiguous(self):
        validate_series(self.sequence,24,24)
        return self


class ObservationsRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    observations: list[Observation] = Field(min_length=1,max_length=3000)

    @model_validator(mode='after')
    def contiguous(self):
        validate_series(self.observations,1,3000)
        return self


class RollbackRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    version: str = Field(pattern=r'^[1-9][0-9]*$')


class DemoRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    scenario: Literal['normal','drift','colony','sensor','gate_fail']
