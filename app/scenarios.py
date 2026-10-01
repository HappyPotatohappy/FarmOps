"""Explicit synthetic replay fixtures, never labelled real hive observations."""
from datetime import datetime, timedelta, timezone
import math
import numpy as np


def generate(n=672,start=None,last_weight=None,scenario='normal',hive_id='BEE-DEMO'):
    start=datetime.fromisoformat(start) if isinstance(start,str) else start or datetime(2025,3,1,tzinfo=timezone.utc)
    rng=np.random.default_rng(42+int(start.timestamp()//3600)%10000)
    out=[]; current=last_weight
    for i in range(n):
        t=start+timedelta(hours=i); hour=t.timestamp()/3600
        amplitude=.45 if scenario in ('normal','colony','sensor') else 1.4
        phase=0 if scenario in ('normal','colony','sensor') else 1.0
        wave=lambda h: amplitude*math.sin(2*math.pi*h/24+phase)
        noise=float(rng.normal(0,.001))
        if current is None: current=40+wave(hour)
        else: current+=wave(hour)-wave(hour-1)+noise
        if scenario=='gate_fail': current+=float(rng.uniform(-.5,.5))
        event='normal'
        if i==n-1 and scenario=='colony': current-=3; event='colony_alert'
        if i==n-1 and scenario=='sensor': current+=2; event='sensor_fault'
        out.append({'timestamp':t.isoformat(),'hive_id':hive_id,'weight_kg':round(current,6),
                    'temperature_c':round(20+6*math.sin(2*math.pi*hour/24),4),'event':event})
    return out
