#!/usr/bin/env python3
"""Reproduce one coverage-selected Vecauce hive as JSON, without model selection.

Uses the five original CSV files, published MD5 checksums and official README.
No network access, model training, interpolation, weight offsets or final CSV
writing occurs. Hour [h,h+1) is labelled at h+1 so features use completed bins.
Source timezone is unspecified: +00:00 is bookkeeping, not a UTC claim.

From beeops/: ../.venv/bin/python scripts/prepare_vecauce_data.py \
    --output /tmp/beeops-vecauce-prepared.json
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

RAW=Path(__file__).resolve().parents[1]/'data/raw/vecauce2021'
DATASET_URL='https://dv.dataverse.lv/dataset.xhtml?persistentId=doi:10.71782/DATA/J1CQTJ'


def sha256(content): return hashlib.sha256(content).hexdigest()


def blocks(frame):
    if frame.empty: return []
    labels=frame.index.to_series().diff().ne(pd.Timedelta(hours=1)).cumsum()
    return [part for _,part in frame.groupby(labels)]


def flagged(frame):
    result=frame.copy()
    result['event']=np.where(result.weight_kg.diff().abs().gt(1),'colony_alert','normal')
    return result


def load_hourly(path,datafile):
    content=path.read_bytes(); actual_md5=hashlib.md5(content).hexdigest()
    if actual_md5!=datafile['md5']: raise ValueError(f'MD5 mismatch for original CSV {path.name}')
    raw=pd.read_csv(io.BytesIO(content),na_values=['NULL'])
    expected={'Date','Time','Temperature IN','Temperature OUT','Weight'}
    if set(raw.columns)!=expected: raise ValueError(f'Unexpected original CSV columns in {path.name}')
    timestamp=pd.to_datetime(raw.Date+' '+raw.Time,format='%d/%m/%Y %H:%M:%S',errors='coerce')
    weight=pd.to_numeric(raw.Weight,errors='coerce')
    ambient=pd.to_numeric(raw['Temperature OUT'],errors='coerce')
    missing=timestamp.isna()|weight.isna()|ambient.isna()
    physical=np.isfinite(weight)&weight.between(0,300,inclusive='right')&np.isfinite(ambient)&ambient.between(-50,60)
    in_period=timestamp.ge('2021-06-01')&timestamp.lt('2021-09-01')
    valid=(~missing)&physical&in_period
    paired=pd.DataFrame({'timestamp':timestamp[valid],'weight_kg':weight[valid],
                         'temperature_c':ambient[valid]}).sort_values('timestamp')
    hourly=paired.set_index('timestamp').resample('h',closed='left',label='right').median().dropna()
    segments=blocks(hourly)
    if not segments: raise ValueError(f'No valid hourly observations in {path.name}')
    longest=max(segments,key=len); marked=flagged(longest)
    normal=blocks(marked[marked.event.eq('normal')])
    first_flag=next((i for i,event in enumerate(marked.event) if event!='normal'),len(marked))
    info={'hive_id':'vecauce_2021_'+path.stem.rsplit('_',1)[-1],
          'source_file':path.name,'datafile_id':datafile['id'],
          'download_url':f'https://dv.dataverse.lv/api/access/datafile/{datafile["id"]}?format=original',
          'md5':actual_md5,'published_md5':datafile['md5'],'md5_verified':True,
          'sha256':sha256(content),'source_bytes':len(content),'source_rows':len(raw),
          'valid_source_rows':len(paired),'missing_required_rows':int(missing.sum()),
          'physical_bounds_rejected_rows':int(((~missing)&~physical).sum()),
          'outside_documented_period_rows':int(((~missing)&physical&~in_period).sum()),
          'valid_duplicate_timestamp_rows':int(paired.timestamp.duplicated().sum()),
          'occupied_hourly_bins':len(hourly),'continuous_segments':len(segments),
          'longest_continuous_rows':len(longest),
          'longest_start_hour_end':longest.index[0].isoformat(),
          'longest_end_hour_end':longest.index[-1].isoformat(),
          'initial_normal_prefix_rows':first_flag,'longest_clean_rows':max(map(len,normal),default=0),
          'longest_segment_event_counts':{str(k):int(v) for k,v in marked.event.value_counts().items()},
          'weight_range_kg':[float(paired.weight_kg.min()),float(paired.weight_kg.max())],
          'outside_temperature_range_c':[float(paired.temperature_c.min()),float(paired.temperature_c.max())]}
    return hourly,info


def select_output(hourly,minimum_prefix=168):
    # `minimum_prefix` is retained as an explicit availability threshold, but
    # the clean block may occur anywhere: Service._bootstrap_frame scans blocks.
    selected=flagged(max(blocks(hourly),key=len))
    normal=blocks(selected[selected.event.eq('normal')])
    longest_clean=max(map(len,normal),default=0)
    if longest_clean<minimum_prefix:
        raise ValueError('Coverage-selected interval has no sufficiently long clean training block')
    first_flag=next((i for i,event in enumerate(selected.event) if event!='normal'),len(selected))
    return selected,{'rule':'longest_continuous_interval_with_event_flags_preserved',
                     'initial_normal_prefix_rows':first_flag,'longest_clean_rows':longest_clean,
                     'minimum_clean_block_rows':minimum_prefix,
                     'clean_block_can_follow_event':True,'rows_removed_for_initialization':0}


def prepare(raw_directory=RAW):
    raw_directory=Path(raw_directory)
    metadata_path=raw_directory/'metadata.json'; metadata=json.loads(metadata_path.read_text())
    version=metadata['datasetVersion']
    candidates=[]; frames={}
    originals=sorted((entry['dataFile'] for entry in version['files'] if entry['dataFile'].get('originalFileName')),
                     key=lambda item:item['originalFileName'])
    if len(originals)!=5: raise ValueError('Expected all five official original hive CSVs')
    for datafile in originals:
        hourly,info=load_hourly(raw_directory/datafile['originalFileName'],datafile)
        candidates.append(info); frames[info['hive_id']]=hourly
    # Stable file ordering breaks equal coverage ties, then blocks() preserves
    # the earliest interval. No training, validation or accuracy enters choice.
    winner=max(candidates,key=lambda item:item['longest_continuous_rows'])
    selected,selection=select_output(frames[winner['hive_id']])
    output=selected.reset_index(names='timestamp')
    output['timestamp']=output.timestamp.dt.tz_localize('UTC').map(lambda stamp:stamp.isoformat())
    output['hive_id']=winner['hive_id']
    rows=output[['timestamp','hive_id','weight_kg','temperature_c','event']].to_dict('records')
    start,end=rows[0]['timestamp'],rows[-1]['timestamp']
    manifest={'hive_id':winner['hive_id'],'display_name':'Vecauce Latvia 2021 · hive '+winner['hive_id'].rsplit('_',1)[-1],
        'dataset_title':'Bee colony monitoring data in Vecauce, Latvia, summer 2021',
        'source':{**winner,'dataset_url':DATASET_URL,'doi':metadata['persistentUrl'],
                  'citation':version['citation'],'license':version['license']['name'],
                  'license_url':version['license']['uri'],'dataset_version':'1.0',
                  'metadata_sha256':sha256(metadata_path.read_bytes()),
                  'readme_sha256':sha256((raw_directory/'ReadMe.txt').read_bytes())},
        'selection':{**selection,'hive_rule':'greatest longest continuous hourly coverage across all five hives; filename order breaks ties; earliest interval breaks within-hive ties',
                     'selection_uses_model_performance':False,'all_candidates':candidates},
        'transformation':{'source_weight_column':'Weight','source_weight_unit':'kg','weight_scale_factor':1,
            'source_temperature_column':'Temperature OUT','source_temperature_unit':'Celsius',
            'ignored_temperature_column':'Temperature IN',
            'date_format':'%d/%m/%Y %H:%M:%S; README confirms day/month/year',
            'timestamp_policy':'Timezone unspecified in source; +00:00 is deterministic bookkeeping only, not a claim of UTC acquisition.',
            'hourly_aggregation':'Joint-valid weight/ambient rows only; component medians in [h,h+1), timestamp at h+1.',
            'missing_policy':'NULL or invalid required values are dropped jointly; no interpolation or gap filling.',
            'physical_bounds':{'weight_kg':'0 < weight <= 300','temperature_c':'-50 <= temperature <= 60'},
            'event_policy':'colony_alert when absolute consecutive hourly weight change >1kg; numerical flag, not a diagnosis or known beekeeper intervention.',
            'weight_offsets_or_calibration_applied':False,'synthetic_rows_added':0},
        'output':{'rows':len(rows),'start':start,'end':end,
                  'weight_range_kg':[float(selected.weight_kg.min()),float(selected.weight_kg.max())],
                  'temperature_range_c':[float(selected.temperature_c.min()),float(selected.temperature_c.max())],
                  'event_counts':{str(k):int(v) for k,v in selected.event.value_counts().items()},
                  'rows_sha256':sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode())}}
    return {'datasets':[{'manifest':manifest,'rows':rows}]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-directory',type=Path,default=RAW)
    parser.add_argument('--output',type=Path,default=Path('/tmp/beeops-vecauce-prepared.json'))
    args=parser.parse_args()
    if args.output.resolve().is_relative_to(args.raw_directory.resolve()):
        parser.error('Output must be outside the preserved raw-source directory')
    prepared=prepare(args.raw_directory)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(prepared,ensure_ascii=False,indent=2)+'\n')
    manifest=prepared['datasets'][0]['manifest']
    print(json.dumps({'output':str(args.output),'hive_id':manifest['hive_id'],
                      **manifest['output'],'selection':manifest['selection']['rule']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
