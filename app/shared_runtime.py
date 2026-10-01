"""Common model bundles and alias, with hive-specific observation metadata."""
from copy import deepcopy
import json
from pathlib import Path
from threading import RLock
import pandas as pd
from .store import Store

REGISTRY_ID='BeeOPS_Common_Weight'


class SharedModelCoordinator:
    def __init__(self,root_runtime,model=None):
        self.root_runtime=Path(root_runtime).resolve()
        if model is None:
            from .shared_ml import SharedModelService
            model=SharedModelService(self.root_runtime/'shared_models')
        self.model=model
        self.lock=RLock()

    def for_runtime(self,runtime):
        return HiveSharedModel(self,Path(runtime))

    def frames(self,extra=None):
        from .workspaces import workspace_directory
        runtimes=[self.root_runtime]
        for metadata_path in sorted((self.root_runtime/'workspaces').glob('*/workspace.json')):
            metadata=json.loads(metadata_path.read_text())
            if metadata_path.parent.resolve()!=workspace_directory(self.root_runtime,metadata['hive_id']):
                raise ValueError('Invalid workspace metadata location')
            runtimes.append(metadata_path.parent)
        frames={}
        for runtime in runtimes:
            store=Store(runtime); rows=store.observations()
            if rows:
                frame=pd.DataFrame(rows); frame.attrs['source']=' | '.join(store.sources())
                frames[rows[0]['hive_id']]=frame
        if extra is not None and not extra.empty:
            hive=str(extra.iloc[0]['hive_id'])
            if hive not in frames: frames[hive]=extra
        return frames

    def ensure_ready(self,extra=None):
        with self.lock:
            frames=self.frames(extra)
            if not frames and not self.model.status().get('ready'):
                return {'status':'insufficient_data','ready':False,'versions':[]}
            return self.model.ensure_ready(frames)


class HiveSharedModel:
    model_scope='shared'
    registry_id=REGISTRY_ID
    prediction_namespace='shared'

    def __init__(self,coordinator,runtime):
        self.coordinator=coordinator
        self.runtime=Path(runtime)
        self.store=Store(runtime,prediction_namespace='shared')

    @property
    def shared(self): return self.coordinator.model

    def _hive(self):
        return self.store.get_meta('workspace_binding') or self.store.data_status()['hive_id']

    def status(self):
        state=deepcopy(self.shared.status())
        metadata=self.version_metadata(state['version']) if state.get('version') else {}
        state.update(model_scope='shared',registry_id=REGISTRY_ID,
                     consumed_through=state.get('consumed_through_by_hive',{}).get(self._hive(),metadata.get('monitoring_after')),
                     trained_through=metadata.get('train_end'))
        return state

    def predict(self,windows,version=None): return self.shared.predict(windows,version=version)

    def versions(self): return self.shared.versions()

    def version_metadata(self,version):
        metadata=deepcopy(self.shared.version_metadata(str(version)))
        ranges=metadata.get('per_hive_ranges') or {}
        if isinstance(ranges,str): ranges=json.loads(ranges)
        own=ranges.get(self._hive(),{})
        for key in ('train_start','train_end','validation_start','validation_end','test_start','test_end'):
            metadata[key]=own.get(key)
        metadata['monitoring_after']=own.get('monitoring_after',own.get('test_end'))
        metadata.update(model_scope='shared',registry_id=REGISTRY_ID)
        # The immutable bundle is global; this field must never bind its weights
        # to the hive whose metadata happened to be requested first.
        metadata.pop('hive_id',None)
        return metadata

    def bootstrap(self,frame):
        result=self.coordinator.ensure_ready(frame)
        return {**result,'version':self.shared.status().get('version'),
                'promoted':any(item.get('promoted') for item in result.get('results',[]))}

    def ensure_comparison_version(self,frame):
        return self.coordinator.ensure_ready(frame)

    def train_candidate(self,frame,reason,snapshot_id):
        with self.coordinator.lock:
            return self.shared.train_candidate_pool(self.coordinator.frames(frame),reason=reason,snapshot_id=snapshot_id)

    def rollback(self,version):
        with self.coordinator.lock:
            return {**self.shared.rollback(str(version)),'model_scope':'shared','registry_id':REGISTRY_ID}
