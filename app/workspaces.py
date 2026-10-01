"""Per-hive runtimes with request-scoped selection and preserved registries."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import threading

from .service import Service
from .store import Store


def workspace_directory(root_runtime, hive_id):
    if not isinstance(hive_id,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,40}',hive_id):
        raise ValueError('hive_id must contain 1..40 letters, numbers, underscores or hyphens')
    key=hashlib.sha256(hive_id.encode()).hexdigest()[:24]
    return Path(root_runtime).resolve()/'workspaces'/key


def workspace_metadata(hive_id):
    kind='synthetic' if hive_id=='BEE-DEMO' else 'public_real' if hive_id=='ufc_apis_2' else 'uploaded'
    name='합성 교육 벌통' if kind=='synthetic' else 'UFC Apis 2 공개 실측' if kind=='public_real' else hive_id
    return {'workspace_id':hive_id,'hive_id':hive_id,'name':name,'kind':kind}


class WorkspaceManager:
    def __init__(self,root_runtime,primary_service,model_factory=None):
        self.root_runtime=Path(root_runtime).resolve()
        self.primary=primary_service; self._services={}; self._entries={}; self._lock=threading.RLock()
        if model_factory is None:
            from .ml import ModelService
            model_factory=ModelService
        self.model_factory=model_factory
        folder=self.root_runtime/'workspaces'
        if folder.exists():
            for path in sorted(folder.glob('*/workspace.json')):
                metadata=json.loads(path.read_text()); hive=metadata['hive_id']
                expected=workspace_directory(self.root_runtime,hive)
                if path.parent.resolve()!=expected or metadata.get('workspace_id')!=hive:
                    raise ValueError(f'Invalid workspace metadata location: {path}')
                self._entries[hive]={**metadata,'path':expected}
        self.primary.store.set_meta('workspace_reserved_hives',sorted(self._entries))

    @property
    def default_workspace_id(self):
        return self.primary.store.get_meta('workspace_binding') or self.primary.store.data_status()['hive_id'] or ('@default' if 'default' in self._entries else 'default')

    def get(self,id=None):
        with self._lock:
            if id is None or id==self.default_workspace_id: return self.primary
            if id not in self._entries: raise KeyError(f'Unknown workspace: {id}')
            if id not in self._services:
                path=self._entries[id]['path']
                self._services[id]=Service(path,self.model_factory(path))
            return self._services[id]

    def ensure(self,hive_id):
        path=workspace_directory(self.root_runtime,hive_id)
        with self._lock:
            if hive_id in self._entries: return self.get(hive_id),False
            with self.primary.store.lock:
                pristine=not self.primary.store.get_meta('workspace_binding') and not self.primary.store.data_status()['hive_id']
                if pristine and not self._entries:
                    self.primary.store.set_meta('workspace_binding',hive_id)
                    self.primary.store.set_meta('demo_runtime',False)
                    return self.primary,True
            if hive_id==self.default_workspace_id: return self.primary,False
            if path.exists(): raise ValueError('Workspace directory already exists without registered metadata; inspect before importing')
            path.mkdir(parents=True)
            metadata=workspace_metadata(hive_id)
            # Publish identity before model creation so an initialization failure
            # remains inspectable and cannot silently create a second registry.
            temporary=path/'workspace.json.tmp'
            temporary.write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
            temporary.replace(path/'workspace.json')
            self._entries[hive_id]={**metadata,'path':path}
            self.primary.store.set_meta('workspace_reserved_hives',sorted(self._entries))
            return self.get(hive_id),True

    def catalog(self):
        with self._lock:
            default=self.default_workspace_id
            bound=self.primary.store.get_meta('workspace_binding') or self.primary.store.data_status()['hive_id']
            entries=[{**(workspace_metadata(default) if bound else
                      {'workspace_id':default,'hive_id':None,'name':'기본 작업공간','kind':'empty'}),
                      'path':self.root_runtime}]
            entries.extend(entry for key,entry in self._entries.items() if key!=default)
            result=[]
            for entry in entries:
                error=None; state={}; count=0
                try:
                    service=self.get(entry['workspace_id']); data=service.store.data_status()
                    state=service.model.status(); count=len(service.model.versions())
                except Exception as exc:
                    error=str(exc); data=Store(entry['path']).data_status()
                    dbpath=entry['path']/'mlflow.db'
                    if dbpath.exists():
                        try:
                            with sqlite3.connect(f'file:{dbpath}?mode=ro',uri=True) as db:
                                count=db.execute("SELECT COUNT(*) FROM model_versions WHERE name='BeeOPS_Weight'").fetchone()[0]
                        except sqlite3.Error: pass
                result.append({k:v for k,v in entry.items() if k!='path'}|{
                    'rows':data['rows'],'model_ready':bool(state.get('ready')),
                    'current_version':state.get('version'),'model_count':count,
                    'last_observed_at':data['end'],'error':error})
            return result

    def training_jobs(self):
        """Snapshot only loaded legacy workers; never open models or services."""
        with self._lock:
            loaded = [(None, self.primary), *self._services.items()]
        statuses = {'queued', 'running', 'completed', 'failed', 'partial', 'interrupted'}
        jobs = []
        for workspace_id, service in loaded:
            record = deepcopy(service.training)
            identity = record.get('snapshot_id') or record.get('requested_at')
            if record.get('status') not in statuses or not identity:
                continue
            if workspace_id is None:
                workspace_id = self.default_workspace_id
            record.update(workspace_id=workspace_id, job_id=f'legacy:{workspace_id}:{identity}',
                          source='legacy_lstm', stage='train', steps=[])
            record.pop('training_progress', None)
            if record.get('version') is not None:
                record['new_version'] = record['version']
            jobs.append(record)
        return sorted(jobs, key=lambda job: str(job.get('completed_at') or job.get('requested_at') or ''),
                      reverse=True)

    def close(self):
        with self._lock:
            for service in [self.primary,*self._services.values()]: service.close()
