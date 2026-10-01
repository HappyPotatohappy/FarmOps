import importlib.util
import pytest
from app.service import Service
from test_store import rows


class UploadModel:
    def __init__(self, runtime=None):
        self.ready=False; self.cutoff=None; self.bootstrap_rows=0
    def status(self):
        return {'ready':self.ready,'version':'1' if self.ready else None,'minimum_training_rows':168,
                'consumed_through':self.cutoff,'trained_through':self.cutoff}
    def bootstrap(self,frame):
        self.bootstrap_rows=len(frame); self.cutoff=str(frame.iloc[-1]['timestamp']); self.ready=True
        return {'promoted':True,'version':'1','test_end':self.cutoff,'gate_reasons':[]}
    def version_metadata(self,version):
        return {'version':str(version),'test_end':self.cutoff,'monitoring_after':self.cutoff}
    def versions(self):
        return [{'version':'1','champion':True,'params':{'test_end':self.cutoff}}] if self.ready else []
    def predict(self,windows,version=None):
        return windows[:,-1,0]+.01,version or '1'


def test_upload_initializes_model_replays_only_unseen_tail_and_handles_duplicates(tmp_path):
    model=UploadModel(); service=Service(tmp_path,model); progress=[]
    try:
        result=service.import_upload(rows(724),'uploaded csv',lambda stage,percent,details: progress.append((stage,details)))
        assert result['status']=='completed'
        assert result['ingest']['inserted']==724
        assert model.bootstrap_rows==672
        assert result['model_initialization']['ready'] is True
        assert service.store.get_meta('training')['status']=='completed'
        assert service.last_completed>0
        assert result['analysis']['prediction_count']==52
        assert result['analysis']['excluded_training_targets']==648
        assert result['forecast']['status']=='completed'
        assert result['forecast']['prediction']['actual_weight_kg'] is None
        assert len(service.store.predictions(1000))==53
        assert any(details and details.get('ingest',{}).get('inserted')==724 for _,details in progress)
        repeated=service.import_upload(rows(724),'same csv')
        assert repeated['ingest']['inserted']==0 and repeated['ingest']['duplicate']==724
        assert len(service.store.predictions(1000))==53
    finally: service.close()


def test_small_upload_retains_data_with_explicit_model_unavailable(tmp_path):
    service=Service(tmp_path,UploadModel())
    try:
        result=service.import_upload(rows(30),'small csv')
        assert result['status']=='completed'
        assert result['ingest']['rows']==30
        assert result['model_initialization']['status']=='insufficient_data'
        assert result['model_initialization']['required_rows']==168
        assert result['forecast']['status']=='unavailable'
    finally: service.close()


def test_ready_hive_prepares_second_version_from_original_clean_observations(tmp_path):
    class PairModel(UploadModel):
        def ensure_comparison_version(self,frame):
            assert len(frame)==241
            assert frame.iloc[0]['timestamp']==rows(1)[0]['timestamp']
            assert frame.iloc[-1]['timestamp']==self.cutoff
            return {'status':'created','versions':['1','2'],'comparison_version':'2'}
    model=PairModel(); model.ready=True; model.cutoff=rows(1,start=240)[0]['timestamp']
    service=Service(tmp_path,model)
    try:
        result=service.import_upload(rows(241),'existing single-version hive')
        assert result['status']=='completed'
        assert result['model_versions']['versions']==['1','2']
        assert result['forecast']['status']=='completed'
        assert result['ingest']['rows']==241
    finally: service.close()


def test_second_version_failure_preserves_first_model_and_committed_data(tmp_path):
    class BrokenPair(UploadModel):
        def ensure_comparison_version(self,frame): raise RuntimeError('Second model training failed')
    service=Service(tmp_path,BrokenPair())
    try:
        result=service.import_upload(rows(241),'valid csv')
        assert result['status']=='partial'
        assert result['model_versions']['status']=='failed'
        assert result['forecast']['status']=='completed'
        assert result['forecast']['prediction']['model_version']=='1'
        assert service.store.data_status()['rows']==241
        assert service.model.status()['ready'] is True
    finally: service.close()


def test_downstream_failure_returns_partial_with_committed_ingest(tmp_path):
    class BrokenPredictor(UploadModel):
        def predict(self,*args,**kwargs): raise RuntimeError('inference failed')
    service=Service(tmp_path,BrokenPredictor())
    try:
        result=service.import_upload(rows(700),'valid csv')
        assert result['status']=='partial' and result['ingest']['rows']==700
        assert result['analysis']['status']=='failed'
        assert result['forecast']['status']=='failed'
    finally: service.close()


def test_batch_excludes_training_period_and_repairs_legacy_quality(tmp_path):
    model=UploadModel(); model.ready=True; model.cutoff=rows(1,start=50)[0]['timestamp']
    service=Service(tmp_path,model)
    try:
        service.store.ingest(rows(60),'fixture')
        for i in range(24,51): service.store.record_prediction(rows(24,start=i-24),49,'1')
        result=service.batch(rows(60),auto=False)
        assert len(result['predictions'])==9
        assert result['excluded_training_targets']==27
        assert result['quality']['count']==9
        assert service.health()['quality']['count']==9
    finally: service.close()


def test_workspace_isolation_restart_dynamic_primary_and_safe_directory(tmp_path):
    assert importlib.util.find_spec('app.workspaces'), 'workspace manager is not implemented'
    from app.workspaces import WorkspaceManager, workspace_directory
    primary=Service(tmp_path,UploadModel())
    manager=WorkspaceManager(tmp_path,primary,model_factory=UploadModel)
    primary.store.ingest(rows(24),'legacy')
    assert manager.default_workspace_id=='BEE-01'
    other,created=manager.ensure('ANOTHER')
    assert created is True and manager.ensure('ANOTHER')==(other,False)
    assert other.store.path==workspace_directory(tmp_path,'ANOTHER')
    sample=rows(24)
    for row in sample: row['hive_id']='ANOTHER'
    other.import_upload(sample,'another csv')
    assert {r['workspace_id'] for r in manager.catalog()}=={'BEE-01','ANOTHER'}
    assert manager.get('BEE-01').store.data_status()['rows']==24
    with pytest.raises(ValueError): manager.ensure('../escape')
    manager.close()
    restarted=WorkspaceManager(tmp_path,Service(tmp_path,UploadModel()),model_factory=UploadModel)
    try:
        assert restarted.get('ANOTHER').store.data_status()['hive_id']=='ANOTHER'
        assert len(restarted.catalog())==2
    finally: restarted.close()


def test_first_workspace_binding_prevents_legacy_identity_shadowing_after_restart(tmp_path):
    from app.workspaces import WorkspaceManager
    primary=Service(tmp_path,UploadModel()); manager=WorkspaceManager(tmp_path,primary,model_factory=UploadModel)
    first,created=manager.ensure('BEE-01')
    assert created is True and first is primary
    first.import_upload(rows(30),'first')
    assert manager.get('BEE-01') is first
    wrong=rows(24)
    for row in wrong: row['hive_id']='WRONG'
    with pytest.raises(ValueError): primary.store.ingest(wrong,'legacy wrong hive')
    manager.close()
    manager=WorkspaceManager(tmp_path,Service(tmp_path,UploadModel()),model_factory=UploadModel)
    try:
        assert manager.default_workspace_id=='BEE-01'
        assert manager.get('BEE-01').store.data_status()['rows']==30
        assert [item['workspace_id'] for item in manager.catalog()]==['BEE-01']
    finally: manager.close()


def test_literal_default_hive_is_bound_before_empty_store_can_accept_another(tmp_path):
    from app.workspaces import WorkspaceManager
    primary=Service(tmp_path,UploadModel()); manager=WorkspaceManager(tmp_path,primary,model_factory=UploadModel)
    try:
        selected,created=manager.ensure('default')
        assert created is True and selected is primary
        with pytest.raises(ValueError): primary.store.ingest(rows(24),'wrong identity')
        assert manager.catalog()[0]['hive_id']=='default'
    finally: manager.close()


def test_rejected_initial_candidate_preserves_candidate_identity(tmp_path):
    class RejectedModel(UploadModel):
        def bootstrap(self,frame):
            return {'promoted':False,'version':'1','gate_reasons':['absolute_mae']}
    service=Service(tmp_path,RejectedModel())
    try:
        result=service.import_upload(rows(168),'candidate')
        assert result['status']=='partial'
        assert result['model_initialization']['status']=='quality_rejected'
        assert result['model_initialization']['candidate_version']=='1'
        assert result['model_initialization']['ready'] is False
    finally: service.close()


def test_existing_model_import_records_error_without_blocking_or_auto_training(tmp_path):
    class DegradedModel(UploadModel):
        def predict(self,windows,version=None): return windows[:,-1,0]-1,version or '1'
        def train_candidate(self,frame,reason,snapshot_id):
            return {'promoted':False,'version':'2','gate_reasons':['absolute_mae']}
    model=DegradedModel(); model.ready=True; model.cutoff=rows(1,start=23)[0]['timestamp']
    service=Service(tmp_path,model)
    try:
        result=service.import_upload(rows(240),'new data')
        assert result['analysis']['quality']['status']=='degraded'
        assert result['status']=='completed'
        assert result['forecast']['status']=='completed'
        assert result['analysis']['action']['monitoring_mode']=='record_only'
        assert service.training['status']=='idle'
        assert not any(alert['category']=='model' for alert in service.store.alerts())
    finally: service.close()


def test_explicit_synthetic_drift_demo_keeps_automatic_training(tmp_path):
    class DriftModel(UploadModel):
        def predict(self,windows,version=None): return windows[:,-1,0]-1,version or '1'
        def train_candidate(self,frame,reason,snapshot_id):
            return {'promoted':True,'version':'2','gate_reasons':[]}
    model=DriftModel(); model.ready=True; model.cutoff=rows(1,start=23)[0]['timestamp']
    service=Service(tmp_path,model)
    try:
        service.store.set_meta('demo_runtime',True)
        service.demo={'status':'running','scenario':'drift'}
        result=service.batch(rows(240),'explicit drift demo')
        assert result['action']['status']=='queued'
        assert any(alert['category']=='model' for alert in service.store.alerts())
    finally: service.close()


def test_appending_after_ten_thousand_rows_keeps_analysis_and_latest_forecast(tmp_path):
    model=UploadModel(); model.ready=True; model.cutoff=rows(1,start=9990)[0]['timestamp']
    service=Service(tmp_path,model)
    try:
        service.store.ingest(rows(10000),'earlier history')
        result=service.import_upload(rows(24,start=10000),'new suffix')
        assert result['status']=='completed',result
        assert result['ingest']['rows']==10024
        assert result['analysis']['prediction_count']==33
        assert result['analysis']['excluded_training_targets']==9967
        assert result['forecast']['status']=='completed'
    finally: service.close()


def test_imported_secondary_hive_cannot_be_shadowed_by_empty_legacy_primary(tmp_path):
    import json
    from app.workspaces import WorkspaceManager,workspace_directory,workspace_metadata
    from app.store import Store
    secondary=workspace_directory(tmp_path,'BEE-01'); secondary.mkdir(parents=True)
    Store(secondary).ingest(rows(30),'migrated')
    (secondary/'workspace.json').write_text(json.dumps(workspace_metadata('BEE-01')))
    primary=Service(tmp_path,UploadModel())
    manager=WorkspaceManager(tmp_path,primary,model_factory=UploadModel)
    try:
        with pytest.raises(ValueError): primary.store.ingest(rows(24),'legacy conflicting route')
        assert manager.get('BEE-01').store.data_status()['rows']==30
    finally: manager.close()
