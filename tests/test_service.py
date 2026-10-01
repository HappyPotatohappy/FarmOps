import pandas as pd
from app.service import Service


class FinishedModel:
    def train_candidate(self,frame,reason,snapshot_id):
        return {'promoted':True,'version':'2','parent_version':'1','mae':.02}


def test_reporting_io_error_does_not_hide_success_or_leave_job_running(tmp_path,monkeypatch):
    service=Service(tmp_path,FinishedModel())
    service.training={'status':'queued'}
    def fail(*args,**kwargs): raise OSError('disk reporting unavailable')
    monkeypatch.setattr(service.store,'log',fail)
    try:
        service._train(pd.DataFrame(), 'test','snapshot')
        assert service.training['status']=='completed'
        assert service.training['promoted'] is True
        assert service.training['reporting_warnings']
        assert service.last_completed>0
    finally: service.close()


def test_manual_initial_model_preparation_also_registers_both_versions(tmp_path):
    from test_workspaces import UploadModel
    from test_store import rows
    class PairModel(UploadModel):
        def ensure_comparison_version(self,frame):
            assert len(frame)==241
            return {'status':'created','versions':['1','2']}
    service=Service(tmp_path,PairModel())
    try:
        service.store.ingest(rows(241),'manual preparation')
        service._train(pd.DataFrame(rows(241)),'operator_request','snapshot')
        assert service.training['status']=='completed'
        assert service.training['model_versions']['versions']==['1','2']
    finally: service.close()
