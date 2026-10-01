"""Registry serving must include both real LSTM output and the TiRex worker."""
import numpy as np
import pytest
from app import horizon_models as hm, tirex_adapter
from test_horizon_models import manifest, rows


@pytest.mark.parametrize('lstm_weight', [.1, .05, .65, .95])
def test_tirex_bundle_serves_weighted_actual_lstm_for_any_hive(tmp_path,monkeypatch,lstm_weight):
    tf=pytest.importorskip('tensorflow')
    manifest(tmp_path,components=['lstm','tirex2'],weights={'lstm':lstm_weight,'tirex2':1-lstm_weight},
             files={'lstm':'lstm.keras','tirex2':'tirex2'})
    # Real Keras component; isolate only the expensive external foundation worker.
    x=np.zeros((2,24,2),dtype=np.float32)
    model,_=hm.train_lstm(x,np.ones((2,168),dtype=np.float32),epochs=1)
    model.save(tmp_path/'1/lstm.keras')
    (tmp_path/'1/tirex2').mkdir()
    for name in ('model.ckpt', 'model-config.yaml'):
        (tmp_path/'1/tirex2'/name).write_bytes(b'worker fixture')
    class DeterministicWorker:
        def __init__(self,path): assert path==tmp_path/'1/tirex2'
        def predict(self,contexts,horizon):
            return np.array([np.full(horizon,context[-1,0]+2.) for context in contexts])
    monkeypatch.setattr(tirex_adapter,'TirexAdapter',DeterministicWorker)
    source=rows()
    context=np.array([[[r['weight_kg'],r['temperature_c']] for r in source[-24:]]],dtype=np.float32)
    lstm=np.asarray(model(hm.encode_context(context,{'weight_scale':1.,'temperature_mean':20.,'temperature_scale':1.}),training=False))[0]+context[0,-1,0]
    expected=lstm_weight*lstm+(1-lstm_weight)*(context[0,-1,0]+2.)
    service=hm.HorizonModelService(tmp_path)
    assert service.active_version() == '1'
    first=service.predict(source,24)
    assert first['status']=='ok'
    np.testing.assert_allclose([p['weight_kg'] for p in first['trajectory']],expected[:24],rtol=1e-6)
    for row in source: row['hive_id']='another-hive'
    second=service.predict(source,24)
    assert second['trajectory']==first['trajectory']
    assert second['model']==first['model']
