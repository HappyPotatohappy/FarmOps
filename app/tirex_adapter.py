"""Optional TiRex-2 inference in an isolated Torch environment, offline only.

The serving process can keep its TensorFlow-compatible NumPy. This worker's
environment is explicit and does not modify the original BeeOPS installation.
"""
from __future__ import annotations
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np


def prepare_contexts(contexts):
    values = np.asarray(contexts, dtype=np.float32)
    if values.ndim != 3 or not values.shape[0] or values.shape[1] < 1 or values.shape[2] != 2:
        raise ValueError('Expected nonempty contexts with shape (N, T, 2)')
    if not np.isfinite(values).all():
        raise ValueError('All context observations must be finite')
    return values


def median_forecasts(raw, batch_size, horizon):
    values = np.asarray(raw, dtype=np.float32)
    if values.shape != (batch_size, 1, 9, horizon) or not np.isfinite(values).all():
        raise ValueError('TiRex forecast must have finite shape (N, 1, 9, horizon)')
    return values[:, 0, 4, :].copy()


class TirexAdapter:
    def __init__(self, local_path, worker_python=None, timeout=600):
        self.path = Path(local_path).resolve()
        if not all((self.path / name).is_file() for name in ('model-config.yaml', 'model.ckpt')):
            raise FileNotFoundError(f'Complete local TiRex-2 checkpoint required: {self.path}')
        self.worker_python = str(worker_python or os.environ.get('BEEOPS_TIREX_PYTHON') or sys.executable)
        self.timeout = timeout

    def predict(self, contexts, horizon=168):
        if not isinstance(horizon, int) or not 1 <= horizon <= 168:
            raise ValueError('Forecast horizon must be an integer in [1, 168]')
        values = prepare_contexts(contexts)
        with tempfile.TemporaryDirectory(prefix='beeops-tirex-') as folder:
            input_path, output_path = Path(folder)/'input.npy', Path(folder)/'output.npy'
            np.save(input_path, values, allow_pickle=False)
            env = {**os.environ, 'HF_HUB_OFFLINE': '1', 'HF_HUB_DISABLE_IMPLICIT_TOKEN': '1',
                   'TRANSFORMERS_OFFLINE': '1', 'TORCHDYNAMO_DISABLE': '1',
                   'MPLCONFIGDIR': str(Path(tempfile.gettempdir())/'beeops-upgrade-mpl')}
            result = subprocess.run([self.worker_python, str(Path(__file__).resolve()), '--worker',
                str(self.path), str(input_path), str(output_path), str(horizon)],
                capture_output=True, text=True, timeout=self.timeout, env=env)
            if result.returncode:
                raise RuntimeError(f'TiRex-2 inference failed ({result.returncode}): {result.stderr[-2000:]}')
            if not output_path.is_file():
                raise RuntimeError('TiRex-2 worker did not produce a forecast')
            forecasts = np.load(output_path, allow_pickle=False)
            if forecasts.shape != (len(values), horizon) or not np.isfinite(forecasts).all():
                raise ValueError('TiRex-2 worker returned invalid forecasts')
            return forecasts


def worker(checkpoint, input_path, output_path, horizon):
    import torch
    from tirex2 import TimeseriesType, load_model
    torch.set_num_threads(2)
    contexts = prepare_contexts(np.load(input_path, allow_pickle=False))
    model = load_model(str(Path(checkpoint).resolve()), device='cpu', compile=False,
                       use_flex_attention=False, hf_kwargs={'local_files_only': True, 'token': False})
    inputs = [TimeseriesType(target=torch.from_numpy(x[:,0].copy())[None],
              past_covariates=torch.from_numpy(x[:,1].copy())[None], future_covariates=None) for x in contexts]
    with torch.inference_mode():
        raw = model.forecast(inputs, prediction_length=int(horizon), output_type='numpy', batch_size=16)
    np.save(output_path, median_forecasts(raw, len(contexts), int(horizon)), allow_pickle=False)


if __name__ == '__main__':
    if len(sys.argv) != 6 or sys.argv[1] != '--worker':
        raise SystemExit('Use only the internal --worker checkpoint input output horizon interface')
    worker(*sys.argv[2:])
