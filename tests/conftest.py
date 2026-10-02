"""Keep unit tests independent of a running Docker demo's environment."""

import sys
from pathlib import Path

import pytest

# Tests share helpers across subfolders (e.g. `from test_store import rows`).
for folder in sorted(Path(__file__).parent.iterdir()):
    if folder.is_dir() and not folder.name.startswith(('.', '_')):
        sys.path.insert(0, str(folder))


@pytest.fixture(autouse=True)
def isolate_demo_environment(monkeypatch, tmp_path):
    # Application fixtures start empty unless a test explicitly opts into the
    # observed catalog. Never read or modify the production runtime volume.
    monkeypatch.delenv("BEEOPS_DASHBOARD_DATASET", raising=False)
    monkeypatch.delenv("BEEOPS_HORIZON_MODELS", raising=False)
    monkeypatch.setenv("BEEOPS_RUNTIME", str(tmp_path / "runtime"))
