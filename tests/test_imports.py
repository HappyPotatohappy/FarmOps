import csv
import io
import time

import pytest

from app.imports import ImportService, ImportTooLarge
from app.store import Store
from test_store import rows


def csv_bytes(records):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)
    return stream.getvalue().encode()


class StoredImportBoundary:
    """Actual SQLite ingestion; replace only downstream model work."""
    def __init__(self, path, fail_after_save=False):
        self.store = Store(path)
        self.fail_after_save = fail_after_save
        self.calls = 0

    def import_upload(self, records, source, progress=None):
        self.calls += 1
        ingest = self.store.ingest(records, source)
        progress("model_initialization", 35, {"ingest": ingest})
        if self.fail_after_save:
            raise RuntimeError("model artifact unavailable after observations saved")
        return {
            "ingest": ingest,
            "forecast": {"status": "unavailable"},
            "model_initialization": {"status": "insufficient_data"},
            "analysis": {"status": "insufficient_data"},
            "warnings": ["No ready model; observations were saved."],
        }


class ManagerBoundary:
    def __init__(self, root):
        self.root = root
        self.services = {}
        self.ensure_calls = 0

    def catalog(self):
        return [
            {"workspace_id": key, "hive_id": key, "name": key,
             "model_ready": False, "current_version": None}
            for key in self.services
        ]

    def get(self, workspace_id=None):
        return self.services[workspace_id]

    def ensure(self, hive_id):
        self.ensure_calls += 1
        created = hive_id not in self.services
        if created:
            self.services[hive_id] = StoredImportBoundary(self.root / hive_id)
        return self.services[hive_id], created


@pytest.fixture
def imports(tmp_path):
    manager = ManagerBoundary(tmp_path / "workspaces")
    service = ImportService(manager, tmp_path)
    yield service, manager
    service.close()


def wait_job(service, job_id, timeout=30):
    # Wait for the persisted terminal state; Docker disk/CPU contention can
    # exceed five seconds without violating the asynchronous import contract.
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = service.get_job(job_id)
        if job["status"] not in {"queued", "running"}:
            return job
        time.sleep(.01)
    raise AssertionError(f"Import job did not finish within {timeout}s: {service.get_job(job_id)}")


def test_preview_is_read_only_and_short_new_dataset_is_useful(imports):
    service, manager = imports
    preview = service.preview(csv_bytes(rows(3)), "new.csv")
    assert preview["can_commit"] and preview["row_count"] == preview["insert_count"] == 3
    assert preview["model"] == {"ready": False, "version": None}
    assert manager.services == {} and manager.ensure_calls == 0
    assert preview["workspace_id"] == "BEE-01"


def test_invalid_csv_has_row_diagnostics_without_workspace_mutation(imports):
    service, manager = imports
    sample = rows(4)
    sample[1]["weight_kg"] = "NaN"
    sample[2]["hive_id"] = "OTHER"
    preview = service.preview(csv_bytes(sample), "bad.csv")
    assert not preview["can_commit"] and preview["row_count"] == 4
    assert {error["row"] for error in preview["errors"]} >= {3, 4}
    assert manager.ensure_calls == 0
    with pytest.raises(ValueError):
        service.commit(preview["preview_id"])


def test_preview_counts_duplicates_and_conflicts_without_any_changes(imports):
    service, manager = imports
    existing, _ = manager.ensure("BEE-01")
    existing.store.ingest(rows(4), "original")
    sample = rows(6)
    sample[1]["weight_kg"] += .2
    preview = service.preview(csv_bytes(sample), "overlap.csv")
    assert preview["duplicate_count"] == 3
    assert preview["conflict_count"] == 1
    assert preview["insert_count"] == 2
    assert not preview["can_commit"]
    assert existing.store.observations() == rows(4)


def test_commit_is_idempotent_uses_persisted_snapshot_and_survives_restart(imports, tmp_path):
    service, manager = imports
    preview = service.preview(csv_bytes(rows(4)), "ledger.csv")
    preview["sample"][0]["weight_kg"] = 99
    first = service.commit(preview["preview_id"])
    repeated = service.commit(preview["preview_id"])
    assert first["job_id"] == repeated["job_id"]
    done = wait_job(service, first["job_id"])
    assert done["status"] == "completed" and done["result"]["ingest"]["inserted"] == 4
    assert manager.get("BEE-01").store.observations() == rows(4)
    assert manager.get("BEE-01").calls == 1
    service.close()
    reopened = ImportService(manager, tmp_path)
    try:
        assert reopened.commit(preview["preview_id"])["job_id"] == first["job_id"]
        assert reopened.get_job(first["job_id"])["status"] == "completed"
    finally:
        reopened.close()


def test_new_conflict_after_preview_fails_atomically(imports):
    service, manager = imports
    existing, _ = manager.ensure("BEE-01")
    existing.store.ingest(rows(2), "original")
    preview = service.preview(csv_bytes(rows(4)), "pending.csv")
    changed = rows(1, start=2)
    changed[0]["weight_kg"] += .2
    existing.store.ingest(changed, "concurrent observation")
    job = wait_job(service, service.commit(preview["preview_id"])["job_id"])
    assert job["status"] == "failed"
    assert "Conflicting" in job["error"]
    assert existing.store.data_status()["rows"] == 3


def test_failure_after_ingestion_retains_partial_result(imports):
    service, manager = imports
    existing, _ = manager.ensure("BEE-01")
    existing.fail_after_save = True
    preview = service.preview(csv_bytes(rows(4)), "partial.csv")
    job = wait_job(service, service.commit(preview["preview_id"])["job_id"])
    assert job["status"] == "partial"
    assert job["result"]["ingest"]["inserted"] == 4
    assert "artifact unavailable" in job["error"]
    assert existing.store.data_status()["rows"] == 4


def test_restart_marks_stale_job_interrupted_and_retains_saved_result(imports, tmp_path):
    service, manager = imports
    preview = service.preview(csv_bytes(rows(4)), "restart.csv")
    done = wait_job(service, service.commit(preview["preview_id"])["job_id"])
    service.close()
    import sqlite3
    with sqlite3.connect(tmp_path / "imports.sqlite") as connection:
        connection.execute("UPDATE import_jobs SET status='running',stage='analyzing' WHERE job_id=?", (done["job_id"],))
    reopened = ImportService(manager, tmp_path)
    try:
        job = reopened.get_job(done["job_id"])
        assert job["status"] == "interrupted"
        assert job["result"]["ingest"]["inserted"] == 4
        assert "restart" in job["error"].lower()
    finally:
        reopened.close()


def test_csv_limits_header_and_job_filters(imports):
    service, manager = imports
    with pytest.raises(ImportTooLarge):
        service.preview(b"x" * (5 * 1024 * 1024 + 1), "large.csv")
    invalid = service.preview(b"timestamp,hive_id\n2025-01-01,A\n", "wrong.csv")
    assert not invalid["can_commit"] and invalid["errors"]
    preview = service.preview(csv_bytes(rows(2)), "first.csv")
    done = wait_job(service, service.commit(preview["preview_id"])["job_id"])
    assert service.list_jobs("BEE-01")[0]["job_id"] == done["job_id"]
    assert service.list_jobs("OTHER") == []
    with pytest.raises(KeyError):
        service.get_job("missing")


def test_reupload_reports_zero_new_rows_and_exact_duplicate_count(imports):
    service, manager = imports
    original = service.preview(csv_bytes(rows(4)), "original.csv")
    wait_job(service, service.commit(original["preview_id"])["job_id"])
    duplicate = service.preview(csv_bytes(rows(4)), "duplicate.csv")
    assert duplicate["can_commit"]
    assert duplicate["insert_count"] == 0 and duplicate["duplicate_count"] == 4
    done = wait_job(service, service.commit(duplicate["preview_id"])["job_id"])
    assert done["status"] == "completed"
    assert done["result"]["ingest"]["inserted"] == 0
    assert done["result"]["ingest"]["duplicate"] == 4
    assert manager.get("BEE-01").store.data_status()["rows"] == 4


def test_preview_rejects_ledger_gaps_backfills_and_bad_headers(imports):
    service, manager = imports
    existing, _ = manager.ensure("BEE-01")
    existing.store.ingest(rows(4, start=5), "original")
    for sample in (rows(2, start=10), rows(2, start=1)):
        preview = service.preview(csv_bytes(sample), "gap.csv")
        assert not preview["can_commit"] and preview["errors"]
    duplicate_header = b"timestamp,hive_id,weight_kg,temperature_c,event,event\n"
    assert not service.preview(duplicate_header, "bad.csv")["can_commit"]
    assert not service.preview(b'"unterminated', "bad.csv")["can_commit"]
    assert not service.preview(b'\xff\xfe\x00', "bad.csv")["can_commit"]
    assert existing.store.data_status()["rows"] == 4


def test_broken_workspace_model_is_an_inspectable_preview_error(imports, monkeypatch):
    service, manager = imports
    existing, _ = manager.ensure("BEE-01")
    existing.store.ingest(rows(4), "original")

    def unavailable(_):
        raise RuntimeError("Missing model bundle for existing workspace")

    monkeypatch.setattr(manager, "get", unavailable)
    preview = service.preview(csv_bytes(rows(4)), "existing.csv")
    assert preview["workspace_id"] == "BEE-01"
    assert not preview["can_commit"]
    assert "Missing model bundle" in preview["errors"][0]["message"]
    assert existing.store.data_status()["rows"] == 4
