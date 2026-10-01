"""Inspectable CSV previews and durable, idempotent background import jobs."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timedelta
import csv
import io
import json
from pathlib import Path
import sqlite3
import threading
import uuid

from pydantic import ValidationError

from .contracts import Observation
from .store import same_observation_values, utcnow


MAX_BYTES = 5 * 1024 * 1024
MAX_ROWS = 10000
COLUMNS = ["timestamp", "hive_id", "weight_kg", "temperature_c", "event"]


class ImportTooLarge(ValueError):
    """The HTTP layer maps this separately to 413."""


class ImportService:
    def __init__(self, manager, root_runtime, on_observations=None):
        self.manager = manager
        self.on_observations = on_observations
        self.path = Path(root_runtime) / "imports.sqlite"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.closed = False
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="beeops-import")
        with self.db() as connection:
            connection.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS import_previews (
                    preview_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    rows_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS import_jobs (
                    job_id TEXT PRIMARY KEY,
                    preview_id TEXT NOT NULL UNIQUE,
                    workspace_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    status TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    progress INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    result_json TEXT,
                    error TEXT
                );
            """)
            # The local deployment has one process. A newly started process
            # never silently resumes a potentially partly completed operation.
            connection.execute(
                "UPDATE import_jobs SET status='interrupted',stage='interrupted',updated_at=?,"
                "error='Import interrupted by service restart; saved observations/results are retained. "
                "Preview the file again to continue safely.' WHERE status IN ('queued','running')",
                (utcnow(),),
            )

    @contextmanager
    def db(self):
        with self.lock:
            connection = sqlite3.connect(self.path, timeout=30)
            connection.row_factory = sqlite3.Row
            try:
                yield connection
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
            finally:
                connection.close()

    @staticmethod
    def _filename(filename):
        name = str(filename or "upload.csv").replace("\\", "/").split("/")[-1]
        return "".join(character for character in name if character.isprintable())[:200] or "upload.csv"

    def preview_temperature_practice(self, filename):
        from .temperature_practice import practice_preview_input
        content, metadata = practice_preview_input(filename)
        return self.preview(content, filename, _temperature_practice=metadata)

    def preview(self, content, filename, *, _temperature_practice=None):
        if len(content) > MAX_BYTES:
            raise ImportTooLarge("CSV maximum size is 5 MiB")
        now = utcnow()
        preview = {
            "preview_id": uuid.uuid4().hex,
            "filename": self._filename(filename),
            "status": "invalid",
            "can_commit": False,
            "format": "beeops_hourly_csv",
            "workspace_id": None,
            "hive_id": None,
            "workspace_name": None,
            "row_count": 0,
            "insert_count": 0,
            "duplicate_count": 0,
            "conflict_count": 0,
            "start": None,
            "end": None,
            "columns": [],
            "sample": [],
            "errors": [],
            "warnings": [],
            "model": {"ready": False, "version": None},
            "created_at": now,
        }
        if _temperature_practice:
            preview['temperature_practice'] = dict(_temperature_practice)
        records, line_numbers = [], []
        error_count = 0

        def error(line, message):
            nonlocal error_count
            error_count += 1
            if len(preview["errors"]) < 100:
                preview["errors"].append({"row": int(line), "message": str(message)})

        try:
            text = content.decode("utf-8-sig")
            if "\x00" in text:
                raise ValueError("CSV contains a NUL byte; upload a UTF-8 text CSV.")
            reader = csv.reader(io.StringIO(text, newline=""), strict=True)
            header = next(reader, [])
            preview["columns"] = header
            valid_header = len(header) == len(COLUMNS) and set(header) == set(COLUMNS)
            if not valid_header:
                error(1, "CSV columns must contain exactly: " + ", ".join(COLUMNS))
            previous, hive = None, None
            for cells in reader:
                preview["row_count"] += 1
                line = reader.line_num
                if preview["row_count"] > MAX_ROWS:
                    continue
                if not valid_header:
                    continue
                if len(cells) != len(header):
                    error(line, f"Expected {len(header)} CSV fields, received {len(cells)}.")
                    continue
                raw = dict(zip(header, cells))
                try:
                    observation = Observation(**raw)
                    row = {
                        **observation.model_dump(mode="json"),
                        "timestamp": observation.timestamp.isoformat(),
                    }
                except ValidationError as exc:
                    for issue in exc.errors():
                        error(line, f"{'.'.join(str(part) for part in issue['loc'])}: {issue['msg']}")
                    # Still detect a mixed hive in a subsequent row even when
                    # this row contains invalid numbers.
                    if hive is None and raw.get("hive_id"):
                        hive = raw["hive_id"]
                    continue
                if hive is None:
                    hive = row["hive_id"]
                if row["hive_id"] != hive:
                    error(line, f"Only one hive_id is allowed; expected {hive}, received {row['hive_id']}.")
                if previous is not None:
                    gap = observation.timestamp - datetime.fromisoformat(previous["timestamp"])
                    if gap != timedelta(hours=1):
                        error(line, "Observations must be ordered, unique, and exactly one hour apart.")
                previous = row
                records.append(row)
                line_numbers.append(line)
                if len(preview["sample"]) < 5:
                    preview["sample"].append(row)
            if preview["row_count"] == 0:
                error(2, "CSV contains no observations.")
            if preview["row_count"] > MAX_ROWS:
                error(MAX_ROWS + 2, f"CSV has {preview['row_count']} observations; maximum is {MAX_ROWS}.")
        except UnicodeError:
            error(1, "CSV must be UTF-8 encoded.")
        except (csv.Error, ValueError) as exc:
            error(getattr(locals().get("reader"), "line_num", 1), str(exc))

        if records:
            hive = records[0]["hive_id"]
            preview.update(hive_id=hive, workspace_id=hive, workspace_name=hive,
                           start=min(row["timestamp"] for row in records),
                           end=max(row["timestamp"] for row in records))
            service, old_rows, workspace_unavailable = None, {}, False
            try:
                existing = next((item for item in self.manager.catalog()
                                 if item["workspace_id"] == hive), None)
                if existing:
                    preview["workspace_name"] = existing.get("name") or hive
                    preview["model"] = {"ready": bool(existing.get("model_ready")),
                                        "version": existing.get("current_version")}
                    if existing.get("error"):
                        raise RuntimeError(existing["error"])
                    service = self.manager.get(hive)
                    with service.store.db() as connection:
                        old_rows = {item["timestamp"]: dict(item) for item in
                                    connection.execute("SELECT * FROM observations ORDER BY timestamp")}
            except Exception as exc:
                workspace_unavailable = True
                error(0, f"Cannot inspect existing workspace {hive}: {exc}")
                preview["warnings"].append("대상 워크스페이스를 읽을 수 없어 신규·중복·충돌 행 수를 계산하지 못했습니다.")
            for row, line in (() if workspace_unavailable else zip(records, line_numbers)):
                old = old_rows.get(row["timestamp"])
                if old is None:
                    preview["insert_count"] += 1
                elif same_observation_values(old, row) and row["event"] in (old["event"], old["raw_event"]):
                    preview["duplicate_count"] += 1
                else:
                    preview["conflict_count"] += 1
                    error(line, f"Conflicting observation at {row['timestamp']}; original will be retained.")
            if service and not error_count:
                try:
                    # This shared Store planner performs the same no-gap,
                    # no-backfill and conflict checks as the atomic ingestion.
                    service.store.preview_ingest(records,temperature_practice=bool(
                        _temperature_practice and _temperature_practice['phase']=='heatwave'))
                except ValueError as exc:
                    error(line_numbers[0], str(exc))
            if preview["duplicate_count"]:
                preview["warnings"].append(
                    f"동일한 기존 관측 {preview['duplicate_count']}행은 다시 저장하지 않습니다. 실제 결과는 확정 시점 기준입니다."
                )
            event_count = sum(row["event"] != "normal" for row in records)
            if event_count:
                preview["warnings"].append(
                    f"이벤트가 표시된 {event_count}행과 해당 행이 포함된 예측 입력은 정상 품질 평가·학습에서 제외됩니다."
                )
            jumps = sum(
                row["event"] == "normal" and abs(row["weight_kg"] - prior["weight_kg"]) > 1
                for prior, row in zip(records, records[1:])
            )
            if jumps:
                preview["warnings"].append(f"미표시 급변 {jumps}행은 저장 시 colony_alert로 분류됩니다. 원인 진단이 아닙니다.")
            if not preview["model"]["ready"]:
                preview["warnings"].append(
                    "준비된 모델이 없습니다. 저장 후 연속 정상 관측이 168행 이상이면 LSTM을 학습하고 예측을 생성합니다."
                )
            preview["warnings"].append("시간은 UTC로 정규화합니다. 결측 시각 보간·정렬·관측값 수정은 자동으로 수행하지 않습니다.")
        if error_count > len(preview["errors"]):
            preview["errors"].append({"row": 0, "message": f"{error_count - len(preview['errors'])} additional errors omitted."})
        preview["can_commit"] = bool(records) and error_count == 0
        preview["status"] = "valid" if preview["can_commit"] else "conflict" if preview["conflict_count"] else "invalid"
        with self.db() as connection:
            connection.execute("INSERT INTO import_previews VALUES (?,?,?,?)", (
                preview["preview_id"], json.dumps(preview, ensure_ascii=False),
                json.dumps(records, ensure_ascii=False), now,
            ))
        # The preview's nested sample may be edited by a caller; persisted
        # normalized rows above remain immutable and are the commit input.
        return preview

    @staticmethod
    def _public_job(row):
        return {key: row[key] for key in (
            "job_id", "workspace_id", "filename", "status", "stage", "progress", "created_at", "updated_at"
        )} | {"result": json.loads(row["result_json"]) if row["result_json"] else None,
             "error": row["error"]}

    def commit(self, preview_id):
        with self.lock:
            with self.db() as connection:
                existing = connection.execute("SELECT * FROM import_jobs WHERE preview_id=?", (preview_id,)).fetchone()
                if existing:
                    return self._public_job(existing)
                if self.closed:
                    raise RuntimeError("Import service is closed")
                preview = connection.execute("SELECT * FROM import_previews WHERE preview_id=?", (preview_id,)).fetchone()
                if preview is None:
                    raise KeyError("Unknown preview_id")
                payload = json.loads(preview["payload_json"])
                if not payload["can_commit"]:
                    raise ValueError("Preview contains validation errors or conflicts; choose a corrected CSV.")
                now, job_id = utcnow(), uuid.uuid4().hex
                connection.execute("INSERT INTO import_jobs VALUES (?,?,?,?,?,?,?,?,?,?,?)", (
                    job_id, preview_id, payload["workspace_id"], payload["filename"],
                    "queued", "queued", 0, now, now, None, None,
                ))
            try:
                self.executor.submit(self._run, job_id, preview_id)
            except RuntimeError as exc:
                self._finish(job_id, "failed", error=str(exc))
            return self.get_job(job_id)

    def _progress(self, job_id, stage, percent, details=None):
        with self.db() as connection:
            row = connection.execute("SELECT result_json,progress FROM import_jobs WHERE job_id=?", (job_id,)).fetchone()
            result = json.loads(row["result_json"]) if row["result_json"] else {}
            if details:
                result.update(details)
            connection.execute(
                "UPDATE import_jobs SET status='running',stage=?,progress=?,updated_at=?,result_json=? WHERE job_id=?",
                (str(stage), max(row["progress"], min(99, max(0, int(percent)))), utcnow(),
                 json.dumps(result, ensure_ascii=False) if result else None, job_id),
            )

    def _finish(self, job_id, status, result=None, error=None):
        with self.db() as connection:
            row = connection.execute("SELECT result_json FROM import_jobs WHERE job_id=?", (job_id,)).fetchone()
            saved = json.loads(row["result_json"]) if row["result_json"] else {}
            if result:
                saved.update(result)
            connection.execute(
                "UPDATE import_jobs SET status=?,stage=?,progress=100,updated_at=?,result_json=?,error=? WHERE job_id=?",
                (status, status, utcnow(), json.dumps(saved, ensure_ascii=False) if saved else None, error, job_id),
            )

    def _run(self, job_id, preview_id):
        service = None
        error = None
        try:
            self._progress(job_id, "saving_data", 5)
            with self.db() as connection:
                saved = connection.execute("SELECT * FROM import_previews WHERE preview_id=?", (preview_id,)).fetchone()
            preview = json.loads(saved["payload_json"])
            records = json.loads(saved["rows_json"])
            service, _ = self.manager.ensure(preview["hive_id"])
            practice = preview.get('temperature_practice')
            options = {'temperature_practice':True} if practice and practice['phase']=='heatwave' else {}
            source = f"사용자 CSV: {preview['filename']}"
            if practice:
                files = '01_baseline.csv + 02_heatwave.csv' if practice['phase']=='heatwave' else preview['filename']
                source = f'기본 합성 온도 실습 CSV: {files}'
            result = service.import_upload(
                records, source=source,
                progress=lambda stage, percent, details=None: self._progress(job_id, stage, percent, details),
                **options,
            )
        except Exception as exc:
            current = self.get_job(job_id)
            result = current["result"] or {}
            result['status'] = 'partial' if result.get('ingest') is not None else 'failed'
            error = str(exc)
        if self.on_observations and service is not None and result.get('ingest') is not None:
            try:
                practice = preview.get('temperature_practice')
                if practice and practice['phase']=='heatwave':
                    practice = {**practice,'request_id':job_id,
                                'inserted_observations':result['ingest']['inserted']}
                arguments = (service,preview['hive_id'],practice) if practice else (service,preview['hive_id'])
                monitoring = self.on_observations(*arguments)
                result['temperature_monitoring'] = monitoring
                result['retraining_requested'] = bool(monitoring and monitoring.get('retraining_requested'))
                result['retraining_job_id'] = monitoring.get('retraining_job_id') if monitoring else None
                if monitoring and monitoring.get('retraining_status'):
                    result['retraining_status'] = monitoring['retraining_status']
                    result['retraining_request_id'] = monitoring['retraining_request_id']
            except Exception as exc:
                result.setdefault('warnings', []).append(f'온도 모니터링 실패: {exc}')
                result['status'] = 'partial'
        status = result.get('status') if result.get('status') in ('partial', 'failed') else 'completed'
        self._finish(job_id, status, result=result, error=error)

    def get_job(self, job_id):
        with self.db() as connection:
            row = connection.execute("SELECT * FROM import_jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError("Unknown job_id")
        return self._public_job(row)

    def list_jobs(self, workspace_id=None):
        with self.db() as connection:
            if workspace_id is None:
                rows = connection.execute("SELECT * FROM import_jobs ORDER BY created_at DESC LIMIT 50").fetchall()
            else:
                rows = connection.execute("SELECT * FROM import_jobs WHERE workspace_id=? ORDER BY created_at DESC LIMIT 50",
                                          (workspace_id,)).fetchall()
        return [self._public_job(row) for row in rows]

    def close(self):
        with self.lock:
            self.closed = True
        self.executor.shutdown(wait=True)
