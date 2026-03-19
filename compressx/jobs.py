from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from compressx.config import CompressionConfig
from compressx.pipeline import CompressionPipeline

ACTIVE_JOB_STATES = {
    "QUEUED",
    "ANALYZING",
    "QUANTIZING",
    "PRUNING",
    "DISTILLING",
    "EVALUATING",
}
STATE_MAP = {
    "sensitivity": "ANALYZING",
    "quantization": "QUANTIZING",
    "pruning": "PRUNING",
    "distillation": "DISTILLING",
    "evaluation": "EVALUATING",
    "complete": "DONE",
}


@dataclass
class JobStatus:
    job_id: str
    status: str = "QUEUED"
    stage: str = "QUEUED"
    progress: float = 0.0
    current_size_gb: float | None = None
    original_size_gb: float | None = None
    accuracy_retention: float | None = None
    output_dir: Path | None = None
    error: str | None = None
    notify_url: str | None = None
    model_id: str | None = None
    target_size_gb: float | None = None
    distill: bool = False
    created_at: str | None = None
    updated_at: str | None = None
    stage_started_at: str | None = None


@dataclass(frozen=True, slots=True)
class CompressionStats:
    """Aggregate compression statistics computed from the jobs table."""

    total_jobs: int
    jobs_completed: int
    jobs_failed: int
    average_compression_ratio: float
    average_accuracy_retention: float
    total_storage_saved_gb: float


class CompressionJobManager:
    def __init__(
        self,
        *,
        db_path: Path = Path("jobs.db"),
        max_concurrent_jobs: int = 3,
    ) -> None:
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=max_concurrent_jobs)
        self._db_path = db_path
        self._max_concurrent_jobs = max_concurrent_jobs
        self._initialize_db()

    def _initialize_db(self) -> None:
        with sqlite3.connect(self._db_path, check_same_thread=False) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    progress REAL NOT NULL,
                    current_size_gb REAL,
                    original_size_gb REAL,
                    accuracy_retention REAL,
                    output_dir TEXT,
                    error TEXT,
                    notify_url TEXT,
                    model_id TEXT,
                    target_size_gb REAL,
                    distill INTEGER NOT NULL DEFAULT 0,
                    stage_started_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            self._ensure_columns(connection)
            connection.commit()

    def _ensure_columns(self, connection: sqlite3.Connection) -> None:
        existing_columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(jobs)")
        }
        required_columns: dict[str, str] = {
            "model_id": "TEXT",
            "target_size_gb": "REAL",
            "distill": "INTEGER NOT NULL DEFAULT 0",
            "stage_started_at": "TEXT",
            "original_size_gb": "REAL",
        }
        for column_name, column_definition in required_columns.items():
            if column_name in existing_columns:
                continue
            connection.execute(
                f"ALTER TABLE jobs ADD COLUMN {column_name} {column_definition}"
            )

    def _row_to_status(self, row: tuple[Any, ...]) -> JobStatus:
        return JobStatus(
            job_id=row[0],
            status=row[1],
            stage=row[2],
            progress=row[3],
            current_size_gb=row[4],
            original_size_gb=row[5],
            accuracy_retention=row[6],
            output_dir=Path(row[7]) if row[7] else None,
            error=row[8],
            notify_url=row[9],
            model_id=row[10],
            target_size_gb=row[11],
            distill=bool(row[12]),
            created_at=row[13],
            updated_at=row[14],
            stage_started_at=row[15],
        )

    def _active_job_count(self) -> int:
        with sqlite3.connect(self._db_path, check_same_thread=False) as connection:
            cursor = connection.execute(
                "SELECT COUNT(*) FROM jobs WHERE status IN (?, ?, ?, ?, ?, ?)",
                tuple(ACTIVE_JOB_STATES),
            )
            return int(cursor.fetchone()[0])

    def enqueue_job(
        self,
        config: CompressionConfig,
        *,
        notify_url: str | None = None,
        job_id: str | None = None,
    ) -> JobStatus:
        if self._active_job_count() >= self._max_concurrent_jobs:
            raise RuntimeError("Rate limit exceeded. Maximum 3 concurrent jobs are allowed.")

        job_id = job_id or uuid.uuid4().hex
        now = datetime.now(timezone.utc).isoformat()
        status = JobStatus(
            job_id=job_id,
            output_dir=config.output_dir,
            notify_url=notify_url,
            model_id=config.model_id,
            target_size_gb=config.target_size_gb,
            distill=config.distill,
            created_at=now,
            updated_at=now,
            stage_started_at=now,
        )
        with sqlite3.connect(self._db_path, check_same_thread=False) as connection:
            connection.execute(
                """
                INSERT INTO jobs (
                    job_id, status, stage, progress, current_size_gb, original_size_gb,
                    accuracy_retention,
                    output_dir, error, notify_url, model_id, target_size_gb, distill,
                    stage_started_at, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    status.status,
                    status.stage,
                    status.progress,
                    status.current_size_gb,
                    status.original_size_gb,
                    status.accuracy_retention,
                    str(config.output_dir),
                    status.error,
                    notify_url,
                    config.model_id,
                    config.target_size_gb,
                    int(config.distill),
                    now,
                    now,
                    now,
                ),
            )
            connection.commit()
        return status

    def start_job(
        self,
        job_id: str,
        config: CompressionConfig,
        *,
        notify_url: str | None = None,
        download_url: str | None = None,
    ) -> None:
        self._executor.submit(
            self._run_job,
            job_id,
            config,
            notify_url,
            download_url,
        )

    def get_status(self, job_id: str) -> JobStatus | None:
        with sqlite3.connect(self._db_path, check_same_thread=False) as connection:
            cursor = connection.execute(
                """
                SELECT job_id, status, stage, progress, current_size_gb, original_size_gb,
                       accuracy_retention,
                       output_dir, error, notify_url, model_id, target_size_gb, distill,
                       created_at, updated_at, stage_started_at
                FROM jobs
                WHERE job_id = ?
                """,
                (job_id,),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_status(row)

    def list_jobs(self, *, limit: int = 50) -> list[JobStatus]:
        with sqlite3.connect(self._db_path, check_same_thread=False) as connection:
            cursor = connection.execute(
                """
                SELECT job_id, status, stage, progress, current_size_gb, original_size_gb,
                       accuracy_retention,
                       output_dir, error, notify_url, model_id, target_size_gb, distill,
                       created_at, updated_at, stage_started_at
                FROM jobs
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = cursor.fetchall()
        return [self._row_to_status(row) for row in rows]

    def get_stats(self) -> CompressionStats:
        """Aggregate top-level dashboard statistics from the jobs table."""

        with sqlite3.connect(self._db_path, check_same_thread=False) as connection:
            rows = connection.execute(
                """
                SELECT status, original_size_gb, current_size_gb, accuracy_retention
                FROM jobs
                """
            ).fetchall()

        total_jobs = len(rows)
        jobs_completed = sum(1 for row in rows if row[0] == "DONE")
        jobs_failed = sum(1 for row in rows if row[0] == "FAILED")

        ratio_values: list[float] = []
        accuracy_values: list[float] = []
        total_storage_saved_gb = 0.0

        for status, original_size_gb, current_size_gb, accuracy_retention in rows:
            if status != "DONE":
                continue

            if original_size_gb is not None and current_size_gb not in (None, 0):
                ratio_values.append(float(original_size_gb) / float(current_size_gb))
                total_storage_saved_gb += max(
                    float(original_size_gb) - float(current_size_gb),
                    0.0,
                )

            if accuracy_retention is not None:
                accuracy_values.append(float(accuracy_retention))

        average_ratio = sum(ratio_values) / len(ratio_values) if ratio_values else 0.0
        average_accuracy = (
            sum(accuracy_values) / len(accuracy_values) if accuracy_values else 0.0
        )

        return CompressionStats(
            total_jobs=total_jobs,
            jobs_completed=jobs_completed,
            jobs_failed=jobs_failed,
            average_compression_ratio=round(average_ratio, 2),
            average_accuracy_retention=round(average_accuracy, 2),
            total_storage_saved_gb=round(total_storage_saved_gb, 2),
        )

    def _run_job(
        self,
        job_id: str,
        config: CompressionConfig,
        notify_url: str | None,
        download_url: str | None,
    ) -> None:
        self._update(job_id, status="QUEUED", stage="QUEUED", progress=0.0)

        def progress_callback(stage: str, progress: float, context) -> None:
            mapped_state = STATE_MAP.get(stage, "QUEUED")
            self._update(
                job_id,
                status=mapped_state,
                stage=mapped_state,
                progress=progress,
                current_size_gb=context.current_size_gb,
                accuracy_retention=context.accuracy_retention_percent,
            )

        try:
            pipeline = CompressionPipeline(config)
            context = pipeline.run(progress_callback=progress_callback)
            self._update(
                job_id,
                status="DONE",
                stage="DONE",
                progress=1.0,
                current_size_gb=context.current_size_gb,
                original_size_gb=context.original_size_gb,
                accuracy_retention=context.accuracy_retention_percent,
            )
            if notify_url:
                try:
                    payload = {
                        "job_id": job_id,
                        "status": "DONE",
                        "download_url": download_url,
                    }
                    requests.post(
                        notify_url,
                        data=json.dumps(payload),
                        headers={"Content-Type": "application/json"},
                        timeout=30,
                    )
                except Exception as exc:
                    self._update(
                        job_id,
                        error=f"Webhook notification failed: {exc}",
                    )
        except Exception as exc:
            self._update(
                job_id,
                status="FAILED",
                stage="FAILED",
                error=str(exc),
            )

    def _update(
        self,
        job_id: str,
        *,
        status: str | None = None,
        stage: str | None = None,
        progress: float | None = None,
        current_size_gb: float | None = None,
        original_size_gb: float | None = None,
        accuracy_retention: float | None = None,
        error: str | None = None,
    ) -> None:
        with self._lock, sqlite3.connect(self._db_path, check_same_thread=False) as connection:
            current = self.get_status(job_id)
            if current is None:
                return
            now = datetime.now(timezone.utc).isoformat()
            stage_started_at = (
                now
                if stage is not None and stage != current.stage
                else current.stage_started_at
            )
            connection.execute(
                """
                UPDATE jobs
                SET status = ?, stage = ?, progress = ?, current_size_gb = ?,
                    original_size_gb = ?, accuracy_retention = ?, error = ?,
                    updated_at = ?, stage_started_at = ?
                WHERE job_id = ?
                """,
                (
                    status or current.status,
                    stage or current.stage,
                    progress if progress is not None else current.progress,
                    current_size_gb
                    if current_size_gb is not None
                    else current.current_size_gb,
                    original_size_gb
                    if original_size_gb is not None
                    else current.original_size_gb,
                    accuracy_retention
                    if accuracy_retention is not None
                    else current.accuracy_retention,
                    error if error is not None else current.error,
                    now,
                    stage_started_at,
                    job_id,
                ),
            )
            connection.commit()
