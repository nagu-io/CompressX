from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, HttpUrl, field_validator

from compressx.config import CompressionConfig
from compressx.jobs import CompressionJobManager, JobStatus
from compressx.modeling import estimate_model_source_size_gb
from compressx.utils.io import download_text, read_json, write_json, zip_directory

STAGE_SEQUENCE: list[tuple[str, str]] = [
    ("sensitivity", "Stage 1: Sensitivity Analysis"),
    ("quantization", "Stage 2: Quantization"),
    ("pruning", "Stage 3: Structural Pruning"),
    ("distillation", "Stage 4: Distillation"),
    ("evaluation", "Stage 5: Evaluation"),
]
STATUS_TO_STAGE_KEY = {
    "ANALYZING": "sensitivity",
    "QUANTIZING": "quantization",
    "PRUNING": "pruning",
    "DISTILLING": "distillation",
    "EVALUATING": "evaluation",
}
ACTIVE_JOB_STATUSES = {
    "ANALYZING",
    "QUANTIZING",
    "PRUNING",
    "DISTILLING",
    "EVALUATING",
}


class CompressRequest(BaseModel):
    model_id: str = Field(min_length=1)
    target_size_gb: float = Field(gt=0, le=200)
    distill: bool = False
    domain_data_url: HttpUrl | None = None
    resume: bool = False
    webhook_url: HttpUrl | None = None
    hf_token: str | None = None
    quantization_mode: Literal["int4", "int8", "mixed"] = "mixed"
    head_pruning_threshold: float = Field(default=0.01, ge=0.0)
    layer_redundancy_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    calibration_data_path: str | None = None
    calibration_data_text: str | None = None

    @field_validator("model_id")
    @classmethod
    def validate_model_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("model_id must not be empty.")
        return value.strip()

    @field_validator("hf_token")
    @classmethod
    def validate_hf_token(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @field_validator("calibration_data_path")
    @classmethod
    def validate_calibration_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


class CompressResponse(BaseModel):
    job_id: str
    status: str


class StatusResponse(BaseModel):
    job_id: str
    model_id: str | None = None
    stage: str
    progress: float
    current_size_gb: float | None = None
    accuracy_retention: float | None = None
    status: str
    error: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


class JobStageResponse(BaseModel):
    key: str
    label: str
    status: str
    duration_seconds: float | None = None
    progress: float = 0.0


class JobMetricsResponse(BaseModel):
    original_size_gb: float | None = None
    final_size_gb: float | None = None
    current_size_gb: float | None = None
    compression_ratio: str | None = None
    compression_ratio_value: float | None = None
    accuracy_retention_percent: float | None = None
    layers_pruned: int = 0
    heads_pruned: int = 0
    perplexity_original: float | None = None
    perplexity_compressed: float | None = None
    inference_speed_original: str | None = None
    inference_speed_compressed: str | None = None


class JobSummaryResponse(BaseModel):
    job_id: str
    model_id: str
    status: str
    stage: str
    progress: float
    original_size_gb: float | None = None
    final_size_gb: float | None = None
    compression_ratio: str | None = None
    compression_ratio_value: float | None = None
    accuracy_retention_percent: float | None = None
    total_time_minutes: float | None = None
    created_at: str | None = None
    updated_at: str | None = None
    elapsed_seconds: float | None = None
    error: str | None = None


class JobListResponse(BaseModel):
    jobs: list[JobSummaryResponse]


class JobDetailResponse(BaseModel):
    job_id: str
    model_id: str
    status: str
    stage: str
    progress: float
    current_size_gb: float | None = None
    accuracy_retention: float | None = None
    error: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    stage_started_at: str | None = None
    elapsed_seconds: float | None = None
    stage_elapsed_seconds: float | None = None
    target_size_gb: float | None = None
    distill: bool = False
    stages: list[JobStageResponse]
    metrics: JobMetricsResponse
    report: dict[str, Any] | None = None
    warnings: list[str] = Field(default_factory=list)
    logs: list[str] = Field(default_factory=list)


class StatsSummaryResponse(BaseModel):
    total_jobs_run: int
    average_compression_ratio: float
    total_storage_saved_gb: float
    average_accuracy_retention: float


class StatsResponse(BaseModel):
    total_jobs: int
    jobs_completed: int
    jobs_failed: int
    average_compression_ratio: float
    average_accuracy_retention: float
    total_storage_saved_gb: float


router = APIRouter()
job_manager = CompressionJobManager()


def require_api_key(x_api_key: str = Header(..., alias="X-API-Key")) -> str:
    expected = os.environ.get("COMPRESSX_API_KEY", "compressx-dev-key")
    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key.")
    return x_api_key


def _normalize_output_dir(status: JobStatus) -> Path:
    if status.output_dir is not None:
        return status.output_dir
    return Path("jobs") / status.job_id


def _checkpoint_root(job_id: str) -> Path:
    return Path("checkpoints") / job_id


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return read_json(path)


def _tail_lines(path: Path, *, limit: int = 50) -> list[str]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return lines[-limit:]


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _seconds_between(start_value: str | None, end_value: str | None) -> float | None:
    start = _parse_iso(start_value)
    end = _parse_iso(end_value)
    if start is None or end is None:
        return None
    return max(0.0, (end - start).total_seconds())


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _compression_ratio_value(
    ratio: str | None,
    original_size_gb: float | None,
    final_size_gb: float | None,
) -> float | None:
    if ratio:
        normalized = ratio.strip().rstrip("x")
        try:
            return float(normalized)
        except ValueError:
            pass
    if original_size_gb is None or final_size_gb in (None, 0):
        return None
    return original_size_gb / final_size_gb


def _load_stage_metadata(job_id: str) -> dict[str, dict[str, Any]]:
    checkpoint_root = _checkpoint_root(job_id)
    metadata_by_stage: dict[str, dict[str, Any]] = {}
    for stage_key, _ in STAGE_SEQUENCE:
        metadata = _read_json_if_exists(checkpoint_root / stage_key / "metadata.json")
        if metadata is not None:
            metadata_by_stage[stage_key] = metadata
    return metadata_by_stage


def _load_request_snapshot(output_dir: Path) -> dict[str, Any]:
    return _read_json_if_exists(output_dir / "request.json") or {}


def _load_report(output_dir: Path) -> dict[str, Any] | None:
    return _read_json_if_exists(output_dir / "compression_report.json")


def _load_pruning_log(output_dir: Path) -> dict[str, Any]:
    return _read_json_if_exists(output_dir / "pruning_log.json") or {}


def _summarize_job(status: JobStatus) -> JobSummaryResponse:
    output_dir = _normalize_output_dir(status)
    report = _load_report(output_dir) or {}
    report_model_id = report.get("model_id")
    model_id = (
        status.model_id
        or (str(report_model_id) if report_model_id is not None else None)
        or _load_request_snapshot(output_dir).get("model_id")
        or status.job_id
    )
    original_size_gb = _safe_float(report.get("original_size_gb"))
    final_size_gb = _safe_float(report.get("final_size_gb")) or status.current_size_gb
    compression_ratio = report.get("compression_ratio")
    ratio_value = _compression_ratio_value(
        compression_ratio if isinstance(compression_ratio, str) else None,
        original_size_gb,
        final_size_gb,
    )
    closed_at = (
        status.updated_at
        if status.status in {"DONE", "FAILED"}
        else _now_iso()
    )
    elapsed_seconds = _seconds_between(status.created_at, closed_at)

    return JobSummaryResponse(
        job_id=status.job_id,
        model_id=model_id,
        status=status.status,
        stage=status.stage,
        progress=status.progress,
        original_size_gb=original_size_gb,
        final_size_gb=final_size_gb,
        compression_ratio=compression_ratio if isinstance(compression_ratio, str) else None,
        compression_ratio_value=ratio_value,
        accuracy_retention_percent=_safe_float(
            report.get("accuracy_retention_percent", status.accuracy_retention)
        ),
        total_time_minutes=_safe_float(report.get("total_time_minutes"))
        if report
        else (elapsed_seconds / 60.0 if elapsed_seconds is not None else None),
        created_at=status.created_at,
        updated_at=status.updated_at,
        elapsed_seconds=elapsed_seconds,
        error=status.error,
    )


def _build_stage_items(status: JobStatus) -> list[JobStageResponse]:
    metadata_by_stage = _load_stage_metadata(status.job_id)
    current_stage_key = STATUS_TO_STAGE_KEY.get(status.stage)
    current_stage_elapsed = _seconds_between(
        status.stage_started_at,
        status.updated_at if status.status in {"DONE", "FAILED"} else _now_iso(),
    )
    items: list[JobStageResponse] = []

    for stage_key, label in STAGE_SEQUENCE:
        metadata = metadata_by_stage.get(stage_key)
        duration_seconds = None
        progress = 0.0

        if metadata is not None:
            stage_details = metadata.get("stage_details", {})
            duration_seconds = _safe_float(stage_details.get("duration_seconds"))
            progress = 1.0
            stage_status = "COMPLETE"
        elif stage_key == "distillation" and not status.distill:
            stage_status = "SKIPPED" if status.status in {"DONE", "FAILED"} else "WAITING"
        elif status.status == "FAILED" and current_stage_key == stage_key:
            stage_status = "FAILED"
            duration_seconds = current_stage_elapsed
            progress = status.progress
        elif status.status in ACTIVE_JOB_STATUSES and current_stage_key == stage_key:
            stage_status = "RUNNING"
            duration_seconds = current_stage_elapsed
            progress = status.progress
        else:
            stage_status = "WAITING"

        items.append(
            JobStageResponse(
                key=stage_key,
                label=label,
                status=stage_status,
                duration_seconds=duration_seconds,
                progress=progress,
            )
        )

    return items


def _build_metrics(status: JobStatus) -> JobMetricsResponse:
    output_dir = _normalize_output_dir(status)
    report = _load_report(output_dir) or {}
    pruning_log = _load_pruning_log(output_dir)
    original_size_gb = _safe_float(report.get("original_size_gb"))
    final_size_gb = _safe_float(report.get("final_size_gb")) or status.current_size_gb
    compression_ratio = report.get("compression_ratio")
    ratio_value = _compression_ratio_value(
        compression_ratio if isinstance(compression_ratio, str) else None,
        original_size_gb,
        final_size_gb,
    )

    layers_pruned = int(report.get("layers_pruned", 0))
    if layers_pruned == 0:
        layers_pruned = len(pruning_log.get("layers_removed", []))

    heads_pruned = int(report.get("heads_pruned", 0))
    if heads_pruned == 0:
        heads_pruned = sum(
            len(heads)
            for heads in pruning_log.get("heads_removed", {}).values()
        )

    return JobMetricsResponse(
        original_size_gb=original_size_gb,
        final_size_gb=final_size_gb,
        current_size_gb=status.current_size_gb or final_size_gb,
        compression_ratio=compression_ratio if isinstance(compression_ratio, str) else None,
        compression_ratio_value=ratio_value,
        accuracy_retention_percent=_safe_float(
            report.get("accuracy_retention_percent", status.accuracy_retention)
        ),
        layers_pruned=layers_pruned,
        heads_pruned=heads_pruned,
        perplexity_original=_safe_float(report.get("perplexity_original")),
        perplexity_compressed=_safe_float(report.get("perplexity_compressed")),
        inference_speed_original=report.get("inference_speed_original"),
        inference_speed_compressed=report.get("inference_speed_compressed"),
    )


def _build_job_detail(status: JobStatus) -> JobDetailResponse:
    output_dir = _normalize_output_dir(status)
    report = _load_report(output_dir)
    request_snapshot = _load_request_snapshot(output_dir)
    report_model_id = report.get("model_id") if report else None
    model_id = (
        status.model_id
        or (str(report_model_id) if report_model_id is not None else None)
        or request_snapshot.get("model_id")
        or status.job_id
    )
    closed_at = (
        status.updated_at
        if status.status in {"DONE", "FAILED"}
        else _now_iso()
    )
    elapsed_seconds = _seconds_between(status.created_at, closed_at)
    stage_elapsed_seconds = _seconds_between(
        status.stage_started_at,
        status.updated_at if status.status in {"DONE", "FAILED"} else _now_iso(),
    )

    return JobDetailResponse(
        job_id=status.job_id,
        model_id=model_id,
        status=status.status,
        stage=status.stage,
        progress=status.progress,
        current_size_gb=status.current_size_gb,
        accuracy_retention=status.accuracy_retention,
        error=status.error,
        created_at=status.created_at,
        updated_at=status.updated_at,
        stage_started_at=status.stage_started_at,
        elapsed_seconds=elapsed_seconds,
        stage_elapsed_seconds=stage_elapsed_seconds,
        target_size_gb=status.target_size_gb,
        distill=status.distill,
        stages=_build_stage_items(status),
        metrics=_build_metrics(status),
        report=report,
        warnings=list(report.get("warnings", [])) if report else [],
        logs=_tail_lines(output_dir / "compress.log", limit=50),
    )


def _build_stats_payload() -> StatsResponse:
    """Build the flat dashboard statistics payload from persisted job data."""

    stats = job_manager.get_stats()
    return StatsResponse(
        total_jobs=stats.total_jobs,
        jobs_completed=stats.jobs_completed,
        jobs_failed=stats.jobs_failed,
        average_compression_ratio=stats.average_compression_ratio,
        average_accuracy_retention=stats.average_accuracy_retention,
        total_storage_saved_gb=stats.total_storage_saved_gb,
    )


@router.post("/compress", response_model=CompressResponse)
def compress_model(
    request: CompressRequest,
    http_request: Request,
    background_tasks: BackgroundTasks,
    _: str = Depends(require_api_key),
) -> CompressResponse:
    estimated_size_gb = estimate_model_source_size_gb(
        request.model_id,
        hf_token=request.hf_token,
    )
    if estimated_size_gb is not None and estimated_size_gb > 200:
        raise HTTPException(status_code=413, detail="Model exceeds the 200GB limit.")

    jobs_root = Path("jobs")
    jobs_root.mkdir(parents=True, exist_ok=True)

    job_id = uuid4().hex
    output_dir = jobs_root / job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    config = CompressionConfig(
        model_id=request.model_id,
        output_dir=output_dir,
        target_size_gb=request.target_size_gb,
        distill=request.distill,
        resume=request.resume,
        log_file=output_dir / "compress.log",
        hf_token=request.hf_token,
        head_prune_threshold=request.head_pruning_threshold,
        redundancy_threshold=request.layer_redundancy_threshold,
    )

    if request.quantization_mode == "int4":
        config.quant_default_bits = 4
        config.quant_sensitive_bits = 4
    elif request.quantization_mode == "int8":
        config.quant_default_bits = 8
        config.quant_sensitive_bits = 8
    else:
        config.quant_default_bits = 4
        config.quant_sensitive_bits = 8

    if request.domain_data_url:
        config.domain_data = download_text(
            str(request.domain_data_url),
            output_dir / "domain_data.txt",
        )
        config.distill = True

    if request.calibration_data_text:
        calibration_path = output_dir / "calibration_data.txt"
        calibration_path.write_text(request.calibration_data_text, encoding="utf-8")
        config.calibration_data = calibration_path
    elif request.calibration_data_path:
        config.calibration_data = Path(request.calibration_data_path)

    write_json(
        output_dir / "request.json",
        {
            "model_id": request.model_id,
            "target_size_gb": request.target_size_gb,
            "distill": config.distill,
            "domain_data_url": str(request.domain_data_url)
            if request.domain_data_url
            else None,
            "resume": request.resume,
            "webhook_url": str(request.webhook_url)
            if request.webhook_url
            else None,
            "quantization_mode": request.quantization_mode,
            "head_pruning_threshold": request.head_pruning_threshold,
            "layer_redundancy_threshold": request.layer_redundancy_threshold,
            "calibration_data_path": str(config.calibration_data)
            if config.calibration_data
            else None,
        },
    )

    try:
        status = job_manager.enqueue_job(
            config,
            notify_url=str(request.webhook_url) if request.webhook_url else None,
            job_id=job_id,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc

    download_url = str(http_request.base_url).rstrip("/") + f"/download/{job_id}"
    background_tasks.add_task(
        job_manager.start_job,
        job_id,
        config,
        notify_url=str(request.webhook_url) if request.webhook_url else None,
        download_url=download_url,
    )
    return CompressResponse(job_id=status.job_id, status=status.status)


@router.get("/status/{job_id}", response_model=StatusResponse)
def get_status(
    job_id: str,
    _: str = Depends(require_api_key),
) -> StatusResponse:
    status = job_manager.get_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return StatusResponse(
        job_id=status.job_id,
        model_id=status.model_id,
        stage=status.stage,
        progress=status.progress,
        current_size_gb=status.current_size_gb,
        accuracy_retention=status.accuracy_retention,
        status=status.status,
        error=status.error,
        created_at=status.created_at,
        updated_at=status.updated_at,
    )


@router.get("/jobs", response_model=JobListResponse)
def list_jobs(
    limit: int = Query(default=25, ge=1, le=100),
    _: str = Depends(require_api_key),
) -> JobListResponse:
    jobs = job_manager.list_jobs(limit=limit)
    return JobListResponse(jobs=[_summarize_job(job) for job in jobs])


@router.get("/jobs/{job_id}", response_model=JobDetailResponse)
def get_job_detail(
    job_id: str,
    _: str = Depends(require_api_key),
) -> JobDetailResponse:
    status = job_manager.get_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return _build_job_detail(status)


@router.get("/stats", response_model=StatsResponse)
def get_stats(
    _: str = Depends(require_api_key),
) -> StatsResponse:
    """Return aggregate job statistics for the dashboard."""

    try:
        return _build_stats_payload()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Unable to compute compression statistics.",
        ) from exc


@router.get("/download/{job_id}")
def download_job(
    job_id: str,
    _: str = Depends(require_api_key),
) -> FileResponse:
    status = job_manager.get_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if status.status != "DONE":
        raise HTTPException(status_code=409, detail="Job has not completed yet.")
    archive = zip_directory(_normalize_output_dir(status))
    return FileResponse(path=archive, filename=archive.name)
