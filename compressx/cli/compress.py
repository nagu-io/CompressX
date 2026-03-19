from __future__ import annotations

from pathlib import Path
from typing import Any

import click

from compressx.config_loader import build_config
from compressx.pipeline import CompressionPipeline


def _default_config_path() -> Path | None:
    default_path = Path("compress_config.yaml")
    if default_path.exists():
        return default_path
    packaged_default = Path(__file__).resolve().parents[1] / "configs" / "default_config.yaml"
    return packaged_default if packaged_default.exists() else None


@click.command()
@click.option(
    "--config-file",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Optional YAML config file. Defaults to ./compress_config.yaml when present.",
)
@click.option("--model", "model_id", default=None, help="Hugging Face model id or path.")
@click.option(
    "--output",
    "output_dir",
    default=None,
    type=click.Path(path_type=Path),
    help="Output directory for the compressed model.",
)
@click.option("--target-size", "target_size_gb", default=None, type=float)
@click.option(
    "--calibration-data",
    type=click.Path(exists=True, path_type=Path),
    default=None,
)
@click.option("--distill/--no-distill", default=None)
@click.option(
    "--domain-data",
    type=click.Path(exists=True, path_type=Path),
    default=None,
)
@click.option("--report/--no-report", default=None)
@click.option("--resume/--no-resume", default=None)
@click.option("--aggressive/--no-aggressive", default=None)
@click.option("--use-qep/--no-use-qep", default=None)
@click.option("--qep-threshold", default=None, type=float)
@click.option("--skip-sensitivity/--run-sensitivity", default=None)
@click.option("--skip-quantization/--run-quantization", default=None)
@click.option("--skip-pruning/--run-pruning", default=None)
@click.option("--skip-evaluation/--run-evaluation", default=None)
@click.option("--distillation-steps", default=None, type=int)
@click.option("--head-prune-threshold", default=None, type=float)
@click.option("--redundancy-threshold", default=None, type=float)
@click.option(
    "--offload-dir",
    type=click.Path(path_type=Path),
    default=None,
)
@click.option(
    "--checkpoint-dir",
    type=click.Path(path_type=Path),
    default=None,
)
@click.option(
    "--execution-device",
    type=click.Choice(["auto", "cpu", "cuda"]),
    default=None,
)
def main(
    *,
    config_file: Path | None,
    model_id: str | None,
    output_dir: Path | None,
    target_size_gb: float | None,
    calibration_data: Path | None,
    distill: bool | None,
    domain_data: Path | None,
    report: bool | None,
    resume: bool | None,
    aggressive: bool | None,
    use_qep: bool | None,
    qep_threshold: float | None,
    skip_sensitivity: bool | None,
    skip_quantization: bool | None,
    skip_pruning: bool | None,
    skip_evaluation: bool | None,
    distillation_steps: int | None,
    head_prune_threshold: float | None,
    redundancy_threshold: float | None,
    offload_dir: Path | None,
    checkpoint_dir: Path | None,
    execution_device: str | None,
) -> None:
    selected_config = config_file or _default_config_path()
    overrides: dict[str, Any] = {
        "model_id": model_id,
        "output_dir": output_dir,
        "target_size_gb": target_size_gb,
        "calibration_data": calibration_data,
        "distill": distill,
        "domain_data": domain_data,
        "report": report,
        "resume": resume,
        "aggressive": aggressive,
        "use_qep": use_qep,
        "qep_threshold": qep_threshold,
        "skip_sensitivity": skip_sensitivity,
        "skip_quantization": skip_quantization,
        "skip_pruning": skip_pruning,
        "skip_evaluation": skip_evaluation,
        "distillation_steps": distillation_steps,
        "head_prune_threshold": head_prune_threshold,
        "redundancy_threshold": redundancy_threshold,
        "offload_dir": offload_dir,
        "checkpoint_dir": checkpoint_dir,
        "execution_device": execution_device,
    }

    try:
        config = build_config(overrides, selected_config)
    except TypeError as exc:
        raise click.ClickException(
            "Missing required configuration. Provide --model, --output, and --target-size "
            "or define them in compress_config.yaml."
        ) from exc
    config.log_file = config.output_dir / "compress.log"

    pipeline = CompressionPipeline(config)
    context = pipeline.run()

    if config.report:
        click.echo((config.output_dir / "compression_report.json").read_text(encoding="utf-8"))
    click.echo(f"Compressed model written to {context.exported_model_path}")
