from __future__ import annotations

from pathlib import Path

from compressx.pipeline import CompressionPipeline


def test_pipeline_target_optimization_stops_on_accuracy_limit(
    compression_context,
    monkeypatch,
) -> None:
    compression_context.config.target_size_gb = 1.0
    compression_context.config.min_accuracy_retention_percent = 90.0
    compression_context.config.skip_evaluation = False

    pipeline = CompressionPipeline(compression_context.config)

    monkeypatch.setattr(
        pipeline,
        "_build_context",
        lambda **kwargs: compression_context,
    )

    def fake_run_stage(context, stage_name, stage_instance, *, save_model_checkpoint):
        if stage_name == "sensitivity":
            context.sensitivity_report = {
                "model.layers.0": 0.2,
                "model.layers.1": 0.8,
                "model.layers.2": 0.1,
            }
            return

        if stage_name == "quantization":
            context.quantization_plan = {
                "model.layers.0": context.config.quant_default_bits,
                "model.layers.1": context.config.quant_sensitive_bits,
                "model.layers.2": context.config.quant_default_bits,
            }
            return

        if stage_name == "pruning":
            context.pruning_log = {"layers_removed": [], "heads_removed": {}}
            return

        if stage_name == "evaluation":
            retention_by_bits = {4: 96.0, 3: 92.0, 2: 88.0}
            context.accuracy_retention_percent = retention_by_bits.get(
                context.config.quant_default_bits,
                88.0,
            )

    monkeypatch.setattr(pipeline, "_run_stage", fake_run_stage)

    def fake_refresh_metrics(context):
        size_by_bits = {4: 4.0, 3: 2.5, 2: 1.5}
        context.current_size_gb = size_by_bits.get(context.config.quant_default_bits, 1.5)

    monkeypatch.setattr(pipeline, "_refresh_compression_metrics", fake_refresh_metrics)

    def fake_export_model(context):
        output_file = context.config.output_dir / "model.safetensors"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_bytes(b"compressed")
        context.exported_model_path = output_file
        return output_file

    monkeypatch.setattr("compressx.pipeline.export_model", fake_export_model)
    monkeypatch.setattr(
        pipeline.checkpoints,
        "record_stage",
        lambda context, stage_name, save_model: None,
    )

    context = pipeline.run()

    assert context.target_stop_reason == "accuracy_limit_reached"
    assert context.optimization_passes == 3
    assert context.stage_details["target_optimization"]["passes"]
    assert Path(context.exported_model_path).exists()
