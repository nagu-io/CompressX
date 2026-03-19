# CompressX

CompressX is a production-focused Python LLM compression pipeline for supported Hugging Face causal language models.

Supported architectures:

- `LlamaForCausalLM`
- `MistralForCausalLM`
- `GPT2LMHeadModel`
- `FalconForCausalLM`

## Project Layout

```text
compressx/
├── core/
│   ├── analyzer.py
│   ├── quantizer.py
│   ├── pruner.py
│   ├── distiller.py
│   └── evaluator.py
├── api/
│   ├── main.py
│   └── routes.py
├── cli/
│   └── compress.py
├── tests/
│   ├── test_analyzer.py
│   ├── test_quantizer.py
│   └── test_pruner.py
├── configs/
│   └── default_config.yaml
├── Dockerfile
├── requirements.txt
├── setup.sh
└── README.md
```

## Installation

1. Clone the repo.
2. Set `HF_TOKEN` if you need gated models.
3. Install dependencies.

Linux or WSL:

```bash
./setup.sh
```

Manual install:

```bash
pip install -r requirements.txt
pip install -e .
```

The install script pins exact versions and installs `bitsandbytes` and `auto-gptq` before the rest of the stack when CUDA is available.

## Configuration

Default YAML config: [compressx/configs/default_config.yaml](d:/projects/ram/compressx/configs/default_config.yaml)

You can also place a project-local `compress_config.yaml` in the repo root.

## Hardware Requirements

| Model Size | Min RAM | Min VRAM | Est. Time |
| --- | --- | --- | --- |
| 7B | 16GB | 8GB | 20 min |
| 13B | 32GB | 16GB | 45 min |
| 70B | 128GB | 40GB | 3 hours |

Local development environment used for this build:

- OS: Windows
- Python: 3.11.9
- RAM: about 16 GB
- GPU: RTX 3050 Laptop GPU with 4 GB VRAM

## Example Commands

Small model:

```bash
python compress.py --model gpt2 --output .\compressed_gpt2 --target-size 1 --report
```

Expected output:

```text
+---------------+------------+--------------------+-----------+
| Original Size | Final Size | Accuracy Retention | Time Taken |
+---------------+------------+--------------------+-----------+
```

Large model with resume and offloading:

```bash
python compress.py ^
  --model meta-llama/Llama-3-8B ^
  --output .\compressed_llama ^
  --target-size 3 ^
  --calibration-data .\calib_data.txt ^
  --resume ^
  --report
```

Expected output:

```text
Compressed model written to <output_dir>\model.safetensors
```

API mode:

```bash
uvicorn compressx.api.main:app --host 0.0.0.0 --port 8000
```

Sample request:

```bash
curl -X POST "http://127.0.0.1:8000/compress" ^
  -H "Content-Type: application/json" ^
  -H "X-API-Key: compressx-dev-key" ^
  -d "{\"model_id\":\"gpt2\",\"target_size_gb\":1,\"distill\":false,\"resume\":false}"
```

Expected output:

```json
{"job_id":"<job_id>","status":"QUEUED"}
```

## API Notes

- Header auth: `X-API-Key`
- Default local API key: `compressx-dev-key`
- Max 3 concurrent jobs
- Max model size: 200GB
- Job states: `QUEUED -> ANALYZING -> QUANTIZING -> PRUNING -> DISTILLING -> EVALUATING -> DONE`
- SQLite job store: `jobs.db`

## Output Files

Compression output directory:

```text
<output_dir>/
├── model.safetensors
├── config.json
├── tokenizer.json
├── compression_report.json
├── sensitivity_report.json
├── head_sensitivity_report.json
├── quantization_plan.json
├── pruning_log.json
└── adapter/
```

Checkpoints are written under `./checkpoints/<output-name>/`.

## Troubleshooting

`Model not found or gated`

- Set `HF_TOKEN`
- Verify the model id
- Confirm the account can access the repo

`CUDA not available`

- CompressX automatically falls back to CPU mode
- Expect slower stage execution

`Calibration data too small`

- Provide at least 128 non-empty samples
- Use the default `wikitext-2-raw-v1` path if needed

`Insufficient RAM or VRAM`

- Reduce concurrent workloads
- Use `--execution-device cpu`
- Resume from checkpoints after increasing available memory

`Output directory is not writable`

- Change `output_dir`
- Check permissions and free disk space

## Development

Run tests:

```bash
python -m pytest -q
```

Main entry points:

- CLI: [compress.py](d:/projects/ram/compress.py)
- Core analyzer: [compressx/core/analyzer.py](d:/projects/ram/compressx/core/analyzer.py)
- API app: [compressx/api/main.py](d:/projects/ram/compressx/api/main.py)
