# CompressX 🗜️

> Compress any LLM to 25% of its original size.

![Dashboard](assets/dashboard.png)

## Real Results

| Model | Original | Compressed | Ratio | Accuracy |
|-------|----------|------------|-------|----------|
| facebook/opt-125m | 0.467 GB | 0.130 GB | 3.6x | 100% |
| Mistral-7B | 14 GB | ~3.5 GB | 4x | 92% |

## How It Works
```
Input: Any HuggingFace Model
         ↓
Stage 1: Sensitivity Analysis
         ↓
Stage 2: Mixed Precision Quantization (INT4/INT8)
         ↓
Stage 3: 1-bit QEP (Error Propagation Correction)
         ↓
Stage 4: Structural Pruning
         ↓
Stage 5: Accuracy Evaluation
         ↓
Output: Compressed Model + Report
```

## Quick Start
```bash
git clone https://github.com/YOURNAME/CompressX
cd CompressX
pip install -r requirements.txt

python compress.py \
  --model facebook/opt-125m \
  --output ./compressed \
  --target-size 0.1 \
  --report
```

## Presets
```bash
# Fast (2x compression, high accuracy)
python compress.py --model ... --preset fast

# Balanced (3x compression, good accuracy)  
python compress.py --model ... --preset balanced

# Mobile (4x compression, for edge devices)
python compress.py --model ... --preset mobile

# Extreme (maximum compression)
python compress.py --model ... --preset extreme
```

## Features

- Multi-architecture support (Llama, Mistral, GPT2, Falcon, OPT, Bloom, MPT, Mixtral)
- 1-bit QEP quantization with error propagation
- Target-driven optimization loop
- Resume interrupted jobs
- FastAPI with job queue
- React dashboard
- 42 passing tests

## Dashboard
```bash
# Start backend
uvicorn compressx.api.main:app --reload

# Start frontend
cd frontend && npm install && npm run dev
```

Open http://localhost:5173

## API
```bash
# Submit compression job
curl -X POST http://localhost:8000/compress \
  -H "X-API-Key: compressx-dev-key" \
  -H "Content-Type: application/json" \
  -d '{"model_id": "facebook/opt-125m", "target_size_gb": 0.1}'

# Check status
curl http://localhost:8000/status/{job_id} \
  -H "X-API-Key: compressx-dev-key"
```

## Hardware Requirements

| Model Size | Min RAM | Min VRAM | Est. Time |
|------------|---------|----------|-----------|
| 7B | 16GB | 16GB | 2-3 hours |
| 13B | 32GB | 24GB | 4-5 hours |
| 70B | 128GB | 80GB | 12-15 hours |

## Built With

- Python 3.11
- PyTorch
- HuggingFace Transformers
- FastAPI
- React 18
- TailwindCSS

## License

MIT

## Contact

Built by Nagu (Pothuraju Nagababu)
LinkedIn: [your linkedin]
Email: your@email.com
```

---

## Before You Push — Create .gitignore

Make sure these are ignored:
```
# In your .gitignore add:

# Model files (too large for GitHub)
*.bin
*.safetensors
*.gguf
*.pt
*.pth

# Output folders
test_output*/
compressed*/
mistral_*/
llama_*/
offload/
checkpoints/
logs/

# Environment
.env
*.env
venv/
__pycache__/

# Database
*.db

# Node
frontend/node_modules/
frontend/dist/

# OS
.DS_Store
Thumbs.db
```

---

## The assets Folder
```
Before pushing:
  Create folder: assets/
  Save your dashboard screenshot as: assets/dashboard.png

This makes the README show the dashboard image
on the GitHub page automatically.

That screenshot alone will get you extra stars.
```

---

## Final Checklist Before Pushing
```
✅ Repository name: CompressX
✅ Description filled in
✅ Public selected
✅ MIT license selected
✅ .gitignore has model files excluded
✅ README.md updated with real numbers
✅ Dashboard screenshot in assets/folder
✅ Your LinkedIn/email in README
✅ Topics added after creation
