# SAMTok RefCOCO-M Benchmark

Benchmark five SAMTok variants on the RefCOCO-M dataset, reporting mIoU, precision, recall, and F1.

## Setup

1. Create a Python environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Clone Sa2VA and set the import path:
   ```bash
   git clone https://github.com/bytedance/Sa2VA.git
   export SAMTOK_SA2VA_PATH=/path/to/Sa2VA
   ```

3. (Optional) Ensure you can access the model weights on Hugging Face if needed.

## Run a Single Model

```bash
python benchmark.py --model zhouyik/Qwen3-VL-8B-SAMTok
```

Options:
- `--max-samples N` to limit the number of expressions evaluated
- `--save-per-sample` to store per-expression metrics
- `--output-dir results` to change output location

## Run All Models (Parallel GPUs)

```bash
chmod +x run_all.sh
./run_all.sh
```

Each process writes a JSON file into `results/` named after the model.

## Output Format

Each result JSON includes:
- `metrics.mIoU`
- `metrics.precision`
- `metrics.recall`
- `metrics.f1`
