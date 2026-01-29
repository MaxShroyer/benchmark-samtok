# SAMTok RefCOCO-M Benchmark

Benchmark SAMTok variants on RefCOCO-style datasets, reporting mIoU, precision, recall, F1,
plus gIoU/cIoU/N-acc for comparison with paper metrics.

## Setup

1. Create a Python environment and install dependencies (Qwen2.5-VL needs transformers >= 4.49.0):

   ```bash
   pip install -r requirements.txt
   ```

2. Clone Sa2VA and set the import path:

   ```bash
   git clone https://github.com/bytedance/Sa2VA.git
   export SAMTOK_SA2VA_PATH=/path/to/Sa2VA
   ```

3. (Optional) Ensure you can access gated Hugging Face assets if needed:

   ```bash
   export HF_TOKEN=your_hf_token
   ```

4. (Optional) Enable Weights & Biases logging:

   ```bash
   export WANDB_API_KEY=your_wandb_key
   ```

## Run a Single Model

```bash
python benchmark.py --model zhouyik/Qwen3-VL-8B-SAMTok
```

Options:

- `--max-samples N` to limit the number of expressions evaluated
- `--save-per-sample` to store per-expression metrics
- `--output-dir results` to change output location
- `--dataset` accepts comma-separated dataset names for comparison
- `--split` accepts comma-separated splits; if omitted, dataset defaults are used
- `--no-wandb` to disable W&B logging even if `WANDB_API_KEY` is set
- `--wandb-project`, `--wandb-entity`, `--wandb-run-name`, `--wandb-tags` to customize W&B runs
- `--wandb-log-every N` to control streaming log frequency

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
- `metrics.gIoU`
- `metrics.cIoU`
- `metrics.n_acc`

## RefCOCO+ Comparison (Shared Dataset)

Example comparison across two RefCOCO+ sources:

```bash
python benchmark.py \
  --model zhouyik/Qwen3-VL-8B-SAMTok \
  --dataset moondream/refcoco_plus_rle_val,lmms-lab/RefCOCOplus \
  --split refcoco_plus_val,validation
```
