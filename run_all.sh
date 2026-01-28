#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=0 python benchmark.py --model zhouyik/Qwen3-VL-8B-SAMTok &
CUDA_VISIBLE_DEVICES=1 python benchmark.py --model zhouyik/Qwen3-VL-4B-SAMTok &
CUDA_VISIBLE_DEVICES=2 python benchmark.py --model zhouyik/Qwen2.5-VL-7B-SAMTok-co &
CUDA_VISIBLE_DEVICES=3 python benchmark.py --model zhouyik/Qwen2.5-VL-3B-SAMTok-co &
CUDA_VISIBLE_DEVICES=4 python benchmark.py --model zhouyik/Qwen3-VL-4B-SAMTok-co &

wait
