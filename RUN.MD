python -m venv .venv  ...

git clone https://github.com/MaxShroyer/benchmark-samtok.git
git clone https://github.com/bytedance/Sa2VA.git

apt install python3.10-venv -Y
touch .env
vi .env

apt install 
python3 -m venv .venv
. .venv/bin/ativate
cd benchmark-samtok
pip install -r req..


python benchmark.py \
  --model zhouyik/Qwen3-VL-8B-SAMTok \
  --dataset moondream/refcoco-m,moondream/refcoco_rle,moondream/refcoco_plus_rle_val,moondream/lvis_segmentation,lmms-lab/RefCOCOplus

Notes for long runs / avoiding lost progress:
- Add `--checkpoint-every 50` to write an on-disk checkpoint periodically.
- If the process OOMs or gets interrupted, re-run with `--resume` to skip already-processed expressions.
- SAM2 decoding VRAM is controlled by `--sam2-decode-batch-size` (default 1; lowest VRAM).

  python benchmark.py \
  --model zhouyik/Qwen3-VL-4B-SAMTok \
  --dataset moondream/refcoco-m,moondream/refcoco_rle,moondream/refcoco_plus_rle_val,moondream/lvis_segmentation,lmms-lab/RefCOCOplus

  python benchmark.py \
  --model zhouyik/Qwen2.5-VL-7B-SAMTok-co \
  --dataset moondream/refcoco-m,moondream/refcoco_rle,moondream/refcoco_plus_rle_val,moondream/lvis_segmentation,lmms-lab/RefCOCOplus

python benchmark.py \
  --model zhouyik/Qwen2.5-VL-3B-SAMTok-co \
  --dataset moondream/refcoco-m,moondream/refcoco_rle,moondream/refcoco_plus_rle_val,moondream/lvis_segmentation,lmms-lab/RefCOCOplus

python benchmark.py \
  --model zhouyik/Qwen3-VL-8B-SAMTok \
  --dataset moondream/lvis_segmentation,lmms-lab/RefCOCOplus

----///////////////////////////////
Running 
python benchmark.py \
  --model zhouyik/Qwen3-VL-8B-SAMTok \
  --dataset moondream/refcoco_plus_rle_val

python benchmark.py \
  --model zhouyik/Qwen3-VL-4B-SAMTok \
  --dataset moondream/refcoco_plus_rle_val


---////////////////////////////////
running 
python benchmark.py \
  --model zhouyik/Qwen3-VL-8B-SAMTok \
  --dataset moondream/refcoco-m

python benchmark.py \
  --model zhouyik/Qwen3-VL-4B-SAMTok \
  --dataset moondream/refcoco-m


-----

python benchmark.py \
  --model zhouyik/Qwen3-VL-8B-SAMTok \
  --dataset moondream/refcoco_rle

python benchmark.py \
  --model zhouyik/Qwen3-VL-4B-SAMTok \
  --dataset moondream/refcoco_rle

-----XXXXXXXXXXXXXXX
brokern
  python benchmark.py \
  --model zhouyik/Qwen3-VL-8B-SAMTok \
  --dataset moondream/lvis_segmentation

  python benchmark.py \
  --model zhouyik/Qwen3-VL-4B-SAMTok \
  --dataset moondream/lvis_segmentation



-----///////////////////////////
Running 
  python benchmark.py \
  --model zhouyik/Qwen3-VL-8B-SAMTok \
  --dataset lmms-lab/RefCOCOplus

  python benchmark.py \
  --model zhouyik/Qwen3-VL-4B-SAMTok \
  --dataset lmms-lab/RefCOCOplus

