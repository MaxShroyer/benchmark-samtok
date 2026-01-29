ssh  -i ~/.ssh/primeintellect_ed25519 ...

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

  python benchmark.py \
  --model zhouyik/Qwen3-VL-4B-SAMTok \
  --dataset moondream/refcoco-m,moondream/refcoco_rle,moondream/refcoco_plus_rle_val,moondream/lvis_segmentation,lmms-lab/RefCOCOplus

  python benchmark.py \
  --model zhouyik/Qwen2.5-VL-7B-SAMTok-co \
  --dataset moondream/refcoco-m,moondream/refcoco_rle,moondream/refcoco_plus_rle_val,moondream/lvis_segmentation,lmms-lab/RefCOCOplus

python benchmark.py \
  --model zhouyik/Qwen2.5-VL-3B-SAMTok-co \
  --dataset moondream/refcoco-m,moondream/refcoco_rle,moondream/refcoco_plus_rle_val,moondream/lvis_segmentation,lmms-lab/RefCOCOplus