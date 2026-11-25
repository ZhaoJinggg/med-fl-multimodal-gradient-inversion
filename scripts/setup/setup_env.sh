#!/bin/bash
set -e
P=~/jacob/mmgia
source ~/miniconda3/etc/profile.d/conda.sh
conda create -y -n mmgia python=3.11
conda activate mmgia
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install transformers lpips scikit-image rouge-score scipy pandas matplotlib seaborn statsmodels tqdm pyyaml lxml
export HF_HOME=$P/pretrained/hf TORCH_HOME=$P/pretrained/torch
python - <<PY
import torch, torchvision
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.device_count())
from transformers import GPT2Model, GPT2TokenizerFast
GPT2TokenizerFast.from_pretrained("gpt2"); GPT2Model.from_pretrained("gpt2")
torchvision.models.resnet18(weights="IMAGENET1K_V1")
import lpips; lpips.LPIPS(net="alex")
print("WEIGHTS CACHED")
PY
echo SETUP_DONE
