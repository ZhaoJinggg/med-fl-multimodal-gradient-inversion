"""Shared pieces: paths, data loading, the fusion model, and parameter groups."""
import csv
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "pretrained", "hf"))
os.environ.setdefault("TORCH_HOME", os.path.join(ROOT, "pretrained", "torch"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from transformers import GPT2Model, GPT2TokenizerFast  # noqa: E402

MAX_TOKENS = 64
FEATURE_NORM = os.path.join(ROOT, "pretrained", "feature_norm_seed0.pt")
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ---------------------------------------------------------------- data
def load_pairs():
    with open(os.path.join(ROOT, "data", "splits", "pairs.csv"), encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_seed(seed):
    return json.load(open(os.path.join(ROOT, "data", "splits", f"seed{seed}.json")))


def load_images(pairs, res):
    """Gray images in [0, 1], shape (N, 1, res, res), cached as uint8 per resolution."""
    from PIL import Image
    cache = os.path.join(ROOT, "data", "cache", f"images{res}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        if list(z["ids"]) == [p["study"] for p in pairs]:
            return torch.from_numpy(z["x"]).float().div(255).unsqueeze(1)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    img_dir = os.path.join(ROOT, "data", "iuxray", "extracted", "images")
    x = np.stack([np.asarray(Image.open(os.path.join(img_dir, p["image_id"] + ".png")).convert("L")
                             .resize((res, res), Image.BICUBIC)) for p in pairs])
    np.savez(cache, x=x, ids=np.array([p["study"] for p in pairs]))
    return torch.from_numpy(x).float().div(255).unsqueeze(1)


def tokenizer():
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    return tok


def tokenize(tok, texts):
    t = tok(texts, truncation=True, max_length=MAX_TOKENS, padding="max_length", return_tensors="pt")
    return t["input_ids"], t["attention_mask"]


class PairData:
    """All pairs in memory: images (N,1,R,R), token ids, masks, labels; index by study id."""

    def __init__(self, res=224):
        self.pairs = load_pairs()
        self.x = load_images(self.pairs, res)
        self.ids, self.mask = tokenize(tokenizer(), [p["findings"] for p in self.pairs])
        self.y = torch.tensor([float(p["label"]) for p in self.pairs])
        self.index = {p["study"]: i for i, p in enumerate(self.pairs)}

    def rows(self, studies):
        return torch.tensor([self.index[s] for s in studies])

    def split(self, name):
        return torch.tensor([i for i, p in enumerate(self.pairs) if p["split"] == name])

    def batch(self, rows, device):
        return (self.x[rows].to(device), self.ids[rows].to(device),
                self.mask[rows].to(device), self.y[rows].to(device))


# ---------------------------------------------------------------- model
class FusionModel(nn.Module):
    """ResNet-18 (BN) on a gray image, GPT-2 small on the findings text, concatenation, one logit."""

    def __init__(self):
        super().__init__()
        r = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        r.fc = nn.Identity()
        self.img = r
        self.txt = GPT2Model.from_pretrained("gpt2", resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
        self.head = nn.Linear(512 + 768, 1)
        self.register_buffer("im_mean", IMAGENET_MEAN.clone(), persistent=False)
        self.register_buffer("im_std", IMAGENET_STD.clone(), persistent=False)
        # fixed per-dimension standardization of the two feature vectors before the head (part of the
        # model definition; computed once by src/feature_norm.py from the pretrained encoders on the
        # training pairs of calibration seed 0). Without it, GPT-2's last-token feature has one dimension
        # near 220 and plain SGD on the head oscillates.
        fn = torch.load(FEATURE_NORM) if os.path.exists(FEATURE_NORM) else None
        for k, d in (("img", 512), ("txt", 768)):
            mu = fn[k + "_mu"] if fn else torch.zeros(d)
            sd = fn[k + "_sd"] if fn else torch.ones(d)
            self.register_buffer(k + "_mu", mu.clone(), persistent=False)
            self.register_buffer(k + "_sd", sd.clone(), persistent=False)

    def image_features(self, x):
        x = (x.repeat(1, 3, 1, 1) - self.im_mean) / self.im_std
        return self.img(x)

    def text_features(self, ids, mask):
        h = self.txt(input_ids=ids, attention_mask=mask).last_hidden_state
        last = mask.sum(1) - 1
        return h[torch.arange(h.shape[0], device=h.device), last]

    def forward(self, x, ids, mask, h_txt=None):
        h_img = self.image_features(x)
        if h_txt is None:
            h_txt = self.text_features(ids, mask)
        z = torch.cat([(h_img - self.img_mu) / self.img_sd, (h_txt - self.txt_mu) / self.txt_sd], 1)
        return self.head(z).squeeze(1)


def is_image_param(name):
    """Group A = ResNet-18 + head (one learning rate); group B = GPT-2."""
    return not name.startswith("txt.")


def sgd_step(model, loss, lr_img, lr_txt, prox=None, mu=0.0):
    """Plain SGD on both groups (no momentum, no weight decay); optional FedProx term."""
    params = [p for p in model.parameters() if p.requires_grad]
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, params)
    with torch.no_grad():
        for n, p, g in zip(names, params, grads):
            if prox is not None and mu > 0:
                g = g + mu * (p - prox[n])
            p -= (lr_img if is_image_param(n) else lr_txt) * g


def bce(logit, y):
    return F.binary_cross_entropy_with_logits(logit, y)
