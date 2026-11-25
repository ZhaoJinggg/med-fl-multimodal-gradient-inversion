"""Control V1: the reference gradient-matching code of Geiping et al. (third_party/invertinggradients,
commit 1157b61) on the same victim gradient, the same initialization and the same budget as
src/attack_image.py at u = 1 with --signed --bn 0. The two results should agree.

The reference optimizer sees only the image-side parameters (ResNet-18 + head); the text feature is
fixed (oracle), BN stays in training mode, and the loss is the binary cross-entropy of the study.

usage: python src/calibration/control_reference.py --global runs/pilot_r128/seed0_fedavg/global_round9.pt --seed 0
          --victims 0 1 2 --res 128 --tv 1e-2 --iters 5000 --out runs/pilot/tune128
"""
import argparse
import json
import os
import sys
import time

import torch
from skimage.metrics import structural_similarity

import os as _os, sys as _sys  # noqa: E401,E402
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # src/ (common.py)
from common import ROOT, FusionModel, PairData, bce, load_seed

sys.path.insert(0, os.path.join(ROOT, "third_party", "invertinggradients"))
from inversefed.reconstruction_algorithms import GradientReconstructor  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--global", dest="glob", required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--victims", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--iters", type=int, default=5000)
ap.add_argument("--opt-lr", type=float, default=0.1)
ap.add_argument("--tv", type=float, default=1e-2)
ap.add_argument("--res", type=int, default=128)
ap.add_argument("--out", default="runs/pilot/tune128")
ap.add_argument("--tag", default="_ref")
a = ap.parse_args()
dev = "cuda"
out = os.path.join(ROOT, a.out)
os.makedirs(out, exist_ok=True)

data = PairData(a.res)
split = load_seed(a.seed)
model = FusionModel().to(dev)
model.load_state_dict(torch.load(os.path.join(ROOT, a.glob), map_location=dev))
model.train()
pub_img = data.x[torch.cat([data.split("val"), data.split("test")])].mean(0, keepdim=True)


class ImageSide(torch.nn.Module):
    """The fusion model as a function of the image only; parameters = ResNet-18 + head."""

    def __init__(self, fusion, h_txt):
        super().__init__()
        self.img, self.head = fusion.img, fusion.head
        self.fusion = [fusion]  # not registered: GPT-2 stays out of parameters()
        self.h_txt = h_txt

    def forward(self, x):
        f = self.fusion[0]
        h_img = f.img((x.repeat(1, 3, 1, 1) - f.im_mean) / f.im_std)
        z = torch.cat([(h_img - f.img_mu) / f.img_sd, (self.h_txt - f.txt_mu) / f.txt_sd], 1)
        return self.head(z).squeeze(1)


class FixedInit(GradientReconstructor):
    def _init_images(self, img_shape):
        return self.x0.clone().unsqueeze(0)


def ssim(o, r):
    return structural_similarity(o, r, data_range=1.0, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)


results = []
for v in a.victims:
    studies = split["victim_batches"][str(v)][:1]
    x, ids, mask, y = data.batch(data.rows(studies), dev)
    with torch.no_grad():
        h_txt = model.text_features(ids, mask)
    side = ImageSide(model, h_txt).train()
    grad = [g.detach() for g in torch.autograd.grad(bce(side(x), y), list(side.parameters()))]

    torch.manual_seed(1000 + v)  # same start as attack_image.py --init mean, restart 0
    x0 = (pub_img + 0.05 * torch.randn(1, 1, a.res, a.res)).clamp(0, 1).to(dev)
    cfg = dict(signed=True, boxed=True, cost_fn="sim", indices="def", weights="equal", lr=a.opt_lr, optim="adam",
               restarts=1, max_iterations=a.iters, total_variation=a.tv, init="zeros", filter="none", lr_decay=True,
               scoring_choice="loss")
    rec = FixedInit(side, (torch.zeros(1, 1, 1, device=dev), torch.ones(1, 1, 1, device=dev)), cfg, num_images=1)
    rec.loss_fn = bce
    rec.x0 = x0
    t0 = time.time()
    xr, stats = rec.reconstruct(grad, y, img_shape=(1, a.res, a.res), eval=False)
    s = ssim(x[0, 0].cpu().numpy(), xr[0, 0].cpu().numpy())
    r = {"victim": v, "batch": 1, "u": 1, "init": "mean", "res": a.res, "tv": a.tv, "iters": a.iters,
         "ssim": round(float(s), 4), "cos": round(float(stats["opt"]), 5), "minutes": round((time.time() - t0) / 60, 2)}
    results.append(r)
    print(f"victim {v} FINAL ssim={s:.4f} cos={stats['opt']:.4f}{a.tag}", flush=True)
    name = f"v{v}_B1_u1_mean{a.tag}"
    torch.save({"rec": xr.cpu(), "orig": x.cpu(), "labels": y.cpu(), "studies": studies}, os.path.join(out, name + ".pt"))
    json.dump(r, open(os.path.join(out, name + ".json"), "w"), indent=1)
print("SUMMARY" + a.tag, json.dumps(results))
