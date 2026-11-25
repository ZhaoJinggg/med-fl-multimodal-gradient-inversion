"""Diagnostic: how fast does the matching loss (cosine distance of the image-side gradient) grow when
the true image is perturbed? White noise and smooth (blurred) noise at several amplitudes, batch 1,
u = 1, BN in training mode. Also lists the share of the squared gradient norm per layer group and the
cosine distance per group, and the loss at the mean image and at another patient's image.

usage: python src/calibration/cos_sensitivity.py --global runs/pilot_r128/seed0_fedavg/global_round9.pt --res 128
"""
import argparse
import os

import torch
import torch.nn.functional as F

import os as _os, sys as _sys  # noqa: E401,E402
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # src/ (common.py)
from common import ROOT, FusionModel, PairData, bce, is_image_param, load_seed

ap = argparse.ArgumentParser()
ap.add_argument("--global", dest="glob", required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--victims", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--res", type=int, default=128)
ap.add_argument("--batch", type=int, default=1)
ap.add_argument("--img-sd", choices=["model", "flat", "median"], default="model",
                help="diagnostic: replace the image-feature sd by its mean (flat) or floor it at the median")
ap.add_argument("--eval-bn", action="store_true", help="BN in eval mode (running statistics)")
a = ap.parse_args()
dev = "cuda"
data = PairData(a.res)
split = load_seed(a.seed)
model = FusionModel().to(dev)
if a.glob == "random":      # randomly initialized ResNet-18 (diagnostic)
    import torchvision
    r = torchvision.models.resnet18(weights=None); r.fc = torch.nn.Identity()
    model.img = r.to(dev)
elif a.glob != "pretrained":  # "pretrained" = round 0: ImageNet ResNet-18, untrained head
    model.load_state_dict(torch.load(os.path.join(ROOT, a.glob), map_location=dev))
model.train()
if a.eval_bn:
    model.img.eval()
sd = model.img_sd
print("img_sd quantiles 0/5/25/50/75/100%:", [round(torch.quantile(sd, q).item(), 4) for q in (0, .05, .25, .5, .75, 1)])
if a.img_sd == "flat":
    model.img_sd = torch.full_like(sd, sd.mean().item())
elif a.img_sd == "median":
    model.img_sd = sd.clamp(min=sd.median().item())
names = [n for n, _ in model.named_parameters() if is_image_param(n)]
params = [p for n, p in model.named_parameters() if is_image_param(n)]
GROUPS = ["img.conv1", "img.bn1", "img.layer1", "img.layer2", "img.layer3", "img.layer4", "head"]
pub = data.x[torch.cat([data.split("val"), data.split("test")])].mean(0, keepdim=True).to(dev)


def grad(x, h, y):
    return [g.detach() for g in torch.autograd.grad(bce(model(x, None, None, h_txt=h), y), params)]


def cos(g1, g2, sel=None):
    idx = [i for i, n in enumerate(names) if sel is None or n.startswith(sel)]
    num = sum((g1[i] * g2[i]).sum() for i in idx)
    den = torch.sqrt(sum(g1[i].pow(2).sum() for i in idx)) * torch.sqrt(sum(g2[i].pow(2).sum() for i in idx))
    return (1 - num / (den + 1e-20)).item()


def smooth(n, sigma=3.0):
    k = int(6 * sigma) | 1
    ax = torch.arange(k, device=n.device) - k // 2
    w = torch.exp(-ax.float() ** 2 / (2 * sigma ** 2)); w = w / w.sum()
    n = F.conv2d(F.pad(n, (k // 2,) * 4, mode="reflect"), w.view(1, 1, 1, k))
    n = F.conv2d(n, w.view(1, 1, k, 1))
    return n / n.std()


for v in a.victims:
    studies = split["victim_batches"][str(v)][:a.batch]
    x, ids, mask, y = data.batch(data.rows(studies), dev)
    other = data.batch(data.rows(split["victim_batches"][str((v + 1) % 10)][:a.batch]), dev)[0]
    with torch.no_grad():
        h = model.text_features(ids, mask)
    g0 = grad(x, h, y)
    tot = sum(g.pow(2).sum() for g in g0)
    share = {s: sum(g.pow(2).sum() for g, n in zip(g0, names) if n.startswith(s)).item() / tot.item() for s in GROUPS}
    print(f"\nvictim {v}: share of squared gradient norm: " + ", ".join(f"{s.replace('img.', '')} {share[s]:.3f}" for s in GROUPS))
    torch.manual_seed(v)
    white = torch.randn_like(x)
    low = smooth(torch.randn_like(x))
    print("perturbation | sigma | cos all | " + " | ".join(s.replace("img.", "") for s in GROUPS))
    for kind, n in (("white", white), ("smooth", low)):
        for s in (0.002, 0.005, 0.01, 0.02, 0.05):
            g = grad((x + s * n).clamp(0, 1), h, y)
            print(f"{kind:6s} | {s:.3f} | {cos(g, g0):.4f} | " + " | ".join(f"{cos(g, g0, sel):.4f}" for sel in GROUPS))
    for kind, xx in (("mean image", pub.repeat(a.batch, 1, 1, 1)), ("other patient", other), ("0.9x+0.05", 0.9 * x + 0.05)):
        g = grad(xx, h, y)
        print(f"{kind:13s} | {cos(g, g0):.4f} | " + " | ".join(f"{cos(g, g0, sel):.4f}" for sel in GROUPS))
