"""Compute the fixed feature standardization constants from the pretrained encoders on the
training pairs of calibration seed 0, and save them to pretrained/feature_norm_seed0.pt."""
import torch
from common import FEATURE_NORM, FusionModel, PairData, load_seed

d = PairData(224)
m = FusionModel().cuda().eval()
rows = torch.cat([d.rows(v) for v in load_seed(0)["clients"].values()])
hi, ht = [], []
with torch.no_grad():
    for i in range(0, len(rows), 64):
        x, ids, mask, _ = d.batch(rows[i:i + 64], "cuda")
        hi.append(m.image_features(x).cpu()); ht.append(m.text_features(ids, mask).cpu())
hi, ht = torch.cat(hi), torch.cat(ht)
def floored_sd(h):
    """Per-dimension spread, floored at the 5th percentile of the modality so that near-constant
    dimensions are not amplified (one GPT-2 dimension has spread 0.002 against a median of 0.25)."""
    sd = h.std(0)
    return sd.clamp(min=torch.quantile(sd, 0.05).item())


torch.save({"img_mu": hi.mean(0), "img_sd": floored_sd(hi), "txt_mu": ht.mean(0), "txt_sd": floored_sd(ht),
            "n": len(rows)}, FEATURE_NORM)
print("saved", FEATURE_NORM, "from", len(rows), "pairs")
