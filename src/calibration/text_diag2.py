"""Diagnostic: which wrong tokens pass the first-layer span check with the exact gradient, and at which positions?"""
import os, sys
import torch
import os as _os, sys as _sys  # noqa: E401,E402
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # src/ (common.py)
from common import ROOT, FusionModel, PairData, bce, load_seed, tokenizer
dev = "cuda"; data = PairData(128); split = load_seed(0); tok = tokenizer()
model = FusionModel().to(dev)
model.load_state_dict(torch.load(os.path.join(ROOT, sys.argv[1]), map_location=dev)); model.train()
gpt = model.txt
x, ids, mask, y = data.batch(data.rows(split["victim_batches"]["0"][:8]), dev)
lens = mask.sum(1).tolist()
pairs = {(p, int(ids[b, p])) for b in range(8) for p in range(int(lens[b]))}
g = torch.autograd.grad(bce(model(x, ids, mask), y), dict(model.named_parameters())["txt.h.0.attn.c_attn.weight"])[0]
U, S, Vh = torch.linalg.svd(g.T.double(), full_matrices=False)
R = Vh[:len(pairs)]
print("text lengths:", lens, "| p - y per text:", [round(v, 4) for v in (torch.sigmoid(model(x, ids, mask)) - y).tolist()])
tot = 0
for p in range(64):
    with torch.no_grad():
        e = gpt.h[0].ln_1(gpt.wte.weight + gpt.wpe.weight[p]).double()
    e = e / e.norm(dim=-1, keepdim=True)
    d = (e - (e @ R.T) @ R).norm(dim=-1)
    true = [t for pp, t in pairs if pp == p]
    false = [t for t in torch.where(d < 5e-3)[0].tolist() if t not in true]
    tot += len(false)
    if p < 6 or p % 10 == 0:
        elsewhere = sum(1 for t in false if any(t == tt for _, tt in pairs))
        print(f"pos {p}: true {len(true)}, false {len(false)} (of these, {elsewhere} occur in the batch at another position); "
              f"examples {[tok.decode([t]) for t in false[:8]]}; median distance of all tokens {d.median():.3f}")
print("total false", tot)
