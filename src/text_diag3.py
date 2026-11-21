"""Diagnostic for DAGER's second-layer search under the float32 update (calibration only; uses the truth).

For every text of the attacked batch and every position: the span distance of the true prefix's
second-layer input (must be small), and the distances of the wrong one-token extensions of the true
prefix (tokens of the batch vocabulary; must be large). Printed per text with its |p - y|, for several
choices of the span size k.

usage: python src/text_diag3.py --global runs/pilot_r128/seed0_fedavg/global_round9.pt --victims 0
"""
import argparse
import os

import torch

from common import ROOT, FusionModel, PairData, bce, is_image_param, load_seed

ap = argparse.ArgumentParser()
ap.add_argument("--global", dest="glob", required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--victims", type=int, nargs="+", default=[0])
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--u", type=int, default=1)
ap.add_argument("--lr-img", type=float, default=0.01)
ap.add_argument("--lr-txt", type=float, default=1e-2)
ap.add_argument("--res", type=int, default=128)
a = ap.parse_args()
dev = "cuda"
data = PairData(a.res)
split = load_seed(a.seed)
model = FusionModel().to(dev)
model.load_state_dict(torch.load(os.path.join(ROOT, a.glob), map_location=dev))
model.train()
gpt = model.txt
KEY = "txt.h.1.attn.c_attn.weight"
THR = (5e-3, 1e-2, 2e-2, 5e-2, 0.1)


def dist(R, v):
    v = v.double()
    v = v / v.norm(dim=-1, keepdim=True)
    return (v - (v @ R.T) @ R).norm(dim=-1)


@torch.no_grad()
def layer2_inputs(seqs):
    h = gpt(input_ids=seqs, output_hidden_states=True).hidden_states
    return gpt.h[1].ln_1(h[1])


for v in a.victims:
    studies = split["victim_batches"][str(v)][: a.batch]
    x, ids, mask, y = data.batch(data.rows(studies), dev)
    lens = mask.sum(1).tolist()
    resid = (torch.sigmoid(model(x, ids, mask)) - y).abs().tolist()
    vocab = torch.tensor(sorted({int(ids[b, p]) for b in range(len(ids)) for p in range(lens[b])}), device=dev)
    m = FusionModel().to(dev); m.load_state_dict(model.state_dict()); m.train()
    w0 = dict(m.named_parameters())[KEY].detach().clone()
    acc = torch.zeros_like(w0, dtype=torch.float64)
    for _ in range(a.u):
        names = [n for n, _ in m.named_parameters()]
        grads = torch.autograd.grad(bce(m(x, ids, mask), y), list(m.parameters()))
        with torch.no_grad():
            for n, p, g in zip(names, m.parameters(), grads):
                lr = a.lr_img if is_image_param(n) else a.lr_txt
                p -= lr * g
                if n == KEY:
                    acc -= lr * g.double()
    scale = -1.0 / (a.lr_txt * a.u)
    for name, g in (("exact", scale * acc), ("float32", scale * (dict(m.named_parameters())[KEY].detach() - w0).double())):
        U, S, Vh = torch.linalg.svd(g.T, full_matrices=False)
        floor = S[-200:].median()
        ks = {"5x floor": int((S > 5 * floor).sum()), "2x floor": int((S > 2 * floor).sum()), "tokens": min(sum(lens), 748)}
        print(f"\nvictim {v} [{name}] {sum(lens)} tokens, vocabulary {len(vocab)}; s[0]={S[0]:.2e} floor={floor:.2e}; k: {ks}")
        for kname, k in ks.items():
            R = Vh[:k]
            print(f"  k = {kname} ({k})")
            for b in range(len(ids)):
                L = lens[b]
                d_true = dist(R, layer2_inputs(ids[b:b + 1, :L])[0])            # (L,)
                # wrong one-token extensions of the true prefix, at a sample of positions
                wrong_pass = {t: [] for t in THR}
                d_wrong_min = []
                for p in list(range(1, L, max(1, L // 8)))[:8]:
                    cand = vocab[vocab != ids[b, p]]
                    seqs = torch.cat([ids[b:b + 1, :p].repeat(len(cand), 1), cand.unsqueeze(1)], 1)
                    dw = dist(R, layer2_inputs(seqs)[:, -1])
                    d_wrong_min.append(dw.min().item())
                    for t in THR:
                        wrong_pass[t].append(int((dw < t).sum()))
                line = "  ".join(f"{t:g}: true {int((d_true < t).sum())}/{L}, wrong {sum(wrong_pass[t]) / len(wrong_pass[t]):.1f}" for t in THR)
                print(f"    text {b} |p-y|={resid[b]:.3f} len {L}: true median {d_true.median():.1e} max {d_true.max():.1e}; "
                      f"closest wrong {min(d_wrong_min):.1e} | {line}")
