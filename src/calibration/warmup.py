"""FedAvg warm-up: 10 clients, full participation, 1 local epoch per round at B = 32, plain SGD.
The server averages the full state dict, so BN running statistics are uploaded and averaged.
Saves the global model after each round and reports test and validation accuracy.

usage: python src/calibration/warmup.py --seed 0 --rounds 9 --lr-img 0.01 --lr-txt 1e-4 --out runs/pilot
"""
import argparse
import copy
import json
import os
import time

import torch

import os as _os, sys as _sys  # noqa: E401,E402
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # src/ (common.py)
from common import ROOT, FusionModel, PairData, bce, load_seed, sgd_step

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--rounds", type=int, default=9)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--lr-img", type=float, default=0.01)
ap.add_argument("--lr-txt", type=float, default=1e-4)
ap.add_argument("--res", type=int, default=224)
ap.add_argument("--out", default="runs/pilot")
ap.add_argument("--save", type=int, nargs="*", default=None, help="rounds to save (default: all)")
a = ap.parse_args()
dev = "cuda"
torch.manual_seed(a.seed)
out = os.path.join(ROOT, a.out, f"seed{a.seed}_fedavg")
if a.save == []:
    out = os.path.join(ROOT, a.out, f"seed{a.seed}_fedavg_img{a.lr_img}_txt{a.lr_txt}")
os.makedirs(out, exist_ok=True)

data = PairData(a.res)
split = load_seed(a.seed)
clients = {int(k): data.rows(v) for k, v in split["clients"].items()}
model = FusionModel().to(dev)


@torch.no_grad()
def accuracy(m, rows):
    m.eval()
    correct = 0
    for i in range(0, len(rows), 64):
        x, ids, mask, y = data.batch(rows[i:i + 64], dev)
        correct += ((m(x, ids, mask) > 0).float() == y).sum().item()
    return correct / len(rows)


log = []
g = torch.Generator().manual_seed(a.seed)
for r in range(1, a.rounds + 1):
    t0 = time.time()
    states = []
    for k in sorted(clients):
        local = copy.deepcopy(model).train()
        perm = clients[k][torch.randperm(len(clients[k]), generator=g)]
        for i in range(0, len(perm), a.batch):
            x, ids, mask, y = data.batch(perm[i:i + a.batch], dev)
            sgd_step(local, bce(local(x, ids, mask), y), a.lr_img, a.lr_txt)
        states.append(local.state_dict())
    n = torch.tensor([len(clients[k]) for k in sorted(clients)], dtype=torch.float)
    w = n / n.sum()
    avg = {}
    for key in states[0]:
        if states[0][key].dtype in (torch.float32, torch.float16, torch.float64):
            avg[key] = sum(wi * s[key] for wi, s in zip(w, states))
        else:
            avg[key] = states[0][key]  # num_batches_tracked
    model.load_state_dict(avg)
    rec = {"round": r, "val_acc": accuracy(model, data.split("val")), "test_acc": accuracy(model, data.split("test")),
           "seconds": round(time.time() - t0, 1)}
    log.append(rec)
    print(rec, flush=True)
    if a.save is None or r in a.save:
        torch.save(model.state_dict(), os.path.join(out, f"global_round{r}.pt"))

json.dump({"args": vars(a), "log": log}, open(os.path.join(out, "warmup_log.json"), "w"), indent=1)
