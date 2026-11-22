"""Federated warm-up under FedAvg, FedProx, SCAFFOLD or FedBN (paper Sec. IV-C): 10 clients, full
participation, E local epochs per round (1 in the paper) at B = 32, plain SGD with two learning rates.
The attacked round (u local steps on the victim's batch) is not run here: the attacks replay it from
the saved broadcast state with protocols.client_upload, because the warm-up does not depend on u.

What is saved after round r (the state broadcast in round r + 1):
  global_round{r}.pt              global state dict (as warmup.py). Under FedBN its BN entries are the
                                  server's: the pretrained values, never averaged (paper Sec. IV-F uses
                                  them in the attacker's simulation).
  local_round{r}.pt               FedBN or --bn-stats local: {"keys", "clients": {k: {key: tensor}}},
                                  each client's local state (client side; the server does not hold it)
  scaffold_c_round{r}.pt          SCAFFOLD: server control variate c {param name: tensor}
  scaffold_client{k}_round{r}.pt  SCAFFOLD: client k's control variate c_k (equal, bit for bit, to what
                                  the server accumulates from the uploaded dc)
  train_log.json                  arguments and per-round accuracies

FedAvg through this script reproduces warmup.py exactly (same seed, same arguments): same client
order, same permutations, same averaging arithmetic (full state dict, num_batches_tracked of client 0).
Checked on the CPU against warmup.py itself (agent_notes/protocols.md, test T9).

usage: python src/train_fl.py --proto scaffold --seed 0 --rounds 9 --lr-img 0.01 --lr-txt 1e-2 --res 128 --out runs/fl
"""
import argparse
import copy
import json
import os
import time

import torch

import protocols as P
from common import ROOT, FusionModel, PairData, bce, load_seed


def get_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--proto", choices=P.PROTOCOLS, default="fedavg")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=9)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=1, help="local epochs per warm-up round (paper: 1)")
    ap.add_argument("--lr-img", type=float, default=0.01)
    ap.add_argument("--lr-txt", type=float, default=1e-2)
    ap.add_argument("--mu", type=float, default=1.0, help="FedProx proximal weight (paper: 1)")
    ap.add_argument("--scaffold-option", choices=["I", "II"], default="II", help="paper: Option II")
    ap.add_argument("--bn-stats", choices=["upload", "local"], default="upload",
                    help="local: BN running statistics are not uploaded or averaged (paper default: upload; "
                         "FedBN keeps all BN entries local regardless)")
    ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--out", default="runs/fl")
    ap.add_argument("--save", type=int, nargs="*", default=None, help="rounds to save (default: all)")
    ap.add_argument("--save-aux", type=int, nargs="*", default=None,
                    help="rounds whose SCAFFOLD control variates are saved (default: the last; ~0.5 GB per variate)")
    ap.add_argument("--aux-clients", type=int, nargs="*", default=None,
                    help="clients whose c_k is saved (default: the victim clients of the split)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--idle", type=float, default=0.0,
                    help="seconds of pause per second of training, after every local step (2 = the GPU works a third of the time)")
    # smoke tests only
    ap.add_argument("--clients", type=int, nargs="*", default=None, help="use only these clients")
    ap.add_argument("--pairs-per-client", type=int, default=None, help="use only the first n pairs of each client")
    ap.add_argument("--eval-pairs", type=int, default=None, help="evaluate on the first n pairs of val and test")
    return ap.parse_args(argv)


def out_dir(a):
    tag = {"fedavg": "fedavg", "fedprox": f"fedprox_mu{a.mu:g}", "scaffold": f"scaffold_opt{a.scaffold_option}",
           "fedbn": "fedbn"}[a.proto]
    if a.bn_stats == "local" and a.proto != "fedbn":
        tag += "_bnlocal"
    return os.path.join(ROOT, a.out, f"seed{a.seed}_{tag}")


@torch.no_grad()
def accuracy(data, m, rows, dev, idle=0.0):
    m.eval()
    correct = 0
    for i in range(0, len(rows), 64):
        tb = time.time()
        x, ids, mask, y = data.batch(rows[i:i + 64], dev)
        correct += ((m(x, ids, mask) > 0).float() == y).sum().item()
        if idle > 0 and str(dev) != "cpu":   # same pause as in training (FedBN evaluates ten client models)
            time.sleep(min((time.time() - tb) * idle, 5.0))
    return correct / len(rows)


def run(a, data=None):
    dev = a.device
    torch.manual_seed(a.seed)          # same RNG order as warmup.py up to FusionModel() (head init)
    out = out_dir(a)
    os.makedirs(out, exist_ok=True)
    data = data if data is not None else PairData(a.res)
    split = load_seed(a.seed)
    keys = sorted(int(k) for k in split["clients"])
    if a.clients is not None:
        keys = [k for k in keys if k in a.clients]
    clients = {k: data.rows(split["clients"][str(k)][: a.pairs_per_client]) for k in keys}
    victims = sorted(int(k) for k in split["victim_batches"])
    aux_clients = [k for k in (a.aux_clients if a.aux_clients is not None else victims) if k in clients]
    save_aux = a.save_aux if a.save_aux is not None else [a.rounds]
    eval_rows = {s: data.split(s)[: a.eval_pairs] for s in ("val", "test")}

    model = FusionModel().to(dev)
    proto = P.Protocol(a.proto, mu=a.mu, option=a.scaffold_option, bn_stats=a.bn_stats)
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    local_keys = P.local_state_keys(model, proto)   # FedBN: all BN entries; bn_stats local: BN buffers
    if a.proto == "scaffold":    # control variates start at zero (SCAFFOLD paper); c on the device, c_k on the CPU
        c = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
        c_cl = {k: {n: torch.zeros_like(v, device="cpu") for n, v in c.items()} for k in clients}
    if local_keys:               # every client starts from the pretrained BN state and keeps its own
        loc_cl = {k: {key: model.state_dict()[key].detach().clone().cpu() for key in local_keys} for k in clients}

    n = torch.tensor([len(clients[k]) for k in sorted(clients)], dtype=torch.float)
    w = n / n.sum()
    log = []
    g = torch.Generator().manual_seed(a.seed)
    for r in range(1, a.rounds + 1):
        t0 = time.time()
        avg, dc_sum, first = {}, {}, True
        anchor = {nm: p.detach().clone() for nm, p in model.named_parameters()} if a.proto == "fedprox" else None
        for wi, k in zip(w, sorted(clients)):
            local = copy.deepcopy(model).train()
            if local_keys:
                local.load_state_dict(loc_cl[k], strict=False)
            batches = []
            for _ in range(a.epochs):
                perm = clients[k][torch.randperm(len(clients[k]), generator=g)]
                batches += [perm[i:i + a.batch] for i in range(0, len(perm), a.batch)]
            if a.proto == "scaffold":
                c_k = {nm: v.to(dev) for nm, v in c_cl[k].items()}
                proto.set_controls(c, c_k)
                w0 = {nm: p.detach().clone() for nm, p in local.named_parameters()}
                grad_w0 = None
                if a.scaffold_option == "I":   # gradient at w^t over the round's local data (first epoch)
                    grad_w0 = P.mean_gradient(local, [data.batch(b, dev) for b in batches[: len(batches) // a.epochs]])
            for rows in batches:
                tb = time.time()
                x, ids, mask, y = data.batch(rows, dev)
                P.local_step(local, bce(local(x, ids, mask), y), a.lr_img, a.lr_txt, proto, anchor=anchor)
                if a.idle > 0 and str(dev) != "cpu":   # pause after each step; changes no result. Short bursts keep
                    torch.cuda.synchronize()           # the die temperature down (the card's cooler contact is poor)
                    time.sleep(min((time.time() - tb) * a.idle, 5.0))
            if a.proto == "scaffold":
                dc = P.scaffold_dc(proto, w0, local, len(batches), a.lr_img, a.lr_txt, grad_w0)
                # the client applies dc exactly as the server accumulates it, so both hold the same c_k
                c_cl[k] = {nm: (c_k[nm] + dc[nm]).cpu() for nm in names}
                for nm in names:
                    dc_sum[nm] = dc_sum.get(nm, 0) + dc[nm]
                del c_k, dc, w0, grad_w0
            st = local.state_dict()
            if local_keys:
                loc_cl[k] = {key: st[key].detach().clone().cpu() for key in local_keys}
            # weighted average in the order and form of warmup.py: 0 + w_0 s_0 + w_1 s_1 + ...
            for key in st:
                if key in local_keys:
                    continue                    # stays local: not uploaded, not averaged
                if st[key].dtype in (torch.float32, torch.float16, torch.float64):
                    avg[key] = avg.get(key, 0) + wi * st[key]
                elif first:
                    avg[key] = st[key].clone()  # num_batches_tracked of the first client, as warmup.py
            first = False
            del local, st
        res = model.load_state_dict(avg, strict=not local_keys)
        if local_keys:              # the server's entries for the local keys stay as they are (pretrained);
            # BatchNorm fills a missing num_batches_tracked with its own value and does not report it
            expect = sorted(k for k in local_keys if not k.endswith("num_batches_tracked"))
            assert sorted(res.missing_keys) == expect and not res.unexpected_keys, res
        if a.proto == "scaffold":   # c <- c + (1/N) sum_k dc_k  (server learning rate 1, full participation)
            for nm in names:
                c[nm] = c[nm] + dc_sum[nm] / len(clients)
        rec = {"round": r, "seconds": None}
        if local_keys:
            # no single global model: mean accuracy of the clients' models (each with its local entries),
            # and the broadcast model with the server's entries (what the attacker simulates)
            ev = copy.deepcopy(model)
            accs = {s: [] for s in eval_rows}
            for k in sorted(clients):
                ev.load_state_dict(loc_cl[k], strict=False)
                for s in eval_rows:
                    accs[s].append(accuracy(data, ev, eval_rows[s], dev, a.idle))
            del ev
            rec.update({"val_acc": sum(accs["val"]) / len(clients), "test_acc": sum(accs["test"]) / len(clients),
                        "test_acc_clients": accs["test"],
                        "test_acc_server_state": accuracy(data, model, eval_rows["test"], dev, a.idle)})
        else:
            rec.update({"val_acc": accuracy(data, model, eval_rows["val"], dev, a.idle),
                        "test_acc": accuracy(data, model, eval_rows["test"], dev, a.idle)})
        rec["seconds"] = round(time.time() - t0, 1)
        log.append(rec)
        print(rec, flush=True)
        if a.save is None or r in a.save:
            torch.save(model.state_dict(), os.path.join(out, f"global_round{r}.pt"))
            if local_keys:
                torch.save({"keys": local_keys, "clients": loc_cl}, os.path.join(out, f"local_round{r}.pt"))
        if a.proto == "scaffold" and r in save_aux:
            torch.save({nm: v.detach().cpu() for nm, v in c.items()}, os.path.join(out, f"scaffold_c_round{r}.pt"))
            for k in aux_clients:
                torch.save(c_cl[k], os.path.join(out, f"scaffold_client{k}_round{r}.pt"))

    meta = {"args": vars(a), "protocol": repr(proto), "local_keys": local_keys, "log": log}
    json.dump(meta, open(os.path.join(out, "train_log.json"), "w"), indent=1)
    return model, out


def load_aux(run_dir, r, client, device="cpu"):
    """Auxiliary state for the attack in round r + 1 on `client` (None where a protocol has none):
      "c", "c_k"   SCAFFOLD control variates of the server and of the client (the server holds both)
      "local"      the client's local state (FedBN BN entries, or running statistics with --bn-stats
                   local); client side only: the victim starts from it, the attacker never sees it."""
    f = lambda name: os.path.join(run_dir, name)
    out = {"c": None, "c_k": None, "local": None}
    if os.path.exists(f(f"scaffold_c_round{r}.pt")):
        out["c"] = torch.load(f(f"scaffold_c_round{r}.pt"), map_location=device)
        out["c_k"] = torch.load(f(f"scaffold_client{client}_round{r}.pt"), map_location=device)
    if os.path.exists(f(f"local_round{r}.pt")):
        out["local"] = {k: v.to(device) for k, v in torch.load(f(f"local_round{r}.pt"))["clients"][client].items()}
    return out


if __name__ == "__main__":
    run(get_args())
