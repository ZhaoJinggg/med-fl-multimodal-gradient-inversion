"""CPU smoke tests for protocols.py and train_fl.py (tiny settings, real FusionModel and data).

T1  client_upload(fedavg) == victim_update of attack_image.py and attack_text.py (copied verbatim below,
    because both scripts parse arguments and use CUDA on import), bit for bit, incl. --exact
T2  simulate_image_steps(fedavg) == attack_image.simulate (copied verbatim), bit for bit, incl. BN stats
    and the gradient w.r.t. the dummy images
T3  simulation at the true images reproduces the upload (fedprox, scaffold, fedbn with the victim's BN;
    fedbn with the server's BN shows the mismatch the paper expects)
T4  fedprox: mu = 0 == fedavg bit for bit; mu = 1: delta == -eta sum_j (1 - eta mu)^(u-j) g_j
T5  scaffold with zero control variates == fedavg bit for bit
T6  scaffold: delta - known correction == -eta sum_j g_j along the corrected trajectory; at u = 1 equals
    the fedavg delta; Option II dc is computable from delta (no new information); Option I dc + c_k ==
    the single-step gradient
T7  fedbn: no BN parameter in the upload, no BN statistics; bn_stats local: BN parameters but no statistics
T8  train_fl tiny runs: scaffold round 1 == fedavg round 1; fedprox mu 0 == fedavg; fedbn keeps the server's
    BN entries pretrained; c == mean of the c_k; Option I and --bn-stats local run; load_aux
T9  train_fl fedavg == warmup.py itself (executed with dev = cpu and a client/pair subset), bit for bit

usage (from mmgia/): python tests/test_protocols.py --out <scratch dir> [--tests T1 T2 ...]
"""
import argparse
import copy
import os
import sys
import time

import torch
from torch.func import functional_call

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))  # noqa: E402
import protocols as P
import train_fl
from common import ROOT, FusionModel, PairData, bce, is_image_param, load_seed

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True, help="scratch directory for train_fl / warmup outputs (absolute)")
ap.add_argument("--tests", nargs="*", default=None)
ap.add_argument("--B", type=int, default=2)
ap.add_argument("--u", type=int, default=3)
ap.add_argument("--res", type=int, default=224, help="224 reads the existing image cache (no new files)")
a = ap.parse_args()
dev = "cpu"
LR_IMG, LR_TXT = 0.01, 1e-2
RUN = lambda t: a.tests is None or t in a.tests
t_start = time.time()

data = PairData(a.res)
split = load_seed(0)
rows = data.rows(split["victim_batches"]["0"][: a.B])
x, ids, mask, y = data.batch(rows, dev)
torch.manual_seed(0)
model = FusionModel().to(dev)
model.train()
img_names = [n for n, _ in model.named_parameters() if is_image_param(n)]
bn_mods = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.BatchNorm2d)]
MOM = 0.1


def same(d1, d2):
    return set(d1) == set(d2) and all(torch.equal(d1[k], d2[k]) for k in d1)


def maxrel(d1, d2, names=None):
    """max over parameters of |d1 - d2|_max / |d2|_max (float64)."""
    names = names or list(d2)
    return max(((d1[n].double() - d2[n].double()).abs().max() / (d2[n].double().abs().max() + 1e-30)).item()
               for n in names)


def cos_dist(d1, d2, names):
    num = sum((d1[n].double() * d2[n].double()).sum() for n in names)
    na = torch.sqrt(sum(d1[n].double().pow(2).sum() for n in names))
    nb = torch.sqrt(sum(d2[n].double().pow(2).sum() for n in names))
    return (1 - num / (na * nb)).item()


def report(name, ok, msg):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {msg}  ({time.time() - t_start:.0f} s)", flush=True)


# ------------------------------------------------ verbatim copies of the attack scripts' functions
def lr_of(name):
    return LR_IMG if is_image_param(name) else LR_TXT


def victim_update_image(x, ids, mask, y, u):          # attack_image.victim_update (a.u -> u, bn_mode train)
    m = FusionModel().to(dev)
    m.load_state_dict(model.state_dict())
    m.train()
    w0 = {n: p.detach().clone() for n, p in m.named_parameters()}
    rs0 = {n: (mod.running_mean.clone(), mod.running_var.clone()) for n, mod in m.named_modules()
           if isinstance(mod, torch.nn.BatchNorm2d)}
    h_steps = []
    for _ in range(u):
        with torch.no_grad():
            h_steps.append(m.text_features(ids, mask).detach())
        loss = bce(m(x, ids, mask), y)
        names = [n for n, _ in m.named_parameters()]
        grads = torch.autograd.grad(loss, list(m.parameters()))
        with torch.no_grad():
            for n, p, g in zip(names, m.parameters(), grads):
                p -= lr_of(n) * g
    dw = {n: (p.detach() - w0[n]) for n, p in m.named_parameters() if is_image_param(n)}
    rs1 = {n: (mod.running_mean.clone(), mod.running_var.clone()) for n, mod in m.named_modules()
           if isinstance(mod, torch.nn.BatchNorm2d)}
    return dw, rs0, rs1, h_steps


def victim_update_text(x, ids, mask, y, u, exact):    # attack_text.victim_update (a.u -> u, a.exact -> exact)
    m = FusionModel().to(dev)
    m.load_state_dict(model.state_dict())
    m.train()
    w0 = {n: p.detach().clone() for n, p in m.named_parameters()}
    acc = {n: torch.zeros_like(p, dtype=torch.float64) for n, p in m.named_parameters()}
    logit0 = None
    for t in range(u):
        logit = m(x, ids, mask)
        if t == 0:
            logit0 = logit.detach()
        loss = bce(logit, y)
        names = [n for n, _ in m.named_parameters()]
        grads = torch.autograd.grad(loss, list(m.parameters()))
        with torch.no_grad():
            for n, p, g in zip(names, m.parameters(), grads):
                p -= lr_of(n) * g
                acc[n] -= lr_of(n) * g.double()
    if exact:
        return {n: v.float() for n, v in acc.items()}, logit0
    return {n: (p.detach() - w0[n]) for n, p in m.named_parameters()}, logit0


def simulate_ref(xd, y, h_steps, collect_bn, u):      # attack_image.simulate (a.u -> u)
    params = {n: p.detach() for n, p in model.named_parameters()}
    buffers = {n: b.detach().clone() for n, b in model.named_buffers()}
    p_img = {n: params[n].clone().requires_grad_(True) for n in img_names}
    start = p_img
    stats = {n: [] for n, _ in bn_mods}
    weights = [MOM * (1 - MOM) ** (u - j) for j in range(1, u + 1)]
    for j in range(u):
        hooks = []
        if collect_bn:
            for n, mod in bn_mods:
                def hk(module, inp, outp, n=n):
                    t = inp[0]
                    stats[n].append((t.mean((0, 2, 3)), t.var((0, 2, 3), unbiased=True)))
                hooks.append(mod.register_forward_hook(hk))
        full = dict(params); full.update(p_img)
        logit = functional_call(model, (full, buffers), (xd, None, None), {"h_txt": h_steps[j]})
        for h in hooks:
            h.remove()
        loss = bce(logit, y)
        g = torch.autograd.grad(loss, list(p_img.values()), create_graph=True)
        p_img = {n: v - LR_IMG * gi for (n, v), gi in zip(p_img.items(), g)}
    dw_sim = {n: p_img[n] - start[n] for n in img_names}
    bn_avg = {}
    if collect_bn:
        tot = sum(weights)
        for n in stats:
            bn_avg[n] = (sum(w * s[0] for w, s in zip(weights, stats[n])) / tot,
                         sum(w * s[1] for w, s in zip(weights, stats[n])) / tot)
    return dw_sim, bn_avg


def manual_trajectory(proto, u, mdl=None):
    """u local steps with protocols.local_step; returns delta and the raw loss gradients of each step."""
    m = copy.deepcopy(mdl or model).train()
    w0 = {n: p.detach().clone() for n, p in m.named_parameters()}
    graw = []
    for _ in range(u):
        graw.append(P.local_step(m, bce(m(x, ids, mask), y), LR_IMG, LR_TXT, proto, anchor=w0))
    return {n: p.detach() - w0[n] for n, p in m.named_parameters()}, graw


def rand_controls(scale_from):
    """Nonzero control variates of the size of a gradient (c and c_k differ)."""
    torch.manual_seed(1)
    c = {n: 0.5 * g + 0.1 * g.abs().mean() * torch.randn_like(g) for n, g in scale_from.items()}
    c_k = {n: -0.3 * g + 0.1 * g.abs().mean() * torch.randn_like(g) for n, g in scale_from.items()}
    return c, c_k


fedavg = P.Protocol("fedavg")
up_avg, cl_avg = P.client_upload(model, x, ids, mask, y, a.u, LR_IMG, LR_TXT, fedavg)
_, graw_avg = manual_trajectory(fedavg, 1)
g0 = graw_avg[0]                     # loss gradient at w^t (used to size the control variates)

# ------------------------------------------------ T1
if RUN("T1"):
    dw, rs0, rs1, h_ref = victim_update_image(x, ids, mask, y, a.u)
    ok1 = same({n: up_avg["delta"][n] for n in img_names}, dw)
    ok2 = all(torch.equal(up_avg["bn_before"][k][i], rs0[k][i]) and torch.equal(up_avg["bn_after"][k][i], rs1[k][i])
              for k in rs0 for i in (0, 1))
    ok3 = all(torch.equal(h1, h2) for h1, h2 in zip(cl_avg["h_txt"], h_ref))
    dt, l0 = victim_update_text(x, ids, mask, y, a.u, exact=False)
    ok4 = same(up_avg["delta"], dt) and torch.equal(cl_avg["logit0"], l0)
    up_ex, _ = P.client_upload(model, x, ids, mask, y, a.u, LR_IMG, LR_TXT, fedavg, exact=True)
    dte, _ = victim_update_text(x, ids, mask, y, a.u, exact=True)
    ok5 = same(up_ex["delta"], dte)
    report("T1 fedavg upload == attack scripts' victim_update", ok1 and ok2 and ok3 and ok4 and ok5,
           f"image delta {ok1}, BN stats {ok2}, h_txt {ok3}, full delta + logit0 {ok4}, exact {ok5}")

# ------------------------------------------------ T2
if RUN("T2"):
    torch.manual_seed(5)
    xd0 = torch.rand_like(x)
    res = []
    for fn in ("ref", "new"):
        xd = xd0.clone().requires_grad_(True)
        if fn == "ref":
            dws, bna = simulate_ref(xd, y, cl_avg["h_txt"], True, a.u)
        else:
            dws, bna = P.simulate_image_steps(model, img_names, bn_mods, xd, y, cl_avg["h_txt"], LR_IMG, fedavg,
                                              collect_bn=True)
        loss = sum((dws[n] * up_avg["delta"][n]).sum() for n in img_names) + sum(v[0].sum() + v[1].sum() for v in bna.values())
        gx, = torch.autograd.grad(loss, xd)
        res.append(({n: v.detach() for n, v in dws.items()}, {k: (v[0].detach(), v[1].detach()) for k, v in bna.items()}, gx))
    ok1 = same(res[0][0], res[1][0])
    ok2 = all(torch.equal(res[0][1][k][i], res[1][1][k][i]) for k in res[0][1] for i in (0, 1))
    ok3 = torch.equal(res[0][2], res[1][2])
    report("T2 simulate_image_steps(fedavg) == attack_image.simulate", ok1 and ok2 and ok3,
           f"delta {ok1}, BN stats {ok2}, d/dx' {ok3}")

# ------------------------------------------------ T3
if RUN("T3"):
    c, c_k = rand_controls(g0)
    cases = [("fedavg", fedavg, None, model), ("fedprox mu=1", P.Protocol("fedprox", mu=1.0), None, model),
             ("scaffold II", P.Protocol("scaffold", c=c, c_k=c_k), None, model)]
    # FedBN: the victim's local BN differs from the server's (pretrained) BN
    torch.manual_seed(2)
    fb = P.Protocol("fedbn")
    sd = model.state_dict()
    bn_local = {k: (sd[k] * (1 + 0.1 * torch.randn_like(sd[k])) if sd[k].is_floating_point() else sd[k].clone())
                for k in P.local_state_keys(model, fb)}
    m_victim_bn = copy.deepcopy(model); m_victim_bn.load_state_dict(bn_local, strict=False)
    cases += [("fedbn, attacker holds victim BN", fb, bn_local, m_victim_bn),
              ("fedbn, attacker holds server BN", fb, bn_local, model)]
    for name, pr, bl, att in cases:
        up, cl = P.client_upload(model, x, ids, mask, y, a.u, LR_IMG, LR_TXT, pr, bn_local=bl)
        names = [n for n in img_names if n in up["delta"]]        # the distance over the uploaded names only
        dws, _ = P.simulate_image_steps(att, img_names, bn_mods, x, y, cl["h_txt"], LR_IMG, pr)
        dws = {n: v.detach() for n, v in dws.items()}
        d = cos_dist(dws, up["delta"], names)
        expect_exact = "server BN" not in name
        ok = d < 1e-5 if expect_exact else d > 1e-4
        report(f"T3 simulation at the true images, {name}", ok,
               f"cosine distance {d:.2e} over {len(names)} uploaded image tensors, max rel err {maxrel(dws, up['delta'], names):.2e}")

# ------------------------------------------------ T4
if RUN("T4"):
    up0, _ = P.client_upload(model, x, ids, mask, y, a.u, LR_IMG, LR_TXT, P.Protocol("fedprox", mu=0.0))
    report("T4a fedprox mu=0 == fedavg", same(up0["delta"], up_avg["delta"]), "bit for bit over all parameters")
    mu = 1.0
    pr = P.Protocol("fedprox", mu=mu)
    up1, _ = P.client_upload(model, x, ids, mask, y, a.u, LR_IMG, LR_TXT, pr)
    up1x, _ = P.client_upload(model, x, ids, mask, y, a.u, LR_IMG, LR_TXT, pr, exact=True)
    d_man, graw = manual_trajectory(pr, a.u)
    closed = {n: -lr_of(n) * sum((1 - lr_of(n) * mu) ** (a.u - j) * graw[j - 1][n].double() for j in range(1, a.u + 1))
              for n in up1["delta"]}
    plain = {n: -lr_of(n) * sum(graw[j][n].double() for j in range(a.u)) for n in up1["delta"]}
    gpt = ["txt.h.0.attn.c_attn.weight", "txt.h.1.attn.c_attn.weight"]
    e_closed, e_plain, e_f32 = maxrel(up1x["delta"], closed), maxrel(plain, closed), maxrel(up1["delta"], closed, gpt)
    report("T4b fedprox mu=1: delta == -eta sum (1-eta mu)^(u-j) g_j",
           same(up1["delta"], d_man) and e_closed < 1e-4 and e_plain > 10 * e_closed,
           f"exact upload vs closed form, all tensors: max rel err {e_closed:.2e}; unweighted sum differs by "
           f"{e_plain:.2e} (about eta mu (u-1) = {LR_IMG * mu * (a.u - 1):.2e}); float32 upload on the DAGER layers {e_f32:.2e}")

# ------------------------------------------------ T5
if RUN("T5"):
    zeros = {n: torch.zeros_like(v) for n, v in g0.items()}
    upz, _ = P.client_upload(model, x, ids, mask, y, a.u, LR_IMG, LR_TXT, P.Protocol("scaffold", c=zeros, c_k=zeros))
    report("T5 scaffold with c = c_k = 0 == fedavg", same(upz["delta"], up_avg["delta"]), "bit for bit over all parameters")

# ------------------------------------------------ T6
if RUN("T6"):
    c, c_k = rand_controls(g0)
    pr = P.Protocol("scaffold", c=c, c_k=c_k)
    up, _ = P.client_upload(model, x, ids, mask, y, a.u, LR_IMG, LR_TXT, pr)
    up_ex, _ = P.client_upload(model, x, ids, mask, y, a.u, LR_IMG, LR_TXT, pr, exact=True)
    _, graw = manual_trajectory(pr, a.u)
    sumg = {n: -lr_of(n) * sum(graw[j][n].double() for j in range(a.u)) for n in up["delta"]}
    sd_ = P.server_delta(up, pr, a.u, LR_IMG, LR_TXT)
    sd_ex = P.server_delta(up_ex, pr, a.u, LR_IMG, LR_TXT)
    corr = P.scaffold_correction(c, c_k, a.u, LR_IMG, LR_TXT)
    ratio = max((corr[n].norm() / sumg[n].norm()).item() for n in ("txt.h.0.attn.c_attn.weight", "img.conv1.weight"))
    gpt = ["txt.h.0.attn.c_attn.weight", "txt.h.1.attn.c_attn.weight"]
    report("T6a scaffold: delta - correction == -eta sum_j g_j (corrected trajectory)",
           maxrel(sd_ex, sumg) < 1e-5 and maxrel(sd_, sumg, gpt) < 5e-2,
           f"exact upload: max rel err {maxrel(sd_ex, sumg):.2e}; float32 upload: {maxrel(sd_, sumg):.2e} overall, "
           f"{maxrel(sd_, sumg, gpt):.2e} on the two DAGER layers (|correction| / |gradient part| up to {ratio:.1f})")
    up1, _ = P.client_upload(model, x, ids, mask, y, 1, LR_IMG, LR_TXT, pr)
    up1_avg, _ = P.client_upload(model, x, ids, mask, y, 1, LR_IMG, LR_TXT, fedavg)
    e1 = maxrel(P.server_delta(up1, pr, 1, LR_IMG, LR_TXT), up1_avg["delta"], gpt)
    report("T6b scaffold u=1: delta - correction == fedavg delta", e1 < 5e-2,
           f"max rel diff on the DAGER layers {e1:.2e} (float32 rounding of two weight differences)")
    # Option II: dc = -c - delta / (u eta), so the server can compute it from what it already holds
    dc_pred = {n: -c[n].double() - up["delta"][n].double() / (a.u * lr_of(n)) for n in up["dc"]}
    e2 = maxrel(up["dc"], dc_pred)
    report("T6c Option II: dc == -c - delta/(u eta) (no information beyond delta)", e2 < 1e-3, f"max rel err {e2:.2e}")
    # Option I: dc + c_k == loss gradient at w^t on the batch; compare with the u = 1 fedavg delta
    p1 = P.Protocol("scaffold", option="I", c=c, c_k=c_k)
    upI, _ = P.client_upload(model, x, ids, mask, y, a.u, LR_IMG, LR_TXT, p1)
    g_view = P.option1_update(upI, p1, LR_IMG, LR_TXT)
    exact1 = {n: -lr_of(n) * g0[n].double() for n in g0}
    eI, eF = maxrel(g_view, exact1, gpt), maxrel(up1_avg["delta"], exact1, gpt)
    same_bn = all(torch.equal(upI["bn_after"][k][i], up["bn_after"][k][i]) for k in up["bn_after"] for i in (0, 1))
    report("T6d Option I: dc + c_k == single-step gradient at w^t (any u)", eI < 1e-4 and same_bn,
           f"rel err of the Option I view on the DAGER layers {eI:.2e} vs {eF:.2e} for the float32 fedavg u=1 "
           f"delta; extra gradient pass leaves the BN statistics unchanged: {same_bn}")

# ------------------------------------------------ T7
if RUN("T7"):
    fb = P.Protocol("fedbn")
    upb, _ = P.client_upload(model, x, ids, mask, y, a.u, LR_IMG, LR_TXT, fb)
    bn_params = {f"{n}.{s}" for n, _ in bn_mods for s in ("weight", "bias")}
    missing = set(up_avg["delta"]) - set(upb["delta"])
    ok = missing == bn_params and upb["bn_before"] is None and upb["bn_after"] is None \
        and any(".downsample.1." in k for k in missing) \
        and all(torch.equal(upb["delta"][n], up_avg["delta"][n]) for n in upb["delta"])
    report("T7a fedbn upload", ok, f"{len(missing)} BN tensors left out ({len(bn_mods)} BN layers incl. "
           f"downsample.1), no BN statistics, the rest equals fedavg bit for bit")
    pl = P.Protocol("fedavg", bn_stats="local")
    upl, _ = P.client_upload(model, x, ids, mask, y, a.u, LR_IMG, LR_TXT, pl)
    report("T7b fedavg --bn-stats local", same(upl["delta"], up_avg["delta"]) and upl["bn_after"] is None,
           "all parameters uploaded (== fedavg), no BN statistics")

# ------------------------------------------------ T8 / T9
def tfl(proto, extra=(), rounds=2):
    args = train_fl.get_args(["--proto", proto, "--seed", "0", "--rounds", str(rounds), "--batch", "2",
                              "--lr-img", str(LR_IMG), "--lr-txt", str(LR_TXT), "--device", "cpu", "--out", a.out,
                              "--clients", "0", "1", "--pairs-per-client", "4", "--eval-pairs", "4", *extra])
    _, out = train_fl.run(args, data=data)
    return out


def load(out, r):
    return torch.load(os.path.join(out, f"global_round{r}.pt"))


if RUN("T8"):
    o_avg = tfl("fedavg")
    o_sc = tfl("scaffold", ["--aux-clients", "0", "1"])
    o_px0 = tfl("fedprox", ["--mu", "0"])
    o_px1 = tfl("fedprox", ["--mu", "1"])
    o_bn = tfl("fedbn")
    o_sc1 = tfl("scaffold", ["--scaffold-option", "I", "--aux-clients", "0"])
    o_loc = tfl("fedavg", ["--bn-stats", "local"])
    A1, A2, S1, S2 = load(o_avg, 1), load(o_avg, 2), load(o_sc, 1), load(o_sc, 2)
    report("T8a train_fl scaffold round 1 == fedavg round 1", same(A1, S1), "bit for bit (zero control variates)")
    diff2 = max((S2[k].float() - A2[k].float()).abs().max().item() for k in A2)
    report("T8b train_fl scaffold round 2 != fedavg round 2", diff2 > 0, f"max abs diff {diff2:.2e}")
    report("T8c train_fl fedprox mu=0 == fedavg", same(load(o_px0, 2), A2), "bit for bit after 2 rounds")
    d_px = max((load(o_px1, 2)[k].float() - A2[k].float()).abs().max().item() for k in A2)
    report("T8d train_fl fedprox mu=1 != fedavg", d_px > 0, f"max abs diff {d_px:.2e}")
    pre = FusionModel().state_dict()
    B2 = load(o_bn, 2)
    bn_keys = P.bn_state_keys(model)
    ok_srv = all(torch.equal(B2[k], pre[k]) for k in bn_keys)
    moved = max((B2[k].float() - pre[k].float()).abs().max().item() for k in B2 if k not in bn_keys)
    aux = train_fl.load_aux(o_bn, 2, 0)
    aux1 = train_fl.load_aux(o_bn, 2, 1)
    cl_moved = max((aux["local"][k].float() - pre[k].float()).abs().max().item() for k in bn_keys)
    cl_diff = max((aux["local"][k].float() - aux1["local"][k].float()).abs().max().item() for k in bn_keys)
    report("T8e train_fl fedbn", ok_srv and moved > 0 and cl_moved > 0 and cl_diff > 0 and aux["c"] is None,
           f"server BN entries == pretrained: {ok_srv}; non-BN moved {moved:.2e}; client 0 BN moved {cl_moved:.2e}; "
           f"clients 0/1 BN differ by {cl_diff:.2e}")
    ax0, ax1 = train_fl.load_aux(o_sc, 2, 0), train_fl.load_aux(o_sc, 2, 1)
    e_c = max(((ax0["c"][n] - (ax0["c_k"][n] + ax1["c_k"][n]) / 2).abs().max() / ax0["c"][n].abs().max().clamp_min(1e-30)).item()
              for n in ax0["c"])
    nz = min(ax0["c_k"][n].abs().max().item() for n in ax0["c_k"])
    report("T8f train_fl scaffold control variates", e_c < 1e-5 and nz > 0,
           f"c == mean_k c_k, max rel err {e_c:.2e}; every c_k tensor nonzero: {nz > 0}")
    ax = train_fl.load_aux(o_sc1, 2, 0)
    report("T8g train_fl scaffold Option I runs", ax["c"] is not None and os.path.exists(os.path.join(o_sc1, "global_round2.pt")),
           "control variates saved")
    L2 = load(o_loc, 2)
    run_keys = [k for k in bn_keys if "running" in k]
    ok_l = all(torch.equal(L2[k], pre[k]) for k in run_keys) and not same({k: L2[k] for k in bn_keys if "weight" in k},
                                                                           {k: pre[k] for k in bn_keys if "weight" in k})
    report("T8h train_fl fedavg --bn-stats local", ok_l, "server running statistics stay pretrained, BN weights averaged")

if RUN("T9"):
    wout = os.path.join(a.out, "warmup_ref")
    wpath = os.path.join(ROOT, "src", "calibration", "warmup.py")
    src = open(wpath, encoding="utf-8").read()
    repl = [('dev = "cuda"', 'dev = "cpu"'),
            ("data = PairData(a.res)", "data = _DATA"),
            ('clients = {int(k): data.rows(v) for k, v in split["clients"].items()}',
             'clients = {int(k): data.rows(v[:4]) for k, v in split["clients"].items() if int(k) in (0, 1)}'),
            ('accuracy(model, data.split("val"))', 'accuracy(model, data.split("val")[:4])'),
            ('accuracy(model, data.split("test"))', 'accuracy(model, data.split("test")[:4])')]
    for old, new in repl:
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    argv = sys.argv
    sys.argv = ["warmup.py", "--seed", "0", "--rounds", "2", "--batch", "2", "--lr-img", str(LR_IMG),
                "--lr-txt", str(LR_TXT), "--out", wout]
    exec(compile(src, "warmup.py", "exec"), {"__name__": "__main__", "_DATA": data, "__file__": wpath})
    sys.argv = argv
    o_avg = tfl("fedavg")
    ok = all(same(torch.load(os.path.join(wout, "seed0_fedavg", f"global_round{r}.pt")), load(o_avg, r)) for r in (1, 2))
    report("T9 train_fl fedavg == warmup.py", ok, "global_round1.pt and global_round2.pt bit for bit (2 clients x 4 pairs, B = 2)")

print(f"done in {time.time() - t_start:.0f} s")
