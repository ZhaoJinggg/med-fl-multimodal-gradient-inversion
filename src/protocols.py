"""Federated protocols: one client's local steps, what the server receives, and the attacker's replay.

Protocols (paper Table I, Secs. II-A, IV-A, IV-C, IV-D, IV-F, fixed-settings table):
  fedavg   u plain SGD steps (Eq. 3); upload: delta of every parameter + BN running statistics.
  fedprox  step direction grad + mu (w - w^t), applied with each group's own learning rate (calibration
           paragraph: "rescales the gradient of step j by (1 - eta mu)^(u-j)", eta mu u per group);
           upload as fedavg.
  scaffold step direction grad + (c - c_k) on every trainable parameter (both encoders and the head;
           buffers have no control variate); after the steps the client's control variate is renewed,
           Option II (paper, fixed settings): c_k+ = c_k - c + (w^t - w^(u)) / (u eta)  (eta per group),
           or Option I (extra cell, design review 2026-11-21): c_k+ = gradient at w^t on the round's
           local data; upload: delta + BN running statistics + dc = c_k+ - c_k.
  fedbn    as fedavg, but BatchNorm2d modules (selected by type, so the shortcut BN `downsample.1` is
           included) stay local: their parameters and statistics are not uploaded.
  bn_stats="local" (any protocol, off by default): the BN running statistics are not uploaded (the
           "choice of whether BN statistics are uploaded" of the implementation paragraph); the BN
           parameters still are, except under fedbn.

The server knows c and c_k (it accumulates the dc of every round; full participation), so it knows the
SCAFFOLD correction -u eta (c - c_k) contained in delta and can remove it (paper Sec. IV-F).

Wiring: client_upload replaces victim_update in attack_image.py / attack_text.py; simulate_image_steps
replaces attack_image.simulate. With proto fedavg both reproduce the existing functions bit for bit
(checked in agent_notes/protocols.md, test T1/T2).
"""
import copy

import torch
import torch.nn as nn
from torch.func import functional_call

from common import bce, is_image_param

PROTOCOLS = ("fedavg", "fedprox", "scaffold", "fedbn")
MOM = 0.1  # BatchNorm2d momentum (torchvision default)


class Protocol:
    """A protocol and its parameters. mu is used by fedprox only; option and the control variates
    (c of the server, c_k of the client; {parameter name: tensor}) by scaffold only.
    bn_stats: "upload" (paper default for fedavg/fedprox/scaffold) or "local"; fedbn is always local."""

    def __init__(self, name="fedavg", mu=1.0, option="II", c=None, c_k=None, bn_stats="upload"):
        assert name in PROTOCOLS, name
        assert option in ("I", "II"), option
        assert bn_stats in ("upload", "local"), bn_stats
        self.name = name
        self.mu = float(mu) if name == "fedprox" else 0.0
        self.option = option
        self.bn_stats_uploaded = bn_stats == "upload" and name != "fedbn"
        self.c = self.c_k = self.corr = None
        if name == "scaffold" and c is not None:
            self.set_controls(c, c_k)

    def set_controls(self, c, c_k):
        """c - c_k is added to the gradient at every local step (constant within the round)."""
        self.c, self.c_k = c, c_k
        self.corr = {n: c[n] - c_k[n] for n in c}

    def __repr__(self):
        return (f"Protocol({self.name}, mu={self.mu}, option={self.option}, "
                f"bn_stats={'upload' if self.bn_stats_uploaded else 'local'})")


def lr_of(name, lr_img, lr_txt):
    return lr_img if is_image_param(name) else lr_txt


# ---------------------------------------------------------------- BN bookkeeping (FedBN)
def bn_modules(model):
    """Names of the BatchNorm2d modules, by type (PFLlib's name test 'bn' would miss downsample.1)."""
    return [n for n, m in model.named_modules() if isinstance(m, nn.BatchNorm2d)]


def bn_state_keys(model):
    """State-dict keys of the BN modules: weight, bias, running_mean, running_var, num_batches_tracked."""
    mods = set(bn_modules(model))
    return [k for k in model.state_dict() if k.rsplit(".", 1)[0] in mods]


def bn_stats(model):
    """{BN module name: (running_mean, running_var)} (copies)."""
    return {n: (m.running_mean.clone(), m.running_var.clone()) for n, m in model.named_modules()
            if isinstance(m, nn.BatchNorm2d)}


def local_state_keys(model, proto):
    """State-dict keys that stay on the client (never uploaded, never averaged): every BN entry under
    fedbn; only the BN buffers (running statistics, num_batches_tracked) with bn_stats local."""
    if proto.name == "fedbn":
        return bn_state_keys(model)
    if not proto.bn_stats_uploaded:
        params = {n for n, _ in model.named_parameters()}
        return [k for k in bn_state_keys(model) if k not in params]
    return []


def uploaded_param_names(model, proto):
    """Parameters whose delta the client uploads: all, except the BN parameters under fedbn."""
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    local = set(local_state_keys(model, proto))
    return [n for n in names if n not in local]


# ---------------------------------------------------------------- one local step
def direction(name, p, g, anchor, proto):
    """Gradient of the client's local objective (the g of Eq. 3) for one parameter. Works on leaf and
    non-leaf tensors (no in-place ops), so the attacker's unrolled simulation can use it too."""
    if proto.mu > 0:
        g = g + proto.mu * (p - anchor[name])
    if proto.corr is not None:
        g = g + proto.corr[name]
    return g


def local_step(model, loss, lr_img, lr_txt, proto, anchor=None, acc=None):
    """One in-place SGD step (no momentum, no weight decay) under proto. anchor: w^t (fedprox).
    acc: optional float64 accumulator of the applied steps (the 'exact' update of attack_text).
    With fedavg this is common.sgd_step operation for operation. Returns the raw loss gradients."""
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, params)
    with torch.no_grad():
        for n, p, g in zip(names, params, grads):
            d = direction(n, p, g, anchor, proto)
            lr = lr_of(n, lr_img, lr_txt)
            p -= lr * d
            if acc is not None:
                acc[n] -= lr * d.double()
    return dict(zip(names, grads))


def mean_gradient(model, batches):
    """SCAFFOLD Option I: gradient of the loss at the current weights over the round's local data, as
    the batch-size-weighted mean of the minibatch gradients (BN in training mode, as in the local
    steps). The BN running statistics are restored afterwards, so the extra pass leaves no trace."""
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    params = [p for p in model.parameters() if p.requires_grad]
    saved = {n: b.clone() for n, b in model.named_buffers()}
    total = sum(len(b[3]) for b in batches)
    out = {n: torch.zeros_like(p) for n, p in zip(names, params)}
    for x, ids, mask, y in batches:
        grads = torch.autograd.grad(bce(model(x, ids, mask), y), params)
        with torch.no_grad():
            for n, g in zip(names, grads):
                out[n] += (len(y) / total) * g
    with torch.no_grad():
        for n, b in model.named_buffers():
            b.copy_(saved[n])
    return out


def scaffold_dc(proto, w0, model, steps, lr_img, lr_txt, grad_w0=None):
    """Change of the client's control variate, dc = c_k+ - c_k (SCAFFOLD Alg. 1, line 12).
    Option II: c_k+ = c_k - c + (w^t - w^(K)) / (K eta); Option I: c_k+ = grad_w0 (mean_gradient at w^t)."""
    dc = {}
    with torch.no_grad():
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if proto.option == "II":
                new = proto.c_k[n] - proto.c[n] + (w0[n] - p) / (steps * lr_of(n, lr_img, lr_txt))
            else:
                new = grad_w0[n]
            dc[n] = new - proto.c_k[n]
    return dc


# ---------------------------------------------------------------- the victim's round (client side)
def client_upload(model, x, ids, mask, y, u, lr_img, lr_txt, proto, bn_local=None, exact=False,
                  bn_eval=False):
    """The victim's round (paper Sec. IV-A): u local steps on its one batch under proto, starting from
    the broadcast model `model` (not modified; deep-copied). The client's own local state `bn_local`
    ({state key: tensor}, from train_fl.load_aux: FedBN all BN entries, bn_stats local the running
    statistics) is loaded on top; without it the client starts from the broadcast entries.
    bn_eval: BN uses running statistics (control only). Under scaffold, proto must hold c and c_k.

    Returns (up, client):
      up      what the server receives
        up["delta"]      {param name: w^(u) - w^t} (float32 difference, or with exact=True the float64
                         sum of the applied steps, cast to float32); fedbn: BN parameters left out
        up["bn_before"]  {BN module: (running_mean, running_var)} of the broadcast model (the server holds
                         it); None when BN statistics are not uploaded (fedbn, bn_stats local)
        up["bn_after"]   the same after the u steps (uploaded with the state); None as above
        up["dc"]         scaffold only: {param name: c_k+ - c_k}
      client  client-side quantities the server does NOT receive
        client["h_txt"]  per-step text features h_txt^(j), j = 1..u (oracle-text condition)
        client["logit0"] logits of the first step
        client["w_end"]  {param name: w^(u)} (checks only)
    """
    if proto.name == "scaffold":
        assert proto.corr is not None, "scaffold: set the control variates (Protocol.set_controls) first"
    m = copy.deepcopy(model)
    m.train()
    if bn_local is not None:
        keys = set(local_state_keys(m, proto))
        assert keys and set(bn_local) == keys, "bn_local does not match the protocol's local keys"
        m.load_state_dict(bn_local, strict=False)
    if bn_eval:
        m.img.eval()
    w0 = {n: p.detach().clone() for n, p in m.named_parameters()}
    grad_w0 = None
    if proto.name == "scaffold" and proto.option == "I":
        grad_w0 = mean_gradient(m, [(x, ids, mask, y)])
    acc = {n: torch.zeros_like(p, dtype=torch.float64) for n, p in m.named_parameters()} if exact else None
    h_steps, logit0 = [], None
    for t in range(u):
        with torch.no_grad():
            h_steps.append(m.text_features(ids, mask).detach())
        logit = m(x, ids, mask)
        if t == 0:
            logit0 = logit.detach()
        local_step(m, bce(logit, y), lr_img, lr_txt, proto, anchor=w0, acc=acc)
    keep = set(uploaded_param_names(m, proto))
    if exact:
        delta = {n: acc[n].float() for n, _ in m.named_parameters() if n in keep}
    else:
        delta = {n: p.detach() - w0[n] for n, p in m.named_parameters() if n in keep}
    up = {"delta": delta, "bn_before": None, "bn_after": None}
    if proto.bn_stats_uploaded:
        up["bn_before"], up["bn_after"] = bn_stats(model), bn_stats(m)
    if proto.name == "scaffold":
        up["dc"] = scaffold_dc(proto, w0, m, u, lr_img, lr_txt, grad_w0)
    client = {"h_txt": h_steps, "logit0": logit0, "w_end": {n: p.detach() for n, p in m.named_parameters()}}
    return up, client


# ---------------------------------------------------------------- server side
def scaffold_correction(c, c_k, u, lr_img, lr_txt, names=None):
    """The part of the SCAFFOLD delta that the server knows: -u eta (c - c_k), per parameter (float64;
    c - c_k in float32 first, exactly as the client forms its correction in Protocol.set_controls)."""
    names = list(c) if names is None else names
    return {n: -u * lr_of(n, lr_img, lr_txt) * (c[n] - c_k[n]).double() for n in names}


def remove_correction(delta, corr):
    """delta - corr where a correction exists (SCAFFOLD; paper Sec. IV-F: before DAGER). Computed in
    float64 and cast back, so the removal adds no rounding beyond that already in the float32 delta."""
    return {n: (v.double() - corr[n].double()).to(v.dtype) if n in corr else v for n, v in delta.items()}


def server_delta(up, proto, u, lr_img, lr_txt):
    """The delta the text attack consumes: the uploaded delta, minus the known correction under
    scaffold, i.e. -eta * (sum of the loss gradients along the client's corrected trajectory). The image
    attack instead keeps up["delta"] and simulates the correction (simulate_image_steps)."""
    if proto.name != "scaffold":
        return up["delta"]
    return remove_correction(up["delta"], scaffold_correction(proto.c, proto.c_k, u, lr_img, lr_txt,
                                                              list(up["delta"])))


def effective_steps(proto, u, lr):
    """sum_j (1 - lr mu)^(u-j): the number of plain gradient steps the delta (after server_delta) is worth
    for a group with learning rate lr. u for fedavg / scaffold / fedbn; below u under fedprox. For the
    scale factors of the text attack (dager's -1/(lr_txt u), infer_labels' bound lr_img u / B)."""
    return sum((1 - lr * proto.mu) ** (u - j) for j in range(1, u + 1))


def option1_update(up, proto, lr_img, lr_txt):
    """SCAFFOLD Option I only: c_k+ = g(w^t; batch), so the server, which knows c_k, obtains the
    single-step gradient at the broadcast weights as dc + c_k, whatever u is. Returned as the u = 1 weight
    difference -eta g (float64), the form the attacks consume with u = 1 (design review, item 3)."""
    assert proto.name == "scaffold" and proto.option == "I"
    return {n: -lr_of(n, lr_img, lr_txt) * (d.double() + proto.c_k[n].double()) for n, d in up["dc"].items()}


# ---------------------------------------------------------------- attacker's replay (image side)
def simulate_image_steps(model, img_names, bn_mods, xd, y, h_steps, lr_img, proto, collect_bn=False):
    """Unrolled local steps on dummy images (differentiable in xd): only the ResNet-18 + head
    parameters in img_names are stepped, step j uses h_steps[j] (u = len(h_steps)); the proximal term
    (fedprox) or the correction c - c_k (scaffold) enters every step. The attacker's model holds the
    broadcast weights; under fedbn its BN entries are the server's (pretrained) ones (paper Sec. IV-F).
    Returns the simulated delta for img_names and, if collect_bn, the BN batch statistics averaged
    with the running-statistics weights of Eq. 7. With fedavg: attack_image.simulate bit for bit.
    The caller restricts the distance to the uploaded names (iterate over the observed delta's keys)."""
    u = len(h_steps)
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
        p_img = {n: v - lr_img * direction(n, v, gi, params, proto) for (n, v), gi in zip(p_img.items(), g)}
    dw_sim = {n: p_img[n] - start[n] for n in img_names}
    bn_avg = {}
    if collect_bn:
        tot = sum(weights)
        for n in stats:
            bn_avg[n] = (sum(w * s[0] for w, s in zip(weights, stats[n])) / tot,
                         sum(w * s[1] for w, s in zip(weights, stats[n])) / tot)
    return dw_sim, bn_avg
