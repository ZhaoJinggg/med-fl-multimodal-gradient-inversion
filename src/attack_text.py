"""Text attack: DAGER on the GPT-2 part of the victim's update.

Reuses the authors' code (third_party/dager): the span decomposition (get_layer_decomp), the token
filter of their attack.py (filter_l1, copied below because attack.py parses arguments on import) and
the decoder sequence search (filter_decoder). A small wrapper supplies what their ModelWrapper
provides, computed from our GPT-2 at the broadcast weights: the first-layer token embeddings and the
inputs to the second layer. (Their helper for the second-layer inputs targets an older version of
Transformers.)

Then the recovered texts are labeled from the update of the head's text weights (infer_labels): the
registered sign rule (sign of the least-squares coefficient), the combined rule (text-only prediction of
the broadcast model plus the box-constrained coefficient rescaled to a residual y - p) and, as the
baseline the update must beat, the model-only rule (text-only prediction alone).

usage: python src/attack_text.py --global runs/pilot/seed0_fedavg/global_round9.pt --seed 0
          --victims 0 1 2 --batch 8 --u 1 --out runs/pilot/text
"""
import argparse
import json
import os
import sys
import time
import types

import numpy as np
import torch
from rouge_score import rouge_scorer
from scipy.optimize import linear_sum_assignment, lsq_linear

from common import ROOT, FusionModel, PairData, bce, is_image_param, load_seed, tokenizer

sys.path.insert(0, os.path.join(ROOT, "third_party", "dager"))
from constants import config as dager_config  # noqa: E402
from utils.functional import get_layer_decomp, get_top_B_in_span  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--global", dest="glob", required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--victims", type=int, nargs="+", default=[0])
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--u", type=int, default=1)
ap.add_argument("--lr-img", type=float, default=0.01)
ap.add_argument("--lr-txt", type=float, default=1e-4)
# DAGER settings: the values of the authors' FedAvg script
ap.add_argument("--noise-factor", type=float, default=5.0, help="rank = singular values above this times the noise floor")
ap.add_argument("--l1-thresh", type=float, default=5e-3)
ap.add_argument("--l2-thresh", type=float, default=5e-3)
ap.add_argument("--rank-cutoff", type=int, default=20)
ap.add_argument("--exact", action="store_true",
                help="reference: the update without float32 rounding (sum of the step gradients in float64), "
                     "as with a double-precision client in the DAGER authors' FedAvg script")
ap.add_argument("--max-beams", type=int, default=500, help="keep at most this many partial sequences per position (best span score)")
ap.add_argument("--gpu-mem", type=float, default=0.5, help="hard limit on this process's share of GPU memory")
ap.add_argument("--parallel", type=int, default=100,
                help="sequences per forward pass of the search (DAGER uses 1000; large values draw full GPU power)")
ap.add_argument("--idle", type=float, default=0.0,
                help="seconds of pause per second of search work (2 = the GPU works a third of the time); "
                     "lowers the average power on a card that cannot hold full load")
ap.add_argument("--res", type=int, default=224)
ap.add_argument("--tag", default="")
ap.add_argument("--out", default="runs/pilot/text")
a = ap.parse_args()
dev = "cuda"
torch.cuda.set_per_process_memory_fraction(a.gpu_mem)  # fail early instead of filling the card

# DAGER's sequence search, with one addition: after each position only the max_beams partial sequences
# with the best span score are kept. Without it, a loose threshold lets the number of sequences grow
# without bound. The authors' file is loaded unchanged and the three lines are inserted here.
_src = open(os.path.join(ROOT, "third_party", "dager", "utils", "filtering_decoder.py")).read()
_anchor = "        scores = torch.cat(next_scores)\n"
assert _src.count(_anchor) == 1
_src = _src.replace(_anchor, _anchor + (
    "        if len(batch) > MAX_BEAMS:\n"
    "            _keep = torch.argsort(scores)[:MAX_BEAMS]\n"
    "            batch, scores, is_batch_incorrect = batch[_keep], scores[_keep], is_batch_incorrect[_keep]\n"))
_mod = types.ModuleType("filtering_decoder_capped")
_mod.MAX_BEAMS = a.max_beams
exec(compile(_src, "filtering_decoder_capped", "exec"), _mod.__dict__)
filter_decoder = _mod.filter_decoder
out = os.path.join(ROOT, a.out)
os.makedirs(out, exist_ok=True)

data = PairData(a.res)
split = load_seed(a.seed)
tok = tokenizer()
model = FusionModel().to(dev)
model.load_state_dict(torch.load(os.path.join(ROOT, a.glob), map_location=dev))
model.train()
gpt = model.txt


def lr_of(name):
    return a.lr_img if is_image_param(name) else a.lr_txt


def victim_update(x, ids, mask, y):
    """u local SGD steps of the full model; returns the full parameter delta and the logits of the first
    step (client side; used only for the calibration record, never by a label rule)."""
    m = FusionModel().to(dev)
    m.load_state_dict(model.state_dict())
    m.train()
    w0 = {n: p.detach().clone() for n, p in m.named_parameters()}
    acc = {n: torch.zeros_like(p, dtype=torch.float64) for n, p in m.named_parameters()}
    logit0 = None
    for t in range(a.u):
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
    if a.exact:
        return {n: v.float() for n, v in acc.items()}, logit0
    return {n: (p.detach() - w0[n]) for n, p in m.named_parameters()}, logit0


# ---------------------------------------------------------------- wrapper for DAGER's functions
class Wrapper:
    """The parts of DAGER's ModelWrapper that filter_l1 / filter_decoder use, for our GPT-2."""

    def __init__(self, args):
        self.args = args
        self.tokenizer = tok
        self.eos_token = tok.eos_token_id
        self.pad_token = tok.eos_token_id
        self.start_token = None
        self.emb_size = gpt.config.n_embd
        dager_config["START_TOKEN"], dager_config["EOS_TOKEN"], dager_config["PAD_TOKEN"] = None, self.eos_token, self.pad_token

    @torch.no_grad()
    def get_embeddings(self, pos=None):
        emb = gpt.wte.weight.unsqueeze(0) + gpt.wpe.weight[pos:pos + 1, None, :]
        return gpt.h[0].ln_1(emb)

    @torch.no_grad()
    def get_layer_inputs(self, sentences, token_type_ids=None, attention_mask=None, layers=1):
        """Input of the second block. Only the first block is run (the search needs nothing deeper, and a
        full forward pass of all 12 blocks costs about 12 times the time and memory). Right padding and
        causal attention make the attention mask irrelevant for the positions before the padding."""
        assert layers == 1
        if a.idle > 0:  # pause in proportion to the work done since the previous call
            torch.cuda.synchronize()
            now = time.time()
            if getattr(self, "_t", None) is not None:
                time.sleep(min((now - self._t) * a.idle, 5.0))
            self._t = time.time()
        sentences = sentences.long()
        pos = torch.arange(sentences.shape[1], device=sentences.device)
        out = gpt.h[0](gpt.wte(sentences) + gpt.wpe(pos))
        return [gpt.h[1].ln_1(out[0] if isinstance(out, tuple) else out)]

    def is_bert(self): return False
    def is_decoder(self): return True
    def has_rope(self): return False
    def has_bos(self): return False
    def is_lower(self): return False


def filter_l1(args, model_wrapper, R_Qs):
    """Copied from DAGER attack.py (no-defense branch for decoders)."""
    tokenizer_ = model_wrapper.tokenizer
    res_pos, res_ids, res_types = [], [], []
    sentence_ends = []
    p = 0
    n_tokens = 0
    while True:
        embeds = model_wrapper.get_embeddings(p)
        _, res_ids_new = get_top_B_in_span(R_Qs[0], embeds, args.batch_size, args.l1_span_thresh, args.dist_norm)
        res_types_new = torch.zeros_like(res_ids_new)
        res_pos_new = torch.ones_like(res_ids_new) * p
        del embeds
        res_types += [res_types_new.tolist()]
        ids = res_ids_new.tolist()
        if len(ids) == 0 or p > tokenizer_.model_max_length or p > args.max_len:
            break
        while model_wrapper.eos_token in ids:
            end_token_ind = ids.index(model_wrapper.eos_token)
            sentence_token_type = res_types[-1][end_token_ind]
            sentence_ends.append((p, sentence_token_type))
            ids = ids[:end_token_ind] + ids[end_token_ind + 1:]
            res_types[-1] = res_types[-1][:end_token_ind] + res_types[-1][end_token_ind + 1:]
        res_ids += [ids]
        res_pos += res_pos_new.tolist()
        n_tokens += len(ids)
        p += 1
    return res_pos, res_ids, res_types, sentence_ends


def dager(dw, B):
    """Recover the batch's texts from the GPT-2 part of the update (as token id lists)."""
    args = types.SimpleNamespace(
        batch_size=B, l1_span_thresh=a.l1_thresh, l2_span_thresh=a.l2_thresh, dist_norm="l2", max_len=63,
        device=dev, pad="right", parallel=a.parallel, defense_noise=None, distinct_thresh=0.7, reduce_incorrect=0,
        l2_std_thrs=5, n_layers=2, rank_cutoff=a.rank_cutoff)
    w = Wrapper(args)
    # DAGER's layer ids 4 and 16 are the c_attn weights of the first two blocks
    # as in DAGER's FedAvg code: the update divided by the learning rate and the number of steps
    scale = -1.0 / (a.lr_txt * a.u)
    grads = [scale * dw["txt.h.0.attn.c_attn.weight"].T, scale * dw["txt.h.1.attn.c_attn.weight"].T]
    # rank from a tolerance relative to the largest singular value (the authors' absolute tolerance
    # assumes the gradient scale of their loss); float64 for the singular values
    # The server subtracts float32 weights, so the update carries rounding noise in every direction.
    # Rank = singular values above noise_factor x the noise floor (median of the smallest 200).
    sv = [torch.linalg.svdvals(g.double()) for g in grads]
    rank = max(int((s > a.noise_factor * s[-200:].median()).sum()) for s in sv)
    Bq = min(rank + (60 if a.u > 1 else 0), w.emb_size - a.rank_cutoff)
    R_Qs = [get_layer_decomp(g, B=Bq)[1].to(dev) for g in grads]
    _, res_ids, _, _ = filter_l1(args, w, R_Qs)
    print(f"  rank {rank}, span size {Bq}, positions {len(res_ids)}, candidates per position: "
          f"{[len(r) for r in res_ids[:5]]} ... max {max((len(r) for r in res_ids), default=0)}", flush=True)
    if len(res_ids) == 0 or len(res_ids[0]) == 0:
        return [], {"rank": int(rank), "B_used": int(Bq), "positions": 0}
    sents, scores, _, _ = filter_decoder(args, w, R_Qs, res_ids)
    # keep the B best-scoring sentences, longest-first among equals is DAGER's convention via scores
    order = np.argsort(scores)[:B] if len(scores) else []
    return [sents[i] for i in order], {"rank": int(rank), "B_used": int(Bq), "positions": len(res_ids)}


def infer_labels(dw, texts_ids, B):
    """Label the recovered texts from the update of the head's text weights.

    For u = 1 the head (learning rate lr_img) moves by  g = (lr_img / B) * sum_i (y_i - p_i) z_i,  z_i the
    standardized text part of the head input (FusionModel.forward: head([img_std, txt_std])). Regressing g
    on the recovered z gives coefficients c_i = lr_img*u/B * (y_i - p_i), so |c_i| <= s = lr_img*u/B.
      sign:     y = 1[c_ls > 0]                (registered rule; plain least squares)
      combined: y = 1[p_hat + c_box / s > 0.5] (c_box: least squares with |c| <= s)
      model:    y = 1[p_hat > 0.5]             (no update used; the baseline)
    p_hat = text-only prediction of the broadcast model: image part of the head input set to 0, its
    training mean. B is the true batch size (known to the server). Exactly duplicate recovered texts are
    collapsed to one column (they share coefficient and labels). Solved on the CPU in float64 (SVD).
    Returns per-recovered-text lists and a few fit statistics."""
    n = len(texts_ids)
    if n == 0:
        return {}
    s = a.lr_img * a.u / B
    keys, grp = {}, []
    for t in texts_ids:
        k = tuple(int(v) for v in t) or (tok.eos_token_id,)  # an empty text (not expected) becomes one EOS
        grp.append(keys.setdefault(k, len(keys)))
    uniq = list(keys)
    L = max(len(t) for t in uniq)
    ids = torch.full((len(uniq), L), tok.eos_token_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, t in enumerate(uniq):
        ids[i, :len(t)] = torch.tensor(t); mask[i, :len(t)] = 1
    with torch.no_grad():
        h = model.text_features(ids.to(dev), mask.to(dev))
        Z = ((h - model.txt_mu) / model.txt_sd).double().cpu().numpy()   # (k, 768) text part of head input
    w = model.head.weight[0, 512:].detach().double().cpu().numpy()
    b = float(model.head.bias[0].detach().double())
    p_hat = 1.0 / (1.0 + np.exp(-(Z @ w + b)))
    g = dw["head.weight"][0, 512:].detach().double().cpu().numpy()        # update of the head's text weights
    c_ls, _, rank, sv = np.linalg.lstsq(Z.T, g, rcond=None)               # g ~ sum_i c_i z_i
    c_box = lsq_linear(Z.T, g, bounds=(-s, s), method="bvls").x
    # c > 0 means p_i < y_i, i.e. y_i = 1 (abnormal)
    lab_sign = (c_ls > 0).astype(int)
    lab_comb = (p_hat + c_box / s > 0.5).astype(int)
    lab_model = (p_hat > 0.5).astype(int)
    r2 = lambda c: float(1 - np.sum((g - Z.T @ c) ** 2) / np.sum(g ** 2))
    pick = lambda arr: [arr[j].item() for j in grp]
    return {"labels_sign": pick(lab_sign), "labels_combined": pick(lab_comb), "labels_model": pick(lab_model),
            "coef_ls": pick(c_ls), "coef_box": pick(c_box), "coef_bound": s, "p_hat_text_only": pick(p_hat),
            "label_fit": {"n_unique": len(uniq), "rank_Z": int(rank), "cond_Z": float(sv[0] / sv[-1]) if sv[-1] > 0 else None,
                          "r2_ls": r2(c_ls), "r2_box": r2(c_box), "n_box_at_bound": int(np.sum(np.abs(c_box) >= s * (1 - 1e-6)))}}


def score_labels(lab, pairs, y_true, thresh=0.5):
    """Accuracy over all originals (an original without a matched recovered text of ROUGE-L >= thresh gets the
    majority guess 1) and over the matched originals with ROUGE-L >= thresh only (with their number)."""
    good = {i: j for i, j, r in pairs if r >= thresh}
    pred_all = [lab[good[i]] if i in good else 1 for i in range(len(y_true))]
    acc_all = float(np.mean([p == t for p, t in zip(pred_all, y_true)]))
    acc_m = float(np.mean([lab[j] == y_true[i] for i, j in good.items()])) if good else None
    return {"acc_all_fallback1": round(acc_all, 4), "acc_matched": None if acc_m is None else round(acc_m, 4),
            "n_matched": len(good)}


# check the shortcut above against the full model once
with torch.no_grad():
    _t = torch.randint(0, 50000, (2, 12), device=dev)
    _ref = gpt.h[1].ln_1(gpt(input_ids=_t, output_hidden_states=True).hidden_states[1])
    _err = (Wrapper(None).get_layer_inputs(_t)[0] - _ref).abs().max().item()
    assert _err < 1e-4, _err

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
results = []
for v in a.victims:
    studies = split["victim_batches"][str(v)][: a.batch]
    rows = data.rows(studies)
    x, ids, mask, y = data.batch(rows, dev)
    torch.manual_seed(1000 + v)
    dw, logit0 = victim_update(x, ids, mask, y)
    t0 = time.time()
    rec_ids, info = dager(dw, len(rows))
    minutes = (time.time() - t0) / 60
    originals = [tok.decode(ids[i][: mask[i].sum()].tolist()) for i in range(len(rows))]
    recovered = [tok.decode(t) for t in rec_ids]
    y_true = [int(t) for t in y.tolist()]
    lab = infer_labels(dw, rec_ids, len(rows))
    # match recovered to original texts by the one-to-one assignment maximizing ROUGE-L
    pairs = []
    if recovered:
        S = np.array([[scorer.score(o, r)["rougeL"].fmeasure for r in recovered] for o in originals])
        ri, ci = linear_sum_assignment(-S)
        rl = [0.0] * len(originals)
        for i, j in zip(ri, ci):
            rl[i] = S[i, j]
            pairs.append((int(i), int(j), float(S[i, j])))
        r1 = np.mean([scorer.score(originals[i], recovered[j])["rouge1"].fmeasure for i, j in zip(ri, ci)] + [0.0] * (len(originals) - len(ri)))
        # registered measure, kept for comparability: sign rule over the returned texts (every assigned pair)
        label_acc = np.mean([lab["labels_sign"][j] == y_true[i] for i, j in zip(ri, ci)])
    else:
        rl, r1, label_acc = [0.0] * len(originals), 0.0, float("nan")
    label_scores = {}
    if recovered:
        for rule in ("sign", "combined", "model"):
            label_scores[rule] = score_labels(lab["labels_" + rule], pairs, y_true)
    good = [i for i, _, r in pairs if r >= 0.5]
    label_scores["always_abnormal"] = {"acc_all_fallback1": round(float(np.mean(y_true)), 4),
                                       "acc_matched": round(float(np.mean([y_true[i] for i in good])), 4) if good else None,
                                       "n_matched": len(good)}
    rec = {"victim": v, "batch": len(rows), "u": a.u, "rougeL": round(float(np.mean(rl)), 4), "rouge1": round(float(r1), 4),
           "n_recovered": len(recovered), "label_acc": None if np.isnan(label_acc) else round(float(label_acc), 3),
           "minutes": round(minutes, 2), **info,
           "examples": [{"orig": originals[i], "rec": recovered[j]} for i, j in list(zip(ri, ci))[:2]] if recovered else [],
           # label step (scores: fallback guess 1 for originals without a match of ROUGE-L >= 0.5)
           "label_scores": label_scores, "labels_true": y_true,
           "rec_ids": [[int(t) for t in s] for s in rec_ids],
           "matching": [{"orig": i, "rec": j, "rougeL": round(r, 4)} for i, j, r in pairs],
           **lab,  # per recovered index: labels_sign/combined/model, coef_ls, coef_box, p_hat_text_only; coef_bound, label_fit
           # calibration information (client side, never used by a rule): |p - y| of each original under the
           # full model at the broadcast weights (first local step, the client's own forward pass)
           "calib_abs_residual": [round(abs(float(p) - t), 4) for p, t in zip(torch.sigmoid(logit0).tolist(), y_true)]}
    results.append(rec)
    print(f"victim {v} B={len(rows)} u={a.u}: ROUGE-L {rec['rougeL']:.3f} ROUGE-1 {rec['rouge1']:.3f} "
          f"recovered {len(recovered)}/{len(rows)} label acc {rec['label_acc']} rank {info['rank']} t={rec['minutes']}min", flush=True)
    print("  labels " + " | ".join(f"{k}: all {d['acc_all_fallback1']} matched {d['acc_matched']} (n={d['n_matched']})"
                                   for k, d in label_scores.items()), flush=True)
    json.dump(rec, open(os.path.join(out, f"v{v}_B{len(rows)}_u{a.u}{a.tag}.json"), "w"), indent=1)
print("SUMMARY" + a.tag, json.dumps([{k: r[k] for k in ("victim", "batch", "u", "rougeL", "rouge1", "n_recovered", "label_acc", "minutes")} for r in results]))
