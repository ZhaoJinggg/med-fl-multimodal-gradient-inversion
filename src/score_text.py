"""Tidy rows for src/analyze.py from the result JSON files of src/attack_text.py (CPU only, no model).

One row per result file (= per victim and cell):
  seed, victim, protocol, u, modality = "text", value = ROUGE-L (mean over all B originals, unmatched 0);
  condition: "recovered" for a main-experiment cell; anything else keeps the row out of every test and out
       of the ANOVA of analyze.py (it only uses condition == "recovered"):
         "exact"   the reference run on the update without rounding (attack_text.py --exact; upper bound,
                   reported next to the main result, never tested)
         "B<k>"    a batch size other than --main-batch (the descriptive batch-size comparison)
         "optI", "bnlocal", "mu<x>"  protocol variants outside the paper's cells (SCAFFOLD Option I,
                   BN statistics not uploaded, FedProx mu other than --main-mu)
       parts are joined with "_", e.g. "exact_B4".

Zero point and controls (all scored like the attack: ROUGE-L, one-to-one assignment maximizing the mean
ROUGE-L, mean over all B originals of the victim with an unassigned original scoring 0; the victim's texts
are its first B batch reports, truncated to 64 GPT-2 tokens and decoded, as in attack_text.py):
  m0                 the guessing baseline m_0 (paper, Evaluation Metrics, revised): the B most frequent
                     findings reports of the PUBLIC pairs (validation + test split, held by no client;
                     PairData.split("val") + split("test")), truncated like the victims' texts, matched to the
                     victim's B texts. Ties in frequency: first occurrence in the pairs file (Counter order).
  m0_single          the earlier definition (reference): the single most frequent report of the seed's client
                     data without the victim's batch, repeated for every text (src/text_baseline.py).
  rougeL_mismatched  the mismatched control: the texts recovered for every OTHER victim of the same cell (same
                     directory, protocol, u, batch, exact flag and file-name suffix = cell suffix + tag; with
                     --mismatch-scope cell, across directories, e.g. over all seeds), each set matched to this
                     victim's texts and scored; mean over the other victims (needs the saved rec_ids).
  rougeL_recomputed  the victim's own ROUGE-L recomputed from rec_ids (check of the stored value).

Label metrics of the registered sign rule (reported, not tested):
  label_acc_returned       (a) accuracy over every returned text against the original it was matched to (what
                           the attacker gets; = attack_text.py's label_acc), n_returned texts
  label_acc_matched        (b) over the returned texts matched with ROUGE-L >= 0.5, with their number n_label_matched
  label_acc_random_counts  (c) count-respecting random baseline: an attacker who knows the batch's label counts
                           (k abnormal of B) and assigns them at random is right with probability
                           (k^2 + (B - k)^2) / B^2; labels_mixed says whether the batch has both labels (n_abnormal = k)
  label_acc_majority       the majority label (of the seed's client data without the batch) for every text
  label_acc_always_abnormal  label 1 for every text
  ..._oracle_fallback      the JSON's "all B" accuracies (an original without a returned text of ROUGE-L >= 0.5 gets
                           label 1): an UPPER BOUND, since the fallback uses ROUGE-L, which the attacker cannot know
                           (sign, combined, model rules); combined / model matched accuracies for reference.
Other auxiliary metrics: rouge1, rouge2, token_acc (all over all B originals on the ROUGE-L matching), n_recovered,
the cell's settings (protocol options, thresholds, broadcast model).

Files that predate the cell fields: seed from --default-seed; exact = "exact" in the file name; protocol fedavg;
missing metrics are left empty. Non-result JSON files (e.g. text_baseline.py output) are skipped.

usage: python src/score_text.py runs/main/text runs/main/text_exact --out runs/scores/text_tidy.csv
       python src/analyze.py test runs/scores/text_tidy.csv [image_tidy.csv] --out runs/analysis
"""
import argparse
import collections
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from rouge_score import rouge_scorer
from scipy.optimize import linear_sum_assignment

from common import load_pairs, load_seed, tokenize, tokenizer

PROTOCOLS = ("fedavg", "fedprox", "scaffold", "fedbn")


def match_rougeL(originals, hyps, scorer):
    """ROUGE-L of each original under the one-to-one assignment maximizing the mean ROUGE-L (0 for an original
    left without a text); the same arithmetic as attack_text.match_texts. Returns the per-original list."""
    rl = [0.0] * len(originals)
    if hyps:
        S = np.array([[scorer.score(o, r)["rougeL"].fmeasure for r in hyps] for o in originals])
        ri, ci = linear_sum_assignment(-S)
        for i, j in zip(ri, ci):
            rl[i] = S[i, j]
    return rl


class TextData:
    """The reports (tokenized and truncated as PairData), labels, the victims' texts and the baselines."""

    def __init__(self):
        self.pairs = load_pairs()
        self.tok = tokenizer()
        self.ids, self.mask = tokenize(self.tok, [p["findings"] for p in self.pairs])
        self.y = [float(p["label"]) for p in self.pairs]
        self.index = {p["study"]: i for i, p in enumerate(self.pairs)}
        self.scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        self._text, self._cache, self._splits = {}, {}, {}
        public = [i for i, p in enumerate(self.pairs) if p["split"] in ("val", "test")]
        self.public_counts = collections.Counter(self.text(i) for i in public)
        self.n_public = len(public)

    def text(self, i):
        if i not in self._text:
            self._text[i] = self.tok.decode(self.ids[i][: int(self.mask[i].sum())].tolist())
        return self._text[i]

    def split(self, seed):
        if seed not in self._splits:
            self._splits[seed] = load_seed(seed)
        return self._splits[seed]

    def victim_rows(self, seed, victim, batch):
        """The victim's batch rows in the attack's order (attack_text.py: data.rows(studies[:batch]))."""
        return [self.index[s] for s in self.split(seed)["victim_batches"][str(victim)][:batch]]

    def originals(self, seed, victim, batch):
        return [self.text(i) for i in self.victim_rows(seed, victim, batch)]

    def __call__(self, seed, victim, batch):
        key = (int(seed), int(victim), int(batch))
        if key not in self._cache:
            split = self.split(seed)
            rows = self.victim_rows(seed, victim, batch)
            vset = set(rows)
            truth = [self.text(i) for i in rows]
            # earlier zero point (src/text_baseline.py): same pool order and tie rule
            pool = [i for c in split["clients"].values() for i in (self.index[s] for s in c) if i not in vset]
            frequent, count = collections.Counter(self.text(i) for i in pool).most_common(1)[0]
            m0_single = float(np.mean([self.scorer.score(self.text(i), frequent)["rougeL"].fmeasure for i in sorted(vset)]))
            majority = int(np.mean([self.y[i] for i in pool]) >= 0.5)
            # revised zero point: the B most frequent public reports, matched
            top = [t for t, _ in self.public_counts.most_common(batch)]
            labels = [int(self.y[i]) for i in rows]
            k = sum(labels)
            self._cache[key] = {
                "m0": float(np.mean(match_rougeL(truth, top, self.scorer))),
                "m0_single": m0_single, "m0_single_count": count,
                "m0_public_counts": ",".join(str(self.public_counts[t]) for t in top),
                "majority_label": majority,
                "label_acc_majority": float(np.mean([lab == majority for lab in labels])),
                "label_acc_always_abnormal": k / len(labels),
                "label_acc_random_counts": (k ** 2 + (len(labels) - k) ** 2) / len(labels) ** 2,
                "n_abnormal": k, "labels_mixed": 0 < k < len(labels), "labels": labels}
        return self._cache[key]


def result_files(inputs):
    """Directories (searched recursively for *.json), glob patterns, or files; sorted, without duplicates."""
    out = []
    for p in inputs:
        if os.path.isdir(p):
            out += glob.glob(os.path.join(p, "**", "*.json"), recursive=True)
        elif any(ch in p for ch in "*?["):
            out += glob.glob(p, recursive=True)
        elif os.path.exists(p):
            out.append(p)
        else:
            print(f"WARNING: {p} not found", file=sys.stderr)
    return sorted(set(os.path.normpath(f) for f in out))


def _num(x):
    return np.nan if x is None else x


def file_suffix(path):
    """The part of the file name after v{v}_B{b}_u{u}: cell suffix of attack_text.py plus --tag."""
    return re.sub(r"^v\d+_B\d+_u\d+", "", os.path.basename(path))[: -len(".json")]


def row_of(path, rec, args, td):
    """The tidy row of one result file, or None (with the reason) if it cannot be placed in a cell."""
    seed = rec.get("seed", args.default_seed)
    if seed is None:
        return None, "no seed in the file and no --default-seed"
    proto = str(rec.get("proto", "fedavg")).lower()
    if proto not in PROTOCOLS:
        return None, f"unknown protocol {proto}"
    exact = rec.get("exact")
    exact_inferred = exact is None
    if exact is None:
        exact = "exact" in os.path.basename(path).lower()
    batch, u, victim = int(rec["batch"]), int(rec["u"]), int(rec["victim"])
    mu = rec.get("mu", args.main_mu if proto == "fedprox" else 0.0)
    parts = []
    if exact:
        parts.append("exact")
    if batch != args.main_batch:
        parts.append(f"B{batch}")
    if proto == "scaffold" and rec.get("scaffold_option", "II") != "II":
        parts.append(f"opt{rec['scaffold_option']}")
    if proto != "fedbn" and rec.get("bn_stats", "upload") == "local":
        parts.append("bnlocal")
    if proto == "fedprox" and mu is not None and float(mu) != args.main_mu:
        parts.append(f"mu{float(mu):g}")
    b = td(seed, victim, batch)
    if "labels_true" in rec and rec["labels_true"] != b["labels"]:
        return None, "labels_true differ from the seed's victim batch (wrong --default-seed?)"
    ls = rec.get("label_scores") or {}
    sign, always = ls.get("sign"), ls.get("always_abnormal") or {}
    if sign is None and rec.get("n_recovered") == 0 and always:
        # no text returned: every text gets the fallback label, nothing is matched
        sign = {"acc_all_fallback1": always.get("acc_all_fallback1"), "acc_matched": None, "n_matched": 0}
    sign = sign or {}
    comb, model = ls.get("combined") or {}, ls.get("model") or {}
    # (a) sign rule over every returned text against its matched original
    if rec.get("matching") is not None and "labels_sign" in rec:
        hits = [rec["labels_sign"][m["rec"]] == b["labels"][m["orig"]] for m in rec["matching"]]
        acc_ret, n_ret = (float(np.mean(hits)) if hits else np.nan), len(hits)
    else:
        acc_ret, n_ret = _num(rec.get("label_acc")), rec.get("n_recovered")
    # the victim's own ROUGE-L recomputed from the saved token ids (check)
    rec_texts = [td.tok.decode(t) for t in rec["rec_ids"]] if "rec_ids" in rec else None
    rl_re = float(np.mean(match_rougeL(td.originals(seed, victim, batch), rec_texts, td.scorer))) if rec_texts is not None else np.nan
    row = {"seed": int(seed), "victim": victim, "protocol": proto, "u": u, "modality": "text",
           "value": rec["rougeL"], "m0": b["m0"], "condition": "_".join(parts) if parts else "recovered",
           "batch": batch, "exact": bool(exact),
           "m0_single": b["m0_single"], "rougeL_mismatched": np.nan, "n_mismatched": 0, "rougeL_recomputed": rl_re,
           "rouge1": _num(rec.get("rouge1")), "rouge2": _num(rec.get("rouge2")), "token_acc": _num(rec.get("token_acc")),
           "n_recovered": rec.get("n_recovered"),
           "label_acc_returned": acc_ret, "n_returned": n_ret,
           "label_acc_matched": _num(sign.get("acc_matched")), "n_label_matched": _num(sign.get("n_matched")),
           "label_acc_random_counts": b["label_acc_random_counts"], "labels_mixed": b["labels_mixed"],
           "n_abnormal": b["n_abnormal"], "label_acc_majority": b["label_acc_majority"], "majority_label": b["majority_label"],
           "label_acc_always_abnormal": b["label_acc_always_abnormal"],
           "label_acc_oracle_fallback": _num(sign.get("acc_all_fallback1")),
           "label_acc_combined_oracle_fallback": _num(comb.get("acc_all_fallback1")),
           "label_acc_combined_matched": _num(comb.get("acc_matched")),
           "label_acc_model_oracle_fallback": _num(model.get("acc_all_fallback1")),
           "label_acc_model_matched": _num(model.get("acc_matched")),
           "rank": _num(rec.get("rank")), "B_used": _num(rec.get("B_used")), "positions": _num(rec.get("positions")),
           "minutes": _num(rec.get("minutes")),
           "mu": _num(mu), "scaffold_option": rec.get("scaffold_option"), "bn_stats": rec.get("bn_stats"),
           "attack_input": rec.get("attack_input"), "aux_round": _num(rec.get("aux_round")),
           "noise_factor": _num(rec.get("noise_factor")), "l1_thresh": _num(rec.get("l1_thresh")),
           "l2_thresh": _num(rec.get("l2_thresh")), "max_beams": _num(rec.get("max_beams")),
           "lr_img": _num(rec.get("lr_img")), "lr_txt": _num(rec.get("lr_txt")), "res": _num(rec.get("res")),
           "eta_mu_u_img": _num(rec.get("eta_mu_u_img")), "eta_mu_u_txt": _num(rec.get("eta_mu_u_txt")),
           "global": rec.get("global"), "m0_public_counts": b["m0_public_counts"], "m0_single_count": b["m0_single_count"],
           "exact_from_filename": exact_inferred, "file": path.replace("\\", "/")}
    return (row, rec_texts), None


def add_mismatched(df, texts, td, scope):
    """rougeL_mismatched: the other victims' recovered texts of the same cell, matched to this victim's texts."""
    key = lambda r: ((os.path.dirname(r["file"]) if scope == "directory" else ""), r["protocol"], r["u"], r["batch"],
                     r["exact"], file_suffix(r["file"]))
    groups = collections.defaultdict(list)
    for idx, r in df.iterrows():
        groups[key(r)].append(idx)
    for members in groups.values():
        for i in members:
            r = df.loc[i]
            orig = td.originals(r["seed"], r["victim"], r["batch"])
            scores = [float(np.mean(match_rougeL(orig, texts[j], td.scorer))) for j in members
                      if j != i and texts[j] is not None and (df.loc[j, "seed"], df.loc[j, "victim"]) != (r["seed"], r["victim"])]
            df.loc[i, "rougeL_mismatched"] = float(np.mean(scores)) if scores else np.nan
            df.loc[i, "n_mismatched"] = len(scores)
    return df


def tidy(inputs, args):
    td = TextData()
    rows, texts, skipped = [], [], []
    for f in result_files(inputs):
        try:
            rec = json.load(open(f, encoding="utf-8"))
        except (OSError, ValueError) as e:
            skipped.append((f, f"unreadable: {e}"))
            continue
        if not isinstance(rec, dict) or "rougeL" not in rec or "victim" not in rec:
            continue                                    # not an attack_text.py result (e.g. a baseline file)
        out, why = row_of(f, rec, args, td)
        if out is None:
            skipped.append((f, why))
        else:
            rows.append(out[0])
            texts.append(out[1])
    for f, why in skipped:
        print(f"WARNING: skipped {f}: {why}", file=sys.stderr)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    dup = df[df.duplicated(["seed", "victim", "protocol", "u", "condition"], keep=False)]
    if not dup.empty:
        raise SystemExit("several files for the same victim and cell (analyze.py would reject them):\n"
                         + dup[["seed", "victim", "protocol", "u", "condition", "file"]].to_string(index=False))
    df = add_mismatched(df, texts, td, args.mismatch_scope)
    bad = df[(df.rougeL_recomputed - df.value).abs() > 1e-3]
    if not bad.empty:
        print("WARNING: ROUGE-L recomputed from rec_ids differs from the stored value:\n"
              + bad[["file", "value", "rougeL_recomputed"]].to_string(index=False), file=sys.stderr)
    main = df[df.condition == "recovered"]
    thr = main[["noise_factor", "l1_thresh", "l2_thresh"]].drop_duplicates()
    if len(thr) > 1:
        print("WARNING: the recovered (main) rows use different DAGER settings:\n" + thr.to_string(index=False), file=sys.stderr)
    if (df.majority_label != 1).any():
        print("WARNING: majority label 0 for some victims; attack_text.py's oracle fallback uses label 1", file=sys.stderr)
    return df.sort_values(["condition", "protocol", "u", "seed", "victim"]).reset_index(drop=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="result JSON files, directories or glob patterns")
    ap.add_argument("--out", required=True, help="tidy CSV")
    ap.add_argument("--default-seed", type=int, default=None, help="seed of files that do not record it")
    ap.add_argument("--main-batch", type=int, default=8, help="batch size of the main experiment")
    ap.add_argument("--main-mu", type=float, default=1.0, help="FedProx mu of the main experiment")
    ap.add_argument("--mismatch-scope", choices=["directory", "cell"], default="directory",
                    help="other victims for rougeL_mismatched: same directory (default) or any directory (all seeds)")
    args = ap.parse_args(argv)
    df = tidy(args.inputs, args)
    if df.empty:
        raise SystemExit("no attack_text.py result found")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)
    cols = ["value", "m0", "m0_single", "rougeL_mismatched", "rouge1", "rouge2", "token_acc", "n_recovered",
            "label_acc_returned", "label_acc_matched", "n_label_matched", "label_acc_random_counts", "label_acc_majority"]
    summary = df.groupby(["condition", "protocol", "u"])[cols].mean().round(4)
    summary.insert(0, "n", df.groupby(["condition", "protocol", "u"]).size())
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(summary.to_string())
    print(f"wrote {args.out} ({len(df)} rows; {int((df.condition == 'recovered').sum())} in recovered cells)")
    return df


if __name__ == "__main__":
    main()
