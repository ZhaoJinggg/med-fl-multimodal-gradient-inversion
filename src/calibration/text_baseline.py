"""Guessing baseline for the text attack (CPU only): an attacker without the update outputs the same
report for every text of the batch. Two guesses, both from the seed's client data without the victim's
batch: the most frequent report, and the report with the highest mean ROUGE-L to a sample of the others
(the strongest single guess). Scored like the attack: ROUGE-L against the truncated victim texts.
Label baseline: always the majority label.

usage: python src/calibration/text_baseline.py --seed 0 --victims 0 1 2 --batch 8
"""
import argparse
import collections
import json
import os
import random

import numpy as np
from rouge_score import rouge_scorer

import os as _os, sys as _sys  # noqa: E401,E402
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # src/ (common.py)
from common import ROOT, PairData, load_seed, tokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--victims", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--sample", type=int, default=300, help="reports used to find the strongest single guess")
ap.add_argument("--out", default="runs/pilot/text128")
a = ap.parse_args()
data = PairData(128)
split = load_seed(a.seed)
tok = tokenizer()
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)


def text(i):
    return tok.decode(data.ids[i][: int(data.mask[i].sum())].tolist())


def rl(ref, hyp):
    return scorer.score(ref, hyp)["rougeL"].fmeasure


results = []
for v in a.victims:
    victim = set(data.rows(split["victim_batches"][str(v)][: a.batch]).tolist())
    pool = [i for c in split["clients"].values() for i in data.rows(c).tolist() if i not in victim]
    texts = [text(i) for i in pool]
    frequent, count = collections.Counter(texts).most_common(1)[0]
    random.seed(0)
    sample = random.sample(sorted(set(texts)), min(a.sample, len(set(texts))))
    central = max(sample, key=lambda t: np.mean([rl(o, t) for o in sample]))
    truth = [text(i) for i in sorted(victim)]
    majority = int(np.mean([data.y[i].item() for i in pool]) >= 0.5)
    rec = {"victim": v, "batch": len(truth),
           "rougeL_frequent": round(float(np.mean([rl(t, frequent) for t in truth])), 4),
           "rougeL_central": round(float(np.mean([rl(t, central) for t in truth])), 4),
           "label_acc_majority": round(float(np.mean([int(data.y[i].item()) == majority for i in sorted(victim)])), 3),
           "frequent_count": count, "frequent": frequent, "central": central}
    results.append(rec)
    print(f"victim {v}: most frequent report (x{count}) ROUGE-L {rec['rougeL_frequent']:.3f}; "
          f"strongest single guess {rec['rougeL_central']:.3f}; majority-label accuracy {rec['label_acc_majority']}", flush=True)
json.dump(results, open(os.path.join(ROOT, a.out, f"baseline_seed{a.seed}_B{a.batch}.json"), "w"), indent=1)
print("MEAN frequent %.3f central %.3f label %.3f" % tuple(np.mean([r[k] for r in results]) for k in
                                                           ("rougeL_frequent", "rougeL_central", "label_acc_majority")))
