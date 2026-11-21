"""Phase 1: build image-text pairs, the fixed split, the client partitions and the victim batches.

Pairs: one frontal image per study (highest p_frontal from view_pred.csv, must be > 0.5) and the
FINDINGS section (non-empty). Label: 0 = normal (MeSH major term "normal"), 1 = abnormal.
Text is truncated to 64 GPT-2 tokens later, at model input.

Split (fixed, independent of the experiment seeds): 20% test, 10% validation, 70% train,
stratified by label. The validation pairs serve only to choose the learning rates.

Partition per seed: 10 equal-sized clients; client k's share of abnormal pairs q_k ~ Beta(0.5, 0.5)
(= Dirichlet(0.5, 0.5) for two classes), drawn independently; if the demand for either class
exceeds its supply, all q are drawn again.

Victims: seed 0 (calibration) uses all 10 clients; seeds 1-5 use 4 clients drawn at random.
Each victim has one fixed batch of 8 pairs drawn from its own data.
"""
import csv
import glob
import json
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

root = sys.argv[1]
out_dir = os.path.join(root, "data", "splits")
os.makedirs(out_dir, exist_ok=True)
N_CLIENTS, VICTIM_B, SEEDS, SPLIT_SEED = 10, 8, range(6), 12345

pf = {r["image_id"]: float(r["p_frontal"]) for r in csv.DictReader(open(os.path.join(root, "data/iuxray/view_pred.csv")))}

pairs = []
skipped = {"no_findings": 0, "no_image": 0, "no_frontal": 0}
for f in sorted(glob.glob(os.path.join(root, "data/iuxray/extracted/ecgen-radiology/*.xml"))):
    t = ET.parse(f).getroot()
    findings = None
    for a in t.iter("AbstractText"):
        if a.get("Label") == "FINDINGS" and a.text and a.text.strip():
            findings = " ".join(a.text.split())
    imgs = [p.get("id") for p in t.iter("parentImage") if p.get("id") in pf]
    if not findings:
        skipped["no_findings"] += 1; continue
    if not imgs:
        skipped["no_image"] += 1; continue
    best = max(imgs, key=lambda k: pf[k])
    if pf[best] <= 0.5:
        skipped["no_frontal"] += 1; continue
    majors = [m.text.strip().lower() for m in t.iter("major") if m.text]
    label = 0 if majors == ["normal"] else 1
    pairs.append({"study": os.path.basename(f)[:-4], "image_id": best, "label": label, "findings": findings})

# fixed stratified split
rng = np.random.default_rng(SPLIT_SEED)
for lab in (0, 1):
    idx = [i for i, p in enumerate(pairs) if p["label"] == lab]
    rng.shuffle(idx)
    n_test, n_val = round(0.2 * len(idx)), round(0.1 * len(idx))
    for j, i in enumerate(idx):
        pairs[i]["split"] = "test" if j < n_test else ("val" if j < n_test + n_val else "train")

with open(os.path.join(out_dir, "pairs.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["study", "image_id", "label", "split", "findings"])
    w.writeheader(); w.writerows(pairs)

train = [p for p in pairs if p["split"] == "train"]
by_label = {lab: [p["study"] for p in train if p["label"] == lab] for lab in (0, 1)}
n_per_client = len(train) // N_CLIENTS
print("pairs:", len(pairs), "| skipped:", skipped)
for sp in ("train", "val", "test"):
    s = [p for p in pairs if p["split"] == sp]
    print(f"  {sp}: {len(s)} pairs, abnormal share {np.mean([p['label'] for p in s]):.3f}")
print("client size:", n_per_client, "| train normal / abnormal:", len(by_label[0]), len(by_label[1]))

summary = {}
for seed in SEEDS:
    rng = np.random.default_rng(seed)
    draws = 0
    while True:
        draws += 1
        q = rng.beta(0.5, 0.5, N_CLIENTS)
        n_ab = np.round(q * n_per_client).astype(int)
        n_no = n_per_client - n_ab
        if n_ab.sum() <= len(by_label[1]) and n_no.sum() <= len(by_label[0]):
            break
    pools = {lab: list(rng.permutation(by_label[lab])) for lab in (0, 1)}
    clients = {}
    for k in range(N_CLIENTS):
        take_ab = [pools[1].pop() for _ in range(n_ab[k])]
        take_no = [pools[0].pop() for _ in range(n_no[k])]
        clients[k] = sorted(take_ab + take_no)
    victims = list(range(N_CLIENTS)) if seed == 0 else sorted(rng.choice(N_CLIENTS, 4, replace=False).tolist())
    batches = {k: sorted(rng.choice(clients[k], VICTIM_B, replace=False).tolist()) for k in victims}
    json.dump({"seed": seed, "redraws": draws, "abnormal_share": q.round(4).tolist(),
               "clients": {str(k): v for k, v in clients.items()},
               "victim_batches": {str(k): v for k, v in batches.items()}},
              open(os.path.join(out_dir, f"seed{seed}.json"), "w"), indent=1)
    summary[seed] = (draws, victims, [round(float(x), 2) for x in n_ab / n_per_client])
    print(f"seed {seed}: draws {draws}, victims {victims}, abnormal share per client {summary[seed][2]}")
