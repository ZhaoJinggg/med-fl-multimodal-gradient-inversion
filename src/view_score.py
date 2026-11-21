"""Score every IU X-Ray image for left-right symmetry (frontal views are near mirror-symmetric).
Writes data/iuxray/view_scores.csv: image id, width, height, symmetry score."""
import csv
import os
import sys

import numpy as np
from PIL import Image

root = sys.argv[1]
img_dir = os.path.join(root, "data", "iuxray", "extracted", "images")
rows = []
for name in sorted(os.listdir(img_dir)):
    if not name.endswith(".png"):
        continue
    im = Image.open(os.path.join(img_dir, name)).convert("L")
    w, h = im.size
    a = np.asarray(im.resize((64, 64), Image.BILINEAR), dtype=np.float32)
    a = (a - a.mean()) / (a.std() + 1e-6)
    sym = float((a * a[:, ::-1]).mean())  # correlation with the mirror image
    rows.append((name[:-4], w, h, round(sym, 4)))
out = os.path.join(root, "data", "iuxray", "view_scores.csv")
with open(out, "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["image_id", "width", "height", "symmetry"])
    wr.writerows(rows)
s = np.array([r[3] for r in rows])
print(len(rows), "images; symmetry quantiles 5/25/50/75/95:", np.round(np.percentile(s, [5, 25, 50, 75, 95]), 3))
print("histogram:", np.histogram(s, bins=10, range=(-0.2, 1.0))[0].tolist())
