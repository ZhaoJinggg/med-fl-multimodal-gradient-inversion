"""Frontal/lateral view classifier for IU X-Ray.
Labels come from 2-image studies whose two images differ clearly in left-right symmetry
(more symmetric = frontal). A logistic regression on frozen ImageNet ResNet-18 features is
trained on those, checked by cross-validation, and applied to every image.
Writes data/iuxray/view_pred.csv (image_id, p_frontal)."""
import csv, glob, os, sys
import xml.etree.ElementTree as ET
import numpy as np
import torch, torchvision
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

root = sys.argv[1]
os.environ.setdefault("TORCH_HOME", os.path.join(root, "pretrained", "torch"))
sc = {r["image_id"]: float(r["symmetry"]) for r in csv.DictReader(open(os.path.join(root, "data/iuxray/view_scores.csv")))}
ids = sorted(sc)
dev = "cuda" if torch.cuda.is_available() else "cpu"
m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
m.fc = torch.nn.Identity(); m.eval().to(dev)
tf = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)), torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
feats = []
with torch.no_grad():
    for i in range(0, len(ids), 64):
        x = torch.stack([tf(Image.open(os.path.join(root, "data/iuxray/extracted/images", k + ".png")).convert("RGB")) for k in ids[i:i + 64]])
        feats.append(m(x.to(dev)).cpu().numpy())
F = np.concatenate(feats); idx = {k: i for i, k in enumerate(ids)}

X, y = [], []
for f in glob.glob(os.path.join(root, "data/iuxray/extracted/ecgen-radiology/*.xml")):
    s = [p.get("id") for p in ET.parse(f).getroot().iter("parentImage") if p.get("id") in sc]
    if len(s) == 2 and abs(sc[s[0]] - sc[s[1]]) > 0.4:
        a, b = sorted(s, key=lambda k: -sc[k])
        X += [F[idx[a]], F[idx[b]]]; y += [1, 0]
X, y = np.array(X), np.array(y)
clf = LogisticRegression(C=1.0, max_iter=3000)
print("training images:", len(y), "| 5-fold CV accuracy: %.4f" % cross_val_score(clf, X, y, cv=5).mean())
clf.fit(X, y)
p = clf.predict_proba(F)[:, 1]
with open(os.path.join(root, "data/iuxray/view_pred.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["image_id", "p_frontal"]); w.writerows((k, round(float(p[idx[k]]), 4)) for k in ids)
print("predicted frontal:", int((p > 0.5).sum()), "of", len(ids), "| uncertain (0.2 < p < 0.8):", int(((p > 0.2) & (p < 0.8)).sum()))
