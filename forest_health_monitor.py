"""
Forest Health Monitor (Uttarakhand)
-----------------------------------
Uses uploaded 'combined.json' to classify vegetation density, detect trends,
and optionally analyze image-based changes.

Run: python forest_health_monitor.py
Dependencies: numpy, pandas, matplotlib, scikit-learn, joblib, opencv-python
"""

import json, os, re, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# === Paths ===
DATA_PATH = "/mnt/data/combined.json"   # your uploaded dataset
OUT_DIR = "reports"; os.makedirs(OUT_DIR, exist_ok=True)

# === Helper ===
def safe_float(x):
    try:
        return float(str(x).replace(",", "").replace("%", "").strip())
    except: return np.nan

def forest_density_proxy(row):
    """Compute vegetation proxy = 1 - (wasteland / total area)."""
    total = row.get("total_geographical_area_hectares") or row.get("total_geographical_area_sqkm")
    waste = (row.get("total_wasteland_area_sq_km") or 
             row.get("total_degraded_area_hectares"))
    if total and waste and total > 0:
        return max(0, 1 - waste / total)
    return np.nan

# === Load and structure JSON ===
data = json.load(open(DATA_PATH, "r"))
rows = []
for k, v in (data.items() if isinstance(data, dict) else enumerate(data)):
    if isinstance(v, dict):
        rec = {kk: safe_float(vv) for kk, vv in v.items()}
        rec["title"] = str(v.get("report_title") or v.get("document_title") or k)
        rec["density"] = forest_density_proxy(rec)
        rows.append(rec)
df = pd.DataFrame(rows)

# === Filter Uttarakhand records ===
mask = df["title"].str.contains("uttarakhand", case=False, na=False)
df = df[mask].copy()
print(f"Uttarakhand records: {len(df)}")

# === Train simple SVR ===
features = [c for c in df.columns if df[c].dtype != object and c != "density"]
df = df.dropna(subset=["density"])
X, y = df[features].fillna(0), df["density"]
if len(X) > 5:
    Xtr, Xts, ytr, yts = train_test_split(X, y, test_size=0.3, random_state=42)
    model = SVR(kernel="rbf").fit(Xtr, ytr)
    preds = model.predict(Xts)
    print(f"MSE={mean_squared_error(yts,preds):.4f}, MAE={mean_absolute_error(yts,preds):.4f}, R2={r2_score(yts,preds):.4f}")
    joblib.dump(model, os.path.join(OUT_DIR, "svr_model.joblib"))
else:
    print("Not enough data for training.")

# === Save report ===
df[["title","density"]].to_csv(os.path.join(OUT_DIR,"summary.csv"), index=False)
plt.plot(df.index, df["density"], marker='o')
plt.title("Vegetation Density Trend - Uttarakhand")
plt.ylabel("Density Proxy")
plt.xlabel("Record Index")
plt.grid(); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"trend.png"))
print("Reports saved in /reports folder.")
