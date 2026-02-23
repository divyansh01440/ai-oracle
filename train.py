"""
train.py  — Improved version
Trains the AI Oracle model with better labeling so it produces
varied, meaningful risk scores instead of always outputting near-zero.

Run with:  python train.py
"""

import os, pickle, requests
import numpy as np
import pandas as pd

print("=" * 60)
print("  AI DOCAL — MODEL TRAINING (IMPROVED)")
print("=" * 60)

def install(p):
    os.system(f"pip install {p} --quiet --break-system-packages")

try:
    import torch, torch.nn as nn
except ImportError:
    install("torch"); import torch, torch.nn as nn

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
except ImportError:
    install("scikit-learn")
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

# ── Fetch 1000 candles ────────────────────────────────────────
print("\nFetching 1000 candles from Binance...")
resp = requests.get("https://api.binance.com/api/v3/klines", params={
    "symbol": "MATICUSDT", "interval": "1m", "limit": 1000
}, timeout=15)
resp.raise_for_status()
raw = resp.json()
print(f"  ✓ Got {len(raw)} candles")

df = pd.DataFrame(raw, columns=[
    "open_time","open","high","low","close","volume",
    "close_time","quote_vol","num_trades","taker_base","taker_quote","ignore"
])
for col in ["open","high","low","close","volume","taker_base"]:
    df[col] = df[col].astype(float)
df["num_trades"] = df["num_trades"].astype(int)

# ── Features ──────────────────────────────────────────────────
print("Computing features...")
df["returns"]         = df["close"].pct_change()
df["price_range"]     = (df["high"] - df["low"]) / df["close"]
df["price_change"]    = (df["close"] - df["open"]) / df["open"]
df["volatility_5"]    = df["returns"].rolling(5).std()
df["volatility_20"]   = df["returns"].rolling(20).std()
df["volume_change"]   = df["volume"].pct_change()
df["volume_ma_ratio"] = df["volume"] / (df["volume"].rolling(20).mean() + 1e-9)
df["taker_ratio"]     = df["taker_base"] / (df["volume"] + 1e-9)
df["avg_trade_size"]  = df["volume"] / (df["num_trades"] + 1e-9)
df["trade_intensity"] = df["num_trades"] / (df["num_trades"].rolling(20).mean() + 1e-9)
df["high_low_ratio"]  = df["high"] / (df["low"] + 1e-9)
df["close_open_ratio"]= df["close"] / (df["open"] + 1e-9)

FEATURES = [
    "price_range","price_change","volatility_5","volatility_20",
    "volume_change","volume_ma_ratio","taker_ratio","avg_trade_size",
    "trade_intensity","high_low_ratio","close_open_ratio"
]
df = df.dropna(subset=FEATURES).reset_index(drop=True)
print(f"  ✓ {len(df)} usable candles")

# ── IMPROVED LABELING: Z-score anomaly detection ──────────────
print("Generating labels with Z-score anomaly detection...")

def zscore(s):
    return (s - s.mean()) / (s.std() + 1e-9)

z_vol    = zscore(df["volume_ma_ratio"]).abs()
z_price  = zscore(df["price_range"]).abs()
z_taker  = zscore((df["taker_ratio"] - 0.5).abs())
z_trades = zscore(df["trade_intensity"]).abs()

# Anomaly score = weighted sum of z-scores (continuous, not binary)
df["anomaly_score"] = (
    0.35 * z_vol +
    0.25 * z_price +
    0.25 * z_taker +
    0.15 * z_trades
)

# Label top 30% as manipulation (ensures good class balance)
threshold = df["anomaly_score"].quantile(0.70)
df["label"] = (df["anomaly_score"] > threshold).astype(int)

pos = df["label"].sum()
neg = len(df) - pos
print(f"  ✓ {pos} manipulation ({pos/len(df)*100:.1f}%), {neg} normal")

# ── Scale features ────────────────────────────────────────────
X = df[FEATURES].values.astype(np.float32)
y = df["label"].values.astype(np.float32)

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# ── Model: outputs raw logit (sigmoid applied at inference) ───
class OracleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(11, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze()

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model     = OracleNet().to(device)
pos_wt    = torch.tensor([neg / pos]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_wt)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)

X_tr = torch.tensor(X_train).to(device)
y_tr = torch.tensor(y_train).to(device)
X_vl = torch.tensor(X_val).to(device)
y_vl = torch.tensor(y_val).to(device)

# ── Train ─────────────────────────────────────────────────────
print(f"\nTraining 500 epochs on {device}...")
best_loss, best_state = float("inf"), None

for epoch in range(1, 501):
    model.train()
    optimizer.zero_grad()
    loss = criterion(model(X_tr), y_tr)
    loss.backward()
    optimizer.step()
    scheduler.step()

    if epoch % 100 == 0:
        model.eval()
        with torch.no_grad():
            vl = criterion(model(X_vl), y_vl).item()
        if vl < best_loss:
            best_loss  = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"  Epoch {epoch} | train={loss.item():.4f} | val={vl:.4f}")

if best_state:
    model.load_state_dict(best_state)

# ── Wrap with sigmoid for inference ──────────────────────────
class InferenceWrapper(nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x): return torch.sigmoid(self.m(x))

inf_model = InferenceWrapper(model)
inf_model.eval()

# ── Check variance ────────────────────────────────────────────
with torch.no_grad():
    preds = inf_model(X_vl).cpu().numpy()
scores = (preds * 10000).astype(int)
print(f"\n  Score range: {scores.min()} – {scores.max()}")
print(f"  Mean: {scores.mean():.0f}  Std: {scores.std():.0f}")
if scores.std() < 200:
    print("  ⚠ Low variance — model may still be biased. Try re-running.")
else:
    print("  ✓ Good variance — scores will change with market conditions")

# ── Save ──────────────────────────────────────────────────────
torch.save({"model_state_dict": inf_model.state_dict()}, "oracle_model.pth")
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# sklearn fallback
try:
    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(hidden_layer_sizes=(64,32,16), max_iter=500,
                        random_state=42, early_stopping=True)
    clf.fit(X_train, y_train.astype(int))
    with open("oracle_model_simple.pkl", "wb") as f:
        pickle.dump({"model": clf}, f)
    acc = clf.score(X_val, y_val.astype(int))
    print(f"  ✓ sklearn model saved (accuracy: {acc:.2%})")
except Exception as e:
    print(f"  sklearn skipped: {e}")

print("\n  ✓ oracle_model.pth saved")
print("  ✓ scaler.pkl saved")
print("\n  Now run: python auto_submit.py")
print("=" * 60)

