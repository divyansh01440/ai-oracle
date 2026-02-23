"""
data_pipeline.py
Fetches last 500 MATIC/USDT 1-minute candles from Binance public API,
computes 11 manipulation-detection features, labels them, and saves to training_data.csv.
No API key required.
"""

import requests
import pandas as pd
import numpy as np
import sys

# ── 1. Fetch raw candle data ─────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Fetching 500 MATIC/USDT 1-min candles from Binance...")
print("=" * 60)

URL = "https://api.binance.com/api/v3/klines"
params = {
    "symbol": "MATICUSDT",
    "interval": "1m",
    "limit": 500,
}

try:
    response = requests.get(URL, params=params, timeout=15)
    response.raise_for_status()
    raw = response.json()
    print(f"  ✓ Received {len(raw)} candles from Binance API")
except Exception as e:
    print(f"  ✗ Failed to fetch data: {e}")
    sys.exit(1)

# ── 2. Parse into DataFrame ──────────────────────────────────────────────────
print("\nSTEP 2: Parsing candle data into DataFrame...")

columns = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "num_trades",
    "taker_buy_base_vol", "taker_buy_quote_vol", "ignore"
]

df = pd.DataFrame(raw, columns=columns)

# Convert types
numeric_cols = ["open", "high", "low", "close", "volume",
                "quote_volume", "taker_buy_base_vol", "taker_buy_quote_vol"]
df[numeric_cols] = df[numeric_cols].astype(float)
df["num_trades"] = df["num_trades"].astype(int)
df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
df.set_index("open_time", inplace=True)

print(f"  ✓ DataFrame shape: {df.shape}")
print(f"  ✓ Time range: {df.index[0]}  →  {df.index[-1]}")

# ── 3. Compute 11 features ───────────────────────────────────────────────────
print("\nSTEP 3: Computing 11 manipulation-detection features...")

feat = pd.DataFrame(index=df.index)

# 1. Price range  (high - low) / close
feat["price_range"] = (df["high"] - df["low"]) / df["close"]
print("  ✓ Feature 1/11: price_range")

# 2. Price change  (close - open) / open
feat["price_change"] = (df["close"] - df["open"]) / df["open"]
print("  ✓ Feature 2/11: price_change")

# 3. Volatility-5  rolling 5-bar std of returns
returns = df["close"].pct_change()
feat["volatility_5"] = returns.rolling(5).std()
print("  ✓ Feature 3/11: volatility_5")

# 4. Volatility-20  rolling 20-bar std of returns
feat["volatility_20"] = returns.rolling(20).std()
print("  ✓ Feature 4/11: volatility_20")

# 5. Volume change  pct change in volume
feat["volume_change"] = df["volume"].pct_change()
print("  ✓ Feature 5/11: volume_change")

# 6. Volume MA ratio  volume / 20-bar moving average of volume
vol_ma20 = df["volume"].rolling(20).mean()
feat["volume_ma_ratio"] = df["volume"] / (vol_ma20 + 1e-9)
print("  ✓ Feature 6/11: volume_ma_ratio")

# 7. Taker ratio  taker buy base vol / total volume
feat["taker_ratio"] = df["taker_buy_base_vol"] / (df["volume"] + 1e-9)
print("  ✓ Feature 7/11: taker_ratio")

# 8. Avg trade size  volume / number of trades
feat["avg_trade_size"] = df["volume"] / (df["num_trades"] + 1e-9)
print("  ✓ Feature 8/11: avg_trade_size")

# 9. Trade intensity  num_trades / rolling 20-bar mean of num_trades
trades_ma20 = df["num_trades"].rolling(20).mean()
feat["trade_intensity"] = df["num_trades"] / (trades_ma20 + 1e-9)
print("  ✓ Feature 9/11: trade_intensity")

# 10. High-low ratio  high / low
feat["high_low_ratio"] = df["high"] / (df["low"] + 1e-9)
print("  ✓ Feature 10/11: high_low_ratio")

# 11. Close-open ratio  close / open
feat["close_open_ratio"] = df["close"] / (df["open"] + 1e-9)
print("  ✓ Feature 11/11: close_open_ratio")

# ── 4. Create manipulation risk labels ──────────────────────────────────────
print("\nSTEP 4: Creating manipulation risk labels...")

"""
Label logic (heuristic — not financial advice):
A candle is flagged as HIGH RISK (1) if it simultaneously shows:
  - abnormal volume  (volume_ma_ratio > 2.5)
  - large price move  (|price_change| > 0.3 %)
  - high taker imbalance  (taker_ratio < 0.25 or > 0.75)
Otherwise label = 0 (normal).
"""

cond_volume   = feat["volume_ma_ratio"] > 2.5
cond_price    = feat["price_change"].abs() > 0.003
cond_taker    = (feat["taker_ratio"] < 0.25) | (feat["taker_ratio"] > 0.75)

feat["manipulation_risk"] = (cond_volume & cond_price & cond_taker).astype(int)

n_flagged = feat["manipulation_risk"].sum()
pct_flagged = 100 * n_flagged / len(feat)
print(f"  ✓ Labels created  →  {n_flagged} high-risk candles ({pct_flagged:.1f}% of data)")

# ── 5. Drop NaN rows (from rolling windows) ──────────────────────────────────
print("\nSTEP 5: Cleaning data (dropping NaN rows from rolling windows)...")
before = len(feat)
feat.dropna(inplace=True)
after = len(feat)
print(f"  ✓ Dropped {before - after} rows with NaN  →  {after} clean rows remain")

# ── 6. Save to CSV ───────────────────────────────────────────────────────────
print("\nSTEP 6: Saving to training_data.csv...")
feat.to_csv("training_data.csv")
print(f"  ✓ Saved {after} rows × {feat.shape[1]} columns to training_data.csv")

# ── 7. Summary ───────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PIPELINE COMPLETE")
print("=" * 60)
print(feat.describe().to_string())
print("\n✓ Ready to run:  python train.py")
