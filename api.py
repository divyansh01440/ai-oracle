"""
api.py
Verifiable AI Oracle — REST API Server
Loads the trained PyTorch model, fetches live MATIC/USDT data from Binance,
computes features, and exposes risk scores via HTTP endpoints.

Run with:  python api.py
"""

import os
import sys
import pickle
import datetime
import requests
import numpy as np

print("=" * 60)
print("  VERIFIABLE AI ORACLE — API SERVER")
print("=" * 60)

# ── Install missing packages automatically ───────────────────
def install(pkg):
    print(f"  Installing {pkg}...")
    os.system(f"pip install {pkg} --quiet")

try:
    from flask import Flask, jsonify
    print("  ✓ Flask ready")
except ImportError:
    install("flask")
    from flask import Flask, jsonify

try:
    from flask_cors import CORS
    print("  ✓ Flask-CORS ready")
except ImportError:
    install("flask-cors")
    from flask_cors import CORS

try:
    import torch
    import torch.nn as nn
    print(f"  ✓ PyTorch ready (version {torch.__version__})")
except ImportError:
    install("torch")
    import torch
    import torch.nn as nn

try:
    from sklearn.preprocessing import StandardScaler
    print("  ✓ scikit-learn ready")
except ImportError:
    install("scikit-learn")
    from sklearn.preprocessing import StandardScaler

# ── Feature column names (must match training) ───────────────
FEATURE_COLS = [
    "price_range", "price_change", "volatility_5", "volatility_20",
    "volume_change", "volume_ma_ratio", "taker_ratio", "avg_trade_size",
    "trade_intensity", "high_low_ratio", "close_open_ratio"
]

SAFE_THRESHOLD = 70  # out of 100

# ── Neural network architecture (must match train.py) ─────────
class OracleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(11, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# ── Load scaler ───────────────────────────────────────────────
print("\nLoading scaler from scaler.pkl...")
if not os.path.exists("scaler.pkl"):
    print("  ERROR: scaler.pkl not found. Run train.py first.")
    sys.exit(1)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
print("  ✓ Scaler loaded")

# ── Load PyTorch model ────────────────────────────────────────
print("Loading model from oracle_model.pth...")
if not os.path.exists("oracle_model.pth"):
    print("  ERROR: oracle_model.pth not found. Run train.py first.")
    sys.exit(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = OracleNet().to(device)

checkpoint = torch.load("oracle_model.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print(f"  ✓ Model loaded (device: {device})")

# ── Try loading sklearn model as fallback ─────────────────────
sklearn_model = None
if os.path.exists("oracle_model_simple.pkl"):
    with open("oracle_model_simple.pkl", "rb") as f:
        bundle = pickle.load(f)
    sklearn_model = bundle.get("model")
    print("  ✓ sklearn fallback model loaded")
else:
    print("  ! oracle_model_simple.pkl not found (PyTorch model will be used)")

# ── Binance API ───────────────────────────────────────────────
BINANCE_URL = "https://api.binance.com/api/v3/klines"

def fetch_recent_candles(symbol="MATICUSDT", interval="1m", limit=25):
    """Fetch the last N candles from Binance (no API key needed)."""
    print(f"  Fetching {limit} candles for {symbol} from Binance...")
    response = requests.get(BINANCE_URL, params={
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }, timeout=10)
    response.raise_for_status()
    raw = response.json()
    print(f"  ✓ Got {len(raw)} candles")
    return raw

def parse_candles(raw):
    """Parse raw Binance kline data into float arrays."""
    candles = []
    for c in raw:
        candles.append({
            "open":                float(c[1]),
            "high":                float(c[2]),
            "low":                 float(c[3]),
            "close":               float(c[4]),
            "volume":              float(c[5]),
            "num_trades":          int(c[8]),
            "taker_buy_base_vol":  float(c[9]),
        })
    return candles

def compute_features(candles):
    """
    Compute all 11 features from a list of candles.
    Uses the LAST candle as the target, with prior candles for rolling stats.
    Requires at least 21 candles for volatility_20.
    """
    if len(candles) < 21:
        raise ValueError(f"Need at least 21 candles, got {len(candles)}")

    closes  = np.array([c["close"]  for c in candles], dtype=float)
    volumes = np.array([c["volume"] for c in candles], dtype=float)
    trades  = np.array([c["num_trades"] for c in candles], dtype=float)
    taker   = np.array([c["taker_buy_base_vol"] for c in candles], dtype=float)

    returns = np.diff(closes) / closes[:-1]  # length N-1

    # Target is the LAST candle (index -1)
    c = candles[-1]

    # 1. Price range
    price_range = (c["high"] - c["low"]) / c["close"]

    # 2. Price change
    price_change = (c["close"] - c["open"]) / c["open"]

    # 3. Volatility 5 (std of last 5 returns)
    volatility_5 = float(np.std(returns[-5:]))

    # 4. Volatility 20 (std of last 20 returns)
    volatility_20 = float(np.std(returns[-20:]))

    # 5. Volume change
    volume_change = (volumes[-1] - volumes[-2]) / (volumes[-2] + 1e-9)

    # 6. Volume MA ratio (volume / 20-bar MA)
    vol_ma20 = float(np.mean(volumes[-20:]))
    volume_ma_ratio = volumes[-1] / (vol_ma20 + 1e-9)

    # 7. Taker ratio
    taker_ratio = taker[-1] / (volumes[-1] + 1e-9)

    # 8. Avg trade size
    avg_trade_size = volumes[-1] / (trades[-1] + 1e-9)

    # 9. Trade intensity
    trades_ma20 = float(np.mean(trades[-20:]))
    trade_intensity = trades[-1] / (trades_ma20 + 1e-9)

    # 10. High-low ratio
    high_low_ratio = c["high"] / (c["low"] + 1e-9)

    # 11. Close-open ratio
    close_open_ratio = c["close"] / (c["open"] + 1e-9)

    features = np.array([[
        price_range, price_change, volatility_5, volatility_20,
        volume_change, volume_ma_ratio, taker_ratio, avg_trade_size,
        trade_intensity, high_low_ratio, close_open_ratio
    ]], dtype=np.float32)

    print(f"  Features computed: {features.tolist()}")
    return features

def run_inference(features_raw):
    """Scale features and run through PyTorch model. Returns probability 0–1."""
    features_scaled = scaler.transform(features_raw).astype(np.float32)
    tensor = torch.tensor(features_scaled).to(device)

    model.eval()
    with torch.no_grad():
        prob = model(tensor).cpu().numpy().flatten()[0]

    print(f"  PyTorch model output (probability): {prob:.6f}")
    return float(prob)

# ── Flask app ─────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)  # Allow dashboard.html (any origin) to call this API

print("\n  ✓ Flask app created with CORS enabled")

# ─────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "model":  "oracle_model.pth",
        "device": str(device),
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
    })

# ─────────────────────────────────────────────────────────────
@app.route("/score", methods=["GET"])
def score():
    """
    Main endpoint.
    1. Fetches latest MATIC/USDT candles from Binance
    2. Computes 11 features
    3. Runs PyTorch model
    4. Returns JSON with risk_score (0–100), is_safe, asset, timestamp
    """
    print("\n" + "-" * 50)
    print(f"  GET /score called at {datetime.datetime.utcnow().isoformat()}Z")

    try:
        # Step 1: Fetch candles
        raw     = fetch_recent_candles(symbol="MATICUSDT", interval="1m", limit=25)
        candles = parse_candles(raw)

        # Step 2: Compute features
        features = compute_features(candles)

        # Step 3: Run model
        prob       = run_inference(features)
        risk_score = round(prob * 100, 2)   # 0.0 – 100.0
        is_safe    = risk_score < SAFE_THRESHOLD

        # Step 4: Build response
        latest   = candles[-1]
        response = {
            "asset":      "MATIC/USDC",
            "risk_score": risk_score,
            "is_safe":    is_safe,
            "timestamp":  datetime.datetime.utcnow().isoformat() + "Z",
            "details": {
                "raw_probability":  prob,
                "safe_threshold":   SAFE_THRESHOLD,
                "latest_price":     latest["close"],
                "latest_volume":    latest["volume"],
            }
        }

        print(f"  ✓ risk_score={risk_score}  is_safe={is_safe}")
        return jsonify(response), 200

    except requests.exceptions.RequestException as e:
        print(f"  ERROR fetching from Binance: {e}")
        return jsonify({"error": "Failed to fetch market data", "detail": str(e)}), 503

    except Exception as e:
        print(f"  ERROR: {e}")
        return jsonify({"error": str(e)}), 500

# ─────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    """Root endpoint — shows available routes."""
    return jsonify({
        "name":      "Verifiable AI Oracle API",
        "version":   "1.0.0",
        "endpoints": {
            "GET /score":  "Returns live AI risk score for MATIC/USDC",
            "GET /health": "Returns server health status",
        },
        "contract": "0x4957Bb834169De7721cC87622FB9cFf839cC6201",
        "network":  "Polygon Amoy (chainId: 80002)"
    })

# ── Start server ──────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  STARTING SERVER")
    print("=" * 60)
    print("  Endpoints:")
    print("  http://localhost:5000/score   <- risk score")
    print("  http://localhost:5000/health  <- health check")
    print("  http://localhost:5000/        <- API info")
    print("\n  Press Ctrl+C to stop\n")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,   # Set True if you want auto-reload during development
    )