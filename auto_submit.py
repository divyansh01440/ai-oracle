"""
auto_submit.py — Fixed final version
Run with:  python auto_submit.py
Press Ctrl+C to stop.
"""

import os, sys, time, datetime, requests
import numpy as np

print("=" * 60)
print("  AI DOCAL — REAL-TIME SUBMITTER")
print("  Press Ctrl+C to stop")
print("=" * 60)

def install(pkg):
    os.system(f"pip install {pkg} --quiet --break-system-packages")

try:
    from web3 import Web3
except ImportError:
    install("web3"); from web3 import Web3

try:
    from dotenv import load_dotenv
except ImportError:
    install("python-dotenv"); from dotenv import load_dotenv

INTERVAL         = 15
CONTRACT_ADDRESS = "0x4957Bb834169De7721cC87622FB9cFf839cC6201"
RPC_URL          = "https://rpc-amoy.polygon.technology"
ASSET            = "MATIC/USDC"
SYMBOL           = "MATICUSDT"
BASE             = "https://api.binance.com/api/v3"

CONTRACT_ABI = [{
    "inputs": [
        {"internalType": "string",  "name": "asset", "type": "string"},
        {"internalType": "uint256", "name": "score", "type": "uint256"}
    ],
    "name": "submitRiskScore",
    "outputs": [],
    "stateMutability": "nonpayable",
    "type": "function"
}]

load_dotenv("contracts/.env")
PRIVATE_KEY = os.getenv("PRIVATE_KEY")
if not PRIVATE_KEY or PRIVATE_KEY == "your_wallet_private_key_here":
    print("  ERROR: PRIVATE_KEY not set in contracts/.env"); sys.exit(1)

w3 = Web3(Web3.HTTPProvider(RPC_URL))
if not w3.is_connected():
    print("  ERROR: Cannot connect to RPC"); sys.exit(1)

account  = w3.eth.account.from_key(PRIVATE_KEY)
contract = w3.eth.contract(
    address=Web3.to_checksum_address(CONTRACT_ADDRESS),
    abi=CONTRACT_ABI
)
bal = w3.from_wei(w3.eth.get_balance(account.address), "ether")
print(f"  ✓ Wallet : {account.address}")
print(f"  ✓ Balance: {bal:.4f} MATIC")
print(f"  ✓ Connected to Polygon Amoy\n")
if bal < 0.01:
    print("  ⚠ LOW MATIC! Get more first:")
    print("  https://faucet.polygon.technology/")
    print(f"  Wallet: {account.address}\n")

# ── Rolling history for proper normalization ──────────────────
history = {"buy_ratio": [], "spread": [], "vol_spike": [], "price_move": []}
HIST = 50

def normalize(key, val):
    history[key].append(float(val))
    if len(history[key]) > HIST:
        history[key].pop(0)
    arr = history[key]
    if len(arr) < 3:
        return 0.5
    mn, mx = min(arr), max(arr)
    if mx - mn < 1e-9:
        return 0.5
    return max(0.0, min(1.0, (val - mn) / (mx - mn)))

def compute_score():
    s = requests.Session()

    # ── 1. Recent trades: buy vs sell pressure ────────────────
    trades_data = s.get(f"{BASE}/trades",
                        params={"symbol": SYMBOL, "limit": 500},
                        timeout=10).json()
    if not isinstance(trades_data, list) or len(trades_data) == 0:
        raise ValueError("Invalid trades response")

    buy_qty  = sum(float(t["qty"]) for t in trades_data if not t["isBuyerMaker"])
    sell_qty = sum(float(t["qty"]) for t in trades_data if t["isBuyerMaker"])
    buy_ratio = buy_qty / (buy_qty + sell_qty + 1e-9)

    # ── 2. Order book spread ───────────────────────────────────
    book_data = s.get(f"{BASE}/depth",
                      params={"symbol": SYMBOL, "limit": 20},
                      timeout=10).json()
    asks = book_data.get("asks", [])
    bids = book_data.get("bids", [])

    if asks and bids and len(asks) > 0 and len(bids) > 0:
        best_ask   = float(asks[0][0])
        best_bid   = float(bids[0][0])
        mid        = (best_ask + best_bid) / 2.0
        spread_pct = (best_ask - best_bid) / (mid + 1e-9) * 100.0
    else:
        spread_pct = 0.001

    # ── 3. Volume spike from last 20 candles ──────────────────
    klines_data = s.get(f"{BASE}/klines",
                        params={"symbol": SYMBOL, "interval": "1m", "limit": 20},
                        timeout=10).json()
    if not isinstance(klines_data, list) or len(klines_data) < 5:
        raise ValueError("Not enough candle data")

    vols      = [float(k[5]) for k in klines_data]
    closes    = [float(k[4]) for k in klines_data]
    vol_mean  = np.mean(vols[:-1]) + 1e-9
    vol_spike = vols[-1] / vol_mean

    # ── 4. Price movement last 5 candles ──────────────────────
    price_move = abs(closes[-1] - closes[-5]) / (closes[-5] + 1e-9) * 100.0
    current_price = closes[-1]

    # ── Normalize all signals against rolling history ─────────
    n_buy    = normalize("buy_ratio",  buy_ratio)
    n_spread = normalize("spread",     spread_pct)
    n_vol    = normalize("vol_spike",  vol_spike)
    n_price  = normalize("price_move", price_move)

    # Buy imbalance: extremes (all buy OR all sell) = risky
    buy_imb = abs(n_buy - 0.5) * 2.0  # 0=balanced, 1=extreme

    composite = (
        0.35 * buy_imb  +
        0.25 * n_spread +
        0.25 * n_vol    +
        0.15 * n_price
    )
    score = int(composite * 10000)
    score = min(9999, max(1, score))

    parts = {
        "buy":  f"{buy_ratio*100:.1f}%",
        "spr":  f"{spread_pct:.4f}",
        "vol":  f"{vol_spike:.3f}x",
        "mv":   f"{price_move:.4f}%",
    }
    return score, current_price, parts

# ── Main loop ─────────────────────────────────────────────────
run_count  = 0
prev_score = None

print(f"  Scores stabilize after first 5 readings (building history...)")
print(f"{'─'*60}\n")

while True:
    run_count += 1
    now = datetime.datetime.now().strftime("%H:%M:%S")

    # Check balance every 10 runs
    if run_count % 10 == 1:
        bal = w3.from_wei(w3.eth.get_balance(account.address), "ether")
        if bal < 0.005:
            print(f"\n  ✕ OUT OF MATIC ({bal:.4f})! Stopping.")
            print(f"  Get more at: https://faucet.polygon.technology/")
            print(f"  Wallet: {account.address}")
            sys.exit(1)

    try:
        score, price, parts = compute_score()
        safe   = score < 7000
        status = "✓ SAFE" if safe else "✕ RISK"

        change = ""
        if prev_score is not None:
            diff  = score - prev_score
            arrow = "↑" if diff > 0 else ("↓" if diff < 0 else "=")
            change = f"({arrow}{abs(diff)})" if diff != 0 else "(=)"
        prev_score = score

        nonce   = w3.eth.get_transaction_count(account.address)
        tx      = contract.functions.submitRiskScore(ASSET, score).build_transaction({
            "from": account.address, "nonce": nonce,
            "gas": 200000, "gasPrice": w3.to_wei("35", "gwei"), "chainId": 80002
        })
        signed  = account.sign_transaction(tx)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)

        print(
            f"  [{now}] #{run_count:04d} | "
            f"Score: {score:>5} {change:<10}| {status} | "
            f"${price:.4f} | "
            f"buy={parts['buy']} spr={parts['spr']} "
            f"vol={parts['vol']} mv={parts['mv']}"
        )

    except requests.exceptions.RequestException as e:
        print(f"  [{now}] ⚠ API error: {e}")
    except Exception as e:
        err = str(e)
        if "nonce too low" in err.lower() or "already known" in err.lower():
            print(f"  [{now}] ⚠ TX pending — skipping")
        elif "insufficient funds" in err.lower():
            print(f"  [{now}] ✕ Out of MATIC! https://faucet.polygon.technology/")
            time.sleep(60)
        else:
            print(f"  [{now}] ⚠ {err[:120]}")

    time.sleep(INTERVAL)