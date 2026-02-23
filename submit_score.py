import os
import sys
import requests
import numpy as np

print("=" * 55)
print("  AI DOCAL - SCORE SUBMISSION")
print("=" * 55)

# Step 1: Check private key
print("\n[1/6] Checking PRIVATE_KEY...")
PRIVATE_KEY = os.environ.get("PRIVATE_KEY", "")
if not PRIVATE_KEY:
    print("  ERROR: PRIVATE_KEY environment variable is empty!")
    print("  Fix: Go to GitHub repo -> Settings -> Secrets -> Actions")
    print("  Add secret named PRIVATE_KEY with your wallet private key")
    sys.exit(1)
if len(PRIVATE_KEY) < 60:
    print(f"  ERROR: PRIVATE_KEY looks wrong (length: {len(PRIVATE_KEY)})")
    print("  It should be a 64-character hex string")
    sys.exit(1)
print(f"  OK: Key found (length: {len(PRIVATE_KEY)})")

# Step 2: Install web3
print("\n[2/6] Installing web3...")
ret = os.system("pip install web3 requests numpy --quiet")
if ret != 0:
    print("  ERROR: pip install failed")
    sys.exit(1)
print("  OK: Dependencies installed")

from web3 import Web3

# Step 3: Connect to blockchain
print("\n[3/6] Connecting to Polygon Amoy...")
RPC_URL = "https://rpc-amoy.polygon.technology"
w3 = Web3(Web3.HTTPProvider(RPC_URL))
if not w3.is_connected():
    print("  ERROR: Cannot connect to Polygon Amoy RPC")
    print("  The RPC might be temporarily down. Try re-running.")
    sys.exit(1)
print("  OK: Connected to Polygon Amoy")

CONTRACT_ADDRESS = "0x4957Bb834169De7721cC87622FB9cFf839cC6201"
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

# Step 4: Check wallet balance
print("\n[4/6] Checking wallet balance...")
try:
    account = w3.eth.account.from_key(PRIVATE_KEY)
    bal_wei  = w3.eth.get_balance(account.address)
    bal      = w3.from_wei(bal_wei, "ether")
    print(f"  OK: Wallet {account.address[:10]}...{account.address[-6:]}")
    print(f"  OK: Balance = {bal:.6f} MATIC")
    if bal < 0.003:
        print(f"  ERROR: Not enough MATIC for gas (need at least 0.003)")
        print(f"  Get free testnet MATIC at: https://faucet.polygon.technology/")
        print(f"  Your wallet: {account.address}")
        sys.exit(1)
except Exception as e:
    print(f"  ERROR: Could not load wallet: {e}")
    print("  Check your PRIVATE_KEY secret is correct")
    sys.exit(1)

# Step 5: Fetch market data and compute score
print("\n[5/6] Fetching market data from Binance...")
try:
    s = requests.Session()

    trades_r = s.get(f"{BASE}/trades",
                     params={"symbol": SYMBOL, "limit": 500}, timeout=10)
    trades_r.raise_for_status()
    trades = trades_r.json()
    print(f"  OK: Got {len(trades)} recent trades")

    book_r = s.get(f"{BASE}/depth",
                   params={"symbol": SYMBOL, "limit": 20}, timeout=10)
    book_r.raise_for_status()
    book = book_r.json()
    asks = book.get("asks", [])
    bids = book.get("bids", [])
    print(f"  OK: Order book loaded ({len(asks)} asks, {len(bids)} bids)")

    klines_r = s.get(f"{BASE}/klines",
                     params={"symbol": SYMBOL, "interval": "1m", "limit": 20},
                     timeout=10)
    klines_r.raise_for_status()
    klines = klines_r.json()
    print(f"  OK: Got {len(klines)} candles")

except Exception as e:
    print(f"  ERROR fetching market data: {e}")
    sys.exit(1)

# Compute score
buy_qty   = sum(float(t["qty"]) for t in trades if not t["isBuyerMaker"])
sell_qty  = sum(float(t["qty"]) for t in trades if t["isBuyerMaker"])
buy_ratio = buy_qty / (buy_qty + sell_qty + 1e-9)
buy_imb   = abs(buy_ratio - 0.5) * 2

if asks and bids:
    best_ask   = float(asks[0][0])
    best_bid   = float(bids[0][0])
    mid        = (best_ask + best_bid) / 2.0
    spread_bps = (best_ask - best_bid) / (mid + 1e-9) * 10000
else:
    spread_bps = 1.0

vols       = [float(k[5]) for k in klines]
closes     = [float(k[4]) for k in klines]
vol_spike  = vols[-1] / (np.mean(vols[:-1]) + 1e-9)
price_move = abs(closes[-1] - closes[-5]) / (closes[-5] + 1e-9) * 100

n_pressure = min(1.0, buy_imb / 0.4)
n_spread   = min(1.0, spread_bps / 5.0)
n_volume   = min(1.0, max(0.0, (vol_spike - 1.0) / 3.0))
n_price    = min(1.0, price_move / 0.5)

composite  = 0.35*n_pressure + 0.25*n_spread + 0.25*n_volume + 0.15*n_price
score      = int(composite * 10000)
score      = min(9999, max(1, score))
is_safe    = score < 7000

print(f"\n  Score   : {score} / 10000")
print(f"  Status  : {'SAFE' if is_safe else 'HIGH RISK'}")
print(f"  Price   : ${closes[-1]:.4f}")
print(f"  Buy%    : {buy_ratio*100:.1f}%")

# Step 6: Submit to blockchain
print("\n[6/6] Submitting to blockchain...")
try:
    contract = w3.eth.contract(
        address=Web3.to_checksum_address(CONTRACT_ADDRESS),
        abi=CONTRACT_ABI
    )
    nonce = w3.eth.get_transaction_count(account.address)
    tx    = contract.functions.submitRiskScore(ASSET, score).build_transaction({
        "from":     account.address,
        "nonce":    nonce,
        "gas":      200000,
        "gasPrice": w3.to_wei("35", "gwei"),
        "chainId":  80002
    })
    signed  = account.sign_transaction(tx)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    print(f"  OK: TX sent: {tx_hash.hex()[:20]}...")
    print(f"  Waiting for confirmation (up to 60s)...")
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)
    if receipt.status == 1:
        print(f"\n  CONFIRMED on block #{receipt.blockNumber}")
        print(f"  Score {score} written to Polygon Amoy!")
        print(f"  Explorer: https://amoy.polygonscan.com/tx/{tx_hash.hex()}")
    else:
        print("  ERROR: Transaction reverted on chain")
        sys.exit(1)
except Exception as e:
    print(f"  ERROR submitting transaction: {e}")
    sys.exit(1)

print("\n" + "=" * 55)
print("  SUCCESS - Dashboard will update in 15 seconds")
print("=" * 55)
