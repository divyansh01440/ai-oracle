"""
submit_score.py — GitHub Actions version
Reads PRIVATE_KEY from environment variable (GitHub Secret).
"""

import os, sys, requests, datetime
import numpy as np

print("=" * 55)
print("  AI DOCAL — SCORE SUBMISSION")
print("=" * 55)

def install(pkg):
    os.system(f"pip install {pkg} --quiet")

try:
    from web3 import Web3
except ImportError:
    install("web3"); from web3 import Web3

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

PRIVATE_KEY = os.environ.get("PRIVATE_KEY", "").strip()
if not PRIVATE_KEY:
    try:
        from dotenv import load_dotenv
        load_dotenv("contracts/.env")
        PRIVATE_KEY = os.getenv("PRIVATE_KEY", "").strip()
    except:
        pass

if not PRIVATE_KEY:
    print("ERROR: PRIVATE_KEY not set.")
    sys.exit(1)

w3 = Web3(Web3.HTTPProvider(RPC_URL))
if not w3.is_connected():
    print("ERROR: Cannot connect to RPC"); sys.exit(1)

account  = w3.eth.account.from_key(PRIVATE_KEY)
contract = w3.eth.contract(
    address=Web3.to_checksum_address(CONTRACT_ADDRESS),
    abi=CONTRACT_ABI
)

bal = w3.from_wei(w3.eth.get_balance(account.address), "ether")
print(f"  Wallet  : {account.address}")
print(f"  Balance : {bal:.4f} MATIC")

if bal < 0.002:
    print("ERROR: Not enough MATIC. Get free testnet: https://faucet.polygon.technology/")
    sys.exit(1)

print("\nFetching live market data...")
s = requests.Session()

trades_data = s.get(f"{BASE}/trades", params={"symbol": SYMBOL, "limit": 500}, timeout=10).json()
buy_qty  = sum(float(t["qty"]) for t in trades_data if not t["isBuyerMaker"])
sell_qty = sum(float(t["qty"]) for t in trades_data if t["isBuyerMaker"])
buy_ratio = buy_qty / (buy_qty + sell_qty + 1e-9)

book_data = s.get(f"{BASE}/depth", params={"symbol": SYMBOL, "limit": 20}, timeout=10).json()
asks = book_data.get("asks", [])
bids = book_data.get("bids", [])
spread_pct = (float(asks[0][0]) - float(bids[0][0])) / (float(asks[0][0]) + 1e-9) * 100 if asks and bids else 0.001

klines = s.get(f"{BASE}/klines", params={"symbol": SYMBOL, "interval": "1m", "limit": 60}, timeout=10).json()
vols   = [float(k[5]) for k in klines]
closes = [float(k[4]) for k in klines]
price  = closes[-1]

vol_spike  = vols[-1] / (np.mean(vols[:-1]) + 1e-9)
returns    = np.abs(np.diff(closes) / (np.array(closes[:-1]) + 1e-9))
vola_ratio = float(np.std(returns[-5:])) / (float(np.std(returns[-20:])) + 1e-9)
price_move = abs(closes[-1] - closes[-10]) / (closes[-10] + 1e-9) * 100.0

print(f"  Price      : ${price:.4f}")
print(f"  Buy ratio  : {buy_ratio*100:.1f}%")
print(f"  Vol spike  : {vol_spike:.3f}x")
print(f"  Vola ratio : {vola_ratio:.3f}")
print(f"  Price move : {price_move:.4f}%")

buy_imb      = abs(buy_ratio - 0.5) * 2.0
spread_score = min(1.0, spread_pct / 0.05)
vol_score    = min(1.0, max(0.0, (vol_spike - 0.8) / 1.2))
vola_score   = min(1.0, max(0.0, (vola_ratio - 0.8) / 2.0))
move_score   = min(1.0, price_move / 0.5)

composite = (
    0.30 * buy_imb +
    0.20 * spread_score +
    0.20 * vol_score +
    0.15 * vola_score +
    0.15 * move_score
)
minute     = datetime.datetime.utcnow().minute
time_nudge = (minute % 11) / 150.0
composite  = min(1.0, composite + time_nudge)

score   = int(composite * 10000)
score   = min(9999, max(10, score))
is_safe = score < 7000

print(f"\n{'='*55}")
print(f"  SCORE  : {score} / 10000  ({score/100:.1f}%)")
print(f"  STATUS : {'SAFE' if is_safe else 'HIGH RISK'}")
print(f"{'='*55}")

print("\nSubmitting to Polygon Amoy...")
nonce  = w3.eth.get_transaction_count(account.address)
tx     = contract.functions.submitRiskScore(ASSET, score).build_transaction({
    "from": account.address, "nonce": nonce,
    "gas": 200000, "gasPrice": w3.to_wei("35", "gwei"), "chainId": 80002
})
signed  = account.sign_transaction(tx)
tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
tx_hex  = tx_hash.hex()
print(f"  TX      : {tx_hex}")
print(f"  Explorer: https://amoy.polygonscan.com/tx/{tx_hex}")

receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)
if receipt.status == 1:
    print(f"\n  Confirmed on block #{receipt.blockNumber}")
    print(f"  Score {score} is now live!")
else:
    print("  Transaction failed."); sys.exit(1)