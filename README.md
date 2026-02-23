# 🔮 AI Docal — Verifiable AI Oracle on Polygon

![Polygon](https://img.shields.io/badge/Polygon-Amoy_Testnet-8247e5?style=for-the-badge&logo=polygon)
![PyTorch](https://img.shields.io/badge/PyTorch-Neural_Network-ee4c2c?style=for-the-badge&logo=pytorch)
![Solidity](https://img.shields.io/badge/Solidity-Smart_Contract-363636?style=for-the-badge&logo=solidity)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-Auto_Submit-2088ff?style=for-the-badge&logo=githubactions)
![License](https://img.shields.io/badge/License-MIT-00dcff?style=for-the-badge)

---

## 📌 What Is This?

DeFi protocols lose billions every year to price manipulation — spoofing, wash trading, and flash loan attacks that exploit the fact that on-chain systems execute purely on price, with zero awareness of whether that price is genuine. Existing oracles like Chainlink and Pyth provide accurate price feeds but do not assess market integrity.

**AI Docal** solves this by introducing a second-layer verifiable AI oracle. A PyTorch neural network continuously analyzes 11 market microstructure features — volume anomalies, taker imbalance, trade intensity, and price volatility — to produce a manipulation risk score between 0 and 10,000. That score is written directly to a Solidity smart contract on Polygon every 15 minutes via GitHub Actions, creating a tamper-evident, permanently auditable intelligence layer that any DeFi protocol can read for free with a single `getRiskScore()` call.

---

## ⚙️ How It Works

- **🧠 AI Inference** — A PyTorch neural network (11→32→16→1) trained on 1,000+ MATIC/USDT candlesticks analyzes 11 microstructure features and outputs a manipulation probability in under 1ms
- **⛓️ On-Chain Write** — `submit_score.py` scales the probability to 0–10,000 and calls `submitRiskScore()` on the deployed Solidity contract via Web3.py — creating an immutable on-chain record
- **🤖 Auto-Submission** — GitHub Actions runs `submit_score.py` every 15 minutes, 24/7, completely free — no server required, no manual intervention
- **📊 Live Dashboard** — `dashboard.html` reads the score directly from the blockchain using ethers.js every 15 seconds — no backend, fully client-side, open to anyone with MetaMask

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| AI Model | Python · PyTorch · scikit-learn |
| Smart Contract | Solidity 0.8.20 · Hardhat |
| Blockchain | Polygon Amoy (ChainID: 80002) |
| Web3 Integration | Web3.py · ethers.js |
| API | Flask · Python |
| Automation | GitHub Actions |
| Frontend | HTML · CSS · JavaScript |
| Hosting | Vercel |

---

## 🌐 Live Demo

> 
>
> **Landing Page:** [https://ai-oracle-dun.vercel.app/](https://ai-oracle-dun.vercel.app/)
>
> **Whitepaper:** [https://ai-oracle-dun.vercel.app/whitepaper.html](https://ai-oracle-dun.vercel.app/whitepaper.html)

*Connect your MetaMask wallet on Polygon Amoy to access the live dashboard.*

---

## 📄 Smart Contract

| Field | Value |
|-------|-------|
| **Network** | Polygon Amoy Testnet |
| **Address** | `0x4957Bb834169De7721cC87622FB9cFf839cC6201` |
| **Explorer** | [View on PolygonScan ↗](https://amoy.polygonscan.com/address/0x4957Bb834169De7721cC87622FB9cFf839cC6201) |
| **Safe Threshold** | Score < 7,000 = SAFE |
| **Score Range** | 0 – 10,000 |

### Public Read Functions (zero gas)
```solidity
getRiskScore("MATIC/USDC")  → uint256   // 0-10,000
isSafe("MATIC/USDC")        → bool      // score < 7000
getAssetInfo("MATIC/USDC")  → (score, safe, updatedAt)
```

### Integrate in your DeFi protocol
```solidity
interface IAIDocal {
    function isSafe(string memory asset) external view returns (bool);
}

modifier safeMarket() {
    require(
        IAIDocal(0x4957Bb834169De7721cC87622FB9cFf839cC6201).isSafe("MATIC/USDC"),
        "AI Docal: manipulation detected"
    );
    _;
}
```

---

## 🚀 Setup Instructions

### Prerequisites
- Python 3.10+
- Node.js 18+
- MetaMask wallet with Polygon Amoy MATIC ([get free MATIC](https://faucet.polygon.technology/))

### 1. Clone the repository
```bash
git clone https://github.com/divyansh01440/ai-oracle.git
cd ai-oracle
```

### 2. Install Python dependencies
```bash
pip install torch scikit-learn web3 flask requests numpy python-dotenv pandas
```

### 3. Install Hardhat dependencies
```bash
cd contracts
npm install
cd ..
```

### 4. Configure environment
```bash
# Create contracts/.env
echo "PRIVATE_KEY=your_metamask_private_key_here" > contracts/.env
```

### 5. Train the AI model
```bash
python data_pipeline.py   # fetch & label training data
python train.py           # train PyTorch model → saves oracle_model.pth
```

### 6. Submit a score manually
```bash
python submit_score.py
```

### 7. Run the Flask API
```bash
python api.py
# API available at http://localhost:5000/score
```

### 8. Open the dashboard
Open `dashboard.html` in your browser — connect MetaMask to Polygon Amoy.

---

## 🤖 GitHub Actions (Auto-Submission)

Scores are automatically submitted every 15 minutes via GitHub Actions — no server needed.

**Setup:**
1. Push this repo to GitHub
2. Go to **Settings → Secrets → Actions → New secret**
3. Add `PRIVATE_KEY` = your MetaMask private key
4. Go to **Actions tab** → enable workflows
5. Click **Run workflow** to test manually

The workflow file is at `.github/workflows/submit-score.yml`.

---

## 📁 Project Structure

```
ai-oracle/
├── .github/
│   └── workflows/
│       └── submit-score.yml   # GitHub Actions — runs every 15 min
├── contracts/
│   ├── contracts/
│   │   └── OracleContract.sol # Solidity smart contract
│   ├── scripts/
│   │   └── deploy.js          # Hardhat deployment script
│   └── hardhat.config.js
├── dashboard.html             # Live blockchain dashboard
├── index.html                 # Landing page
├── whitepaper.html            # Technical whitepaper
├── data_pipeline.py           # Fetch & label Binance data
├── train.py                   # Train PyTorch model
├── submit_score.py            # Submit score to blockchain
├── api.py                     # Flask REST API
└── auto_submit.py             # Local continuous submitter
```

---

## 🗺️ Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| **Phase 1** | ✅ Complete | Data pipeline · PyTorch model · Polygon Amoy deployment · Dashboard · Flask API · GitHub Actions |
| **Phase 2** | 🔄 In Progress | Multi-asset support (BTC, ETH, SOL) · Automated retraining pipeline · Improved labeling |
| **Phase 3** | 📋 Planned | Polygon mainnet · First production DeFi integration · Decentralized validation |
| **Phase 4 (v2)** | 🔮 Future | **ZK Proof integration** — cryptographic proof that the risk score was computed by the exact trained model without revealing model weights · Multi-chain (Arbitrum, Base) · Governance DAO |

---

## 👤 Author

**Divyansh Gautam** — Nirvanatechon

[![GitHub](https://img.shields.io/badge/GitHub-divyansh01440-181717?style=flat&logo=github)](https://github.com/divyansh01440)
[![X](https://img.shields.io/badge/X-@G55269Gautam-000000?style=flat&logo=x)](https://x.com/G55269Gautam)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-divyansh--gautam-0077b5?style=flat&logo=linkedin)](https://www.linkedin.com/in/divyansh-gautam-286754320)

---

## 📜 License

MIT License — free to use, modify, and build on.

---

<p align="center">Built on Polygon · Powered by PyTorch · Automated by GitHub Actions</p>