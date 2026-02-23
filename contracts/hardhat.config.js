require("@nomicfoundation/hardhat-toolbox");
require("dotenv").config();

// ── Validate environment variables ──────────────────────────────────────────
const PRIVATE_KEY = process.env.PRIVATE_KEY;
const POLYGONSCAN_API_KEY = process.env.POLYGONSCAN_API_KEY || "";

if (!PRIVATE_KEY) {
  console.warn(
    "\n⚠  WARNING: PRIVATE_KEY not found in .env file.\n" +
    "   Create a .env file with: PRIVATE_KEY=your_wallet_private_key_here\n"
  );
}

/** @type import('hardhat/config').HardhatUserConfig */
module.exports = {
  // ── Solidity compiler ──────────────────────────────────────────────────────
  solidity: {
    version: "0.8.20",
    settings: {
      optimizer: {
        enabled: true,
        runs: 200,
      },
    },
  },

  // ── Networks ───────────────────────────────────────────────────────────────
  networks: {
    // Local development (default, no key needed)
    hardhat: {
      chainId: 31337,
    },

    // Polygon Amoy Testnet
    amoy: {
      url: process.env.AMOY_RPC_URL || "https://rpc-amoy.polygon.technology",
      chainId: 80002,
      accounts: PRIVATE_KEY ? [PRIVATE_KEY] : [],
      gasPrice: 30000000000, // 30 gwei
      gas: 3000000,
    },

    // Polygon Mainnet (for future production use)
    polygon: {
      url: process.env.POLYGON_RPC_URL || "https://polygon-rpc.com",
      chainId: 137,
      accounts: PRIVATE_KEY ? [PRIVATE_KEY] : [],
    },
  },

  // ── Contract verification (optional) ──────────────────────────────────────
  etherscan: {
    apiKey: {
      polygonAmoy: POLYGONSCAN_API_KEY,
      polygon: POLYGONSCAN_API_KEY,
    },
    customChains: [
      {
        network: "polygonAmoy",
        chainId: 80002,
        urls: {
          apiURL: "https://api-amoy.polygonscan.com/api",
          browserURL: "https://amoy.polygonscan.com",
        },
      },
    ],
  },

  // ── Gas reporter (optional, shows gas costs after tests) ──────────────────
  gasReporter: {
    enabled: process.env.REPORT_GAS !== undefined,
    currency: "USD",
  },

  // ── Paths ──────────────────────────────────────────────────────────────────
  paths: {
    sources:   "./contracts",
    tests:     "./test",
    cache:     "./cache",
    artifacts: "./artifacts",
  },
};
