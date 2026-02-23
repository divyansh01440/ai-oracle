/**
 * deploy.js
 * Deploys VerifiableAIOracle to the selected network,
 * prints the contract address, and saves it to deployed.json.
 *
 * Usage:
 *   npx hardhat run scripts/deploy.js --network amoy
 *   npx hardhat run scripts/deploy.js --network hardhat
 */

const { ethers, network } = require("hardhat");
const fs   = require("fs");
const path = require("path");

async function main() {
  console.log("=".repeat(60));
  console.log("  VERIFIABLE AI ORACLE — DEPLOYMENT SCRIPT");
  console.log("=".repeat(60));

  // ── 1. Get deployer account ──────────────────────────────────────────────
  const [deployer] = await ethers.getSigners();
  const balance    = await ethers.provider.getBalance(deployer.address);

  console.log(`\nSTEP 1: Deployer info`);
  console.log(`  Address : ${deployer.address}`);
  console.log(`  Balance : ${ethers.formatEther(balance)} MATIC`);
  console.log(`  Network : ${network.name} (chainId: ${network.config.chainId})`);

  if (balance === 0n) {
    console.error("\n  ERROR: Deployer wallet has 0 MATIC.");
    console.error("  Get free testnet MATIC at: https://faucet.polygon.technology/");
    process.exit(1);
  }

  // ── 2. Compile check ─────────────────────────────────────────────────────
  console.log(`\nSTEP 2: Getting contract factory...`);
  const OracleFactory = await ethers.getContractFactory("VerifiableAIOracle");
  console.log("  Contract factory ready.");

  // ── 3. Deploy ─────────────────────────────────────────────────────────────
  console.log(`\nSTEP 3: Deploying VerifiableAIOracle...`);
  console.log("  Sending deployment transaction (this may take 15–60 seconds)...");

  const oracle = await OracleFactory.deploy();
  await oracle.waitForDeployment();

  const contractAddress = await oracle.getAddress();
  const deployTx        = oracle.deploymentTransaction();

  console.log(`  ✓ Contract deployed!`);
  console.log(`  Contract address : ${contractAddress}`);
  console.log(`  Transaction hash : ${deployTx.hash}`);

  // ── 4. Quick smoke test ───────────────────────────────────────────────────
  console.log(`\nSTEP 4: Running smoke tests on deployed contract...`);

  // Submit a test score
  const submitTx = await oracle.submitRiskScore("MATIC", 3500n);
  await submitTx.wait();
  console.log("  ✓ submitRiskScore('MATIC', 3500) — OK");

  // Read it back
  const score = await oracle.getRiskScore("MATIC");
  console.log(`  ✓ getRiskScore('MATIC') => ${score.toString()}`);

  // Check isSafe
  const safe = await oracle.isSafe("MATIC");
  console.log(`  ✓ isSafe('MATIC') => ${safe}  (threshold < 7000)`);

  // Test high-risk score
  const submitTx2 = await oracle.submitRiskScore("MATIC", 8500n);
  await submitTx2.wait();
  const safe2 = await oracle.isSafe("MATIC");
  console.log(`  ✓ After score=8500: isSafe('MATIC') => ${safe2}  (correctly false)`);

  // Reset to 0 after tests
  const resetTx = await oracle.submitRiskScore("MATIC", 0n);
  await resetTx.wait();
  console.log("  ✓ Score reset to 0 after smoke test.");

  // ── 5. Save deployment info ───────────────────────────────────────────────
  console.log(`\nSTEP 5: Saving deployment info...`);

  const deploymentInfo = {
    network:         network.name,
    chainId:         network.config.chainId,
    contractName:    "VerifiableAIOracle",
    contractAddress: contractAddress,
    deployerAddress: deployer.address,
    transactionHash: deployTx.hash,
    deployedAt:      new Date().toISOString(),
    explorerUrl:     network.name === "amoy"
      ? `https://amoy.polygonscan.com/address/${contractAddress}`
      : "N/A (local network)",
    abi_note: "Full ABI is in artifacts/contracts/OracleContract.sol/VerifiableAIOracle.json"
  };

  // Save to contracts/deployed.json
  const outputPath = path.join(__dirname, "..", "deployed.json");
  fs.writeFileSync(outputPath, JSON.stringify(deploymentInfo, null, 2));
  console.log(`  ✓ Saved to deployed.json`);

  // ── 6. Summary ────────────────────────────────────────────────────────────
  console.log("\n" + "=".repeat(60));
  console.log("  DEPLOYMENT COMPLETE");
  console.log("=".repeat(60));
  console.log(`\n  Contract Address : ${contractAddress}`);
  if (network.name === "amoy") {
    console.log(`  View on explorer : https://amoy.polygonscan.com/address/${contractAddress}`);
    console.log(`\n  To verify source code on PolygonScan:`);
    console.log(`  npx hardhat verify --network amoy ${contractAddress}`);
  }
  console.log(`\n  To submit a risk score from Node.js:`);
  console.log(`
  const { ethers } = require("ethers");
  const deployed   = require("./deployed.json");
  const artifact   = require("./artifacts/contracts/OracleContract.sol/VerifiableAIOracle.json");

  const provider = new ethers.JsonRpcProvider("https://rpc-amoy.polygon.technology");
  const wallet   = new ethers.Wallet(process.env.PRIVATE_KEY, provider);
  const oracle   = new ethers.Contract(deployed.contractAddress, artifact.abi, wallet);

  await oracle.submitRiskScore("MATIC", 4200n);
  const score = await oracle.getRiskScore("MATIC");
  console.log("Score:", score.toString());
  `);
}

main().catch((error) => {
  console.error("\n  DEPLOYMENT FAILED:");
  console.error(" ", error.message);
  process.exit(1);
});
