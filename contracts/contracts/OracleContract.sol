// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title VerifiableAIOracle
 * @notice Stores AI-computed manipulation risk scores (0–10000) for crypto assets.
 *         Only the contract owner (you) can submit scores.
 *         Anyone can read scores or check if an asset is "safe".
 */
contract VerifiableAIOracle {

    // ── State ────────────────────────────────────────────────────────────────
    address public owner;

    // asset symbol (e.g. "MATIC") => risk score 0–10000
    mapping(string => uint256) private riskScores;

    // asset symbol => timestamp of last update
    mapping(string => uint256) private lastUpdated;

    // threshold below which an asset is considered "safe"
    uint256 public constant SAFE_THRESHOLD = 7000;

    // ── Events ───────────────────────────────────────────────────────────────
    /**
     * @notice Emitted every time a risk score is updated.
     * @param asset   The asset symbol (e.g. "MATIC")
     * @param score   The new risk score (0–10000)
     * @param timestamp Block timestamp of the update
     * @param submitter Address that submitted the score (always owner)
     */
    event RiskScoreUpdated(
        string indexed asset,
        uint256 score,
        uint256 timestamp,
        address indexed submitter
    );

    /**
     * @notice Emitted when ownership is transferred.
     */
    event OwnershipTransferred(
        address indexed previousOwner,
        address indexed newOwner
    );

    // ── Modifiers ────────────────────────────────────────────────────────────
    modifier onlyOwner() {
        require(msg.sender == owner, "OracleContract: caller is not the owner");
        _;
    }

    modifier validScore(uint256 score) {
        require(score <= 10000, "OracleContract: score must be between 0 and 10000");
        _;
    }

    modifier nonEmptyAsset(string memory asset) {
        require(bytes(asset).length > 0, "OracleContract: asset name cannot be empty");
        _;
    }

    // ── Constructor ──────────────────────────────────────────────────────────
    constructor() {
        owner = msg.sender;
        emit OwnershipTransferred(address(0), msg.sender);
    }

    // ── Write Functions (owner only) ─────────────────────────────────────────

    /**
     * @notice Submit or update the risk score for an asset.
     * @param asset  Asset symbol string, e.g. "MATIC" or "BTC"
     * @param score  Risk score from 0 (no risk) to 10000 (maximum risk)
     */
    function submitRiskScore(string memory asset, uint256 score)
        external
        onlyOwner
        validScore(score)
        nonEmptyAsset(asset)
    {
        riskScores[asset]  = score;
        lastUpdated[asset] = block.timestamp;

        emit RiskScoreUpdated(asset, score, block.timestamp, msg.sender);
    }

    /**
     * @notice Transfer ownership to a new address.
     * @param newOwner The address of the new owner.
     */
    function transferOwnership(address newOwner) external onlyOwner {
        require(newOwner != address(0), "OracleContract: new owner is zero address");
        emit OwnershipTransferred(owner, newOwner);
        owner = newOwner;
    }

    // ── Read Functions (public) ──────────────────────────────────────────────

    /**
     * @notice Get the current risk score for an asset.
     * @param asset  Asset symbol string.
     * @return score The risk score (0–10000). Returns 0 if never set.
     */
    function getRiskScore(string memory asset)
        external
        view
        nonEmptyAsset(asset)
        returns (uint256 score)
    {
        return riskScores[asset];
    }

    /**
     * @notice Check whether an asset is considered safe (score < SAFE_THRESHOLD).
     * @param asset  Asset symbol string.
     * @return True if the score is below 7000, false otherwise.
     */
    function isSafe(string memory asset)
        external
        view
        nonEmptyAsset(asset)
        returns (bool)
    {
        return riskScores[asset] < SAFE_THRESHOLD;
    }

    /**
     * @notice Get the timestamp of the last score update for an asset.
     * @param asset  Asset symbol string.
     * @return Unix timestamp of the last update (0 if never updated).
     */
    function getLastUpdated(string memory asset)
        external
        view
        nonEmptyAsset(asset)
        returns (uint256)
    {
        return lastUpdated[asset];
    }

    /**
     * @notice Get the full details for an asset in one call.
     * @param asset  Asset symbol string.
     * @return score        Current risk score.
     * @return safe         Whether the asset is considered safe.
     * @return updatedAt    Timestamp of last update.
     */
    function getAssetInfo(string memory asset)
        external
        view
        nonEmptyAsset(asset)
        returns (
            uint256 score,
            bool safe,
            uint256 updatedAt
        )
    {
        score     = riskScores[asset];
        safe      = score < SAFE_THRESHOLD;
        updatedAt = lastUpdated[asset];
    }
}
