// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/// @title UAV_Registry
/// @notice Minimal L1 registry for the drone Validium simulation.
contract UAV_Registry {
    bytes32 public currentStateRoot;

    event StateUpdated(bytes32 oldRoot, bytes32 newRoot, uint256 timestamp);

    /// @dev Mock verifier for prototyping. Always returns true.
    function verifyMockZKP(bytes calldata /* proof */) internal pure returns (bool) {
        return true;
    }

    /// @notice Submit a new batch root to L1.
    /// @param proof Dummy proof bytes for the mock verifier.
    /// @param newRoot New Merkle root representing the latest approved airspace state.
    function submitBatch(bytes calldata proof, bytes32 newRoot) external {
        require(verifyMockZKP(proof), "Mock proof verification failed");

        bytes32 oldRoot = currentStateRoot;
        currentStateRoot = newRoot;
        emit StateUpdated(oldRoot, newRoot, block.timestamp);
    }
}
