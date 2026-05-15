// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "./Verifier.sol";

/// @title UAV_Registry
/// @notice Minimal L1 registry for the drone Validium simulation.
contract UAV_Registry is Groth16Verifier {
    bytes32 public currentStateRoot;

    event StateUpdated(bytes32 oldRoot, bytes32 newRoot, uint256 timestamp);

    /// @dev Verify a serialized Groth16 proof produced by the circom/snarkjs toolchain.
    /// The `proof` bytes must be an abi-encoded tuple `(uint[2], uint[2][2], uint[2], uint[1])`.
    function verifyZKP(bytes calldata proof) internal view returns (bool) {
        (uint[2] memory a, uint[2][2] memory b, uint[2] memory c, uint[1] memory input) = abi.decode(proof, (uint[2], uint[2][2], uint[2], uint[1]));
        return verifyProof(a, b, c, input);
    }

    /// @notice Submit a new batch root to L1.
    /// @param proof ABI-encoded Groth16 proof `(a, b, c, publicInputs)`.
    /// @param newRoot New Merkle root representing the latest approved airspace state.
    function submitBatch(bytes calldata proof, bytes32 newRoot) external {
        require(verifyZKP(proof), "ZK proof verification failed");

        bytes32 oldRoot = currentStateRoot;
        currentStateRoot = newRoot;
        emit StateUpdated(oldRoot, newRoot, block.timestamp);
    }
}
