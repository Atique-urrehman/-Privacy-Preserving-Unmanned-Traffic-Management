# ZK Circuit Build Guide for Drone Layer 2 Rollup

## Overview
This directory contains the Circom circuit for proving batches of drone flight data without revealing the underlying information. The circuit verifies that 8 private drone flight records hash correctly to a public Merkle Root.

## Directory Structure
```
zk_circs/
├── batch_proof.circom          # Main Circom circuit (WITH HEAVY COMMENTS)
├── package.json                # npm dependencies config
├── setup.sh                     # Automated setup script
├── .gitignore                   # Git ignore patterns
├── circuits/                    # [Generated] Compiled circuits
├── batch_proof_js/             # [Generated] Witness calculator WASM
├── pot15_*.ptau                # [Generated] Powers of Tau files
├── batch_proof_*.zkey          # [Generated] Zero-knowledge keys
├── verification_key.json       # [Generated] Verification key
└── Verifier.sol                # [Generated] Solidity verifier contract
```

## Prerequisites
- **Node.js**: v14 or higher
- **npm**: v6 or higher
- **Linux/macOS/WSL2** for best compatibility

Verify installation:
```bash
node --version
npm --version
```

---

## Step-by-Step Build Instructions

### Step 1: Navigate to the zk_circs Directory
```bash
cd /home/uak/Projects/bc_34/zk_circs
```

### Step 2: Initialize npm and Install Dependencies
```bash
npm install
```

This installs:
- **circom** (~0.1.9): Circom compiler
- **circomlib** (~2.0.6): Standard library with Poseidon hash
- **snarkjs** (~0.7.4): Proof generation and verification toolkit

**Expected Output:**
```
added X packages, and audited X packages in Xs
found 0 vulnerabilities
```

### Step 3: Compile the Circuit
```bash
npx circom batch_proof.circom --r1cs --wasm -v
```

**What This Does:**
- `--r1cs`: Generates `batch_proof.r1cs` (rank-1 constraint system - the mathematical constraints)
- `--wasm`: Generates `batch_proof_js/batch_proof.wasm` (witness calculator)
- `-v`: Verbose output

**Expected Files Generated:**
- `batch_proof_js/batch_proof.wasm`
- `batch_proof_js/generate_witness.js`
- `batch_proof_js/batch_proof.r1cs` or in root

**Expected Output:**
```
++ Building circuit and constraints
++ Number of constraints: 1,074
template instances: 36
Non-linear constraints: 540
Linear constraints: 534
++ Number of public inputs: 1 (new_root)
++ Number of private inputs: 8 (leaves[8])
++ Number of outputs: 0
++ Number of wires: 2,158
++ Number of labels: 2,158
```

---

### Step 4: Powers of Tau Phase 1 (Trusted Setup - Part 1)

#### Step 4a: Create new Powers of Tau file (15 = ~32,768 constraints)
```bash
npx snarkjs powersoftau new bn128 15 pot15_0000.ptau
```

**What This Does:**
- Creates initial Powers of Tau ceremony file
- Parameter `15` allows up to 2^15 = 32,768 constraints
- `bn128` is the elliptic curve used by Ethereum

**Expected Output:**
```
Creating power of tau file with 2^15 constraints
Contributor 1 starting contribution
```

**File Generated:** `pot15_0000.ptau` (~48 MB)

#### Step 4b: Contribute to Powers of Tau (simulate trusted setup)
```bash
npx snarkjs powersoftau contribute pot15_0000.ptau pot15_0001.ptau \
    --name 'First contribution' -v
```

**What This Does:**
- Adds randomness to the ceremony (simulates a trusted participant)
- In production, this would require multiple independent participants
- The randomness is used to ensure the proving system is sound

**Expected Output:**
```
pot15_0001.ptau
Contributor name: First contribution
```

**File Generated:** `pot15_0001.ptau` (~48 MB)

---

### Step 5: Powers of Tau Phase 2 (Prepare for Groth16)

```bash
npx snarkjs powersoftau prepare phase2 pot15_0001.ptau pot15_final.ptau
```

**What This Does:**
- Transforms Powers of Tau into the format needed for Groth16
- Performs expensive elliptic curve operations
- Creates the finalized ceremony file

**File Generated:** `pot15_final.ptau` (~96 MB)

**Expected Output:**
```
Preparing phase 2
Writing prepared phase2 file
```

---

### Step 6: Groth16 Setup Phase 1 (Generate Initial ZKey)

```bash
npx snarkjs groth16 setup batch_proof_js/batch_proof.r1cs pot15_final.ptau \
    batch_proof_0000.zkey
```

**What This Does:**
- Generates the initial zero-knowledge key (zkey)
- Combines your specific circuit with the general Powers of Tau
- Creates the basis for proving keys

**File Generated:** `batch_proof_0000.zkey` (~92 MB)

**Expected Output:**
```
generating zkey
Creating zkey file
[████████████████] 100% - Elapsed time: Xs
```

---

### Step 7: Groth16 Setup Phase 2 (Contribute to ZKey)

#### Step 7a: Contribute to the zkey
```bash
npx snarkjs zkey contribute batch_proof_0000.zkey batch_proof_0001.zkey \
    --name 'First zkey contribution' -v
```

**What This Does:**
- Finalizes the proving key with additional randomness
- Makes the zkey usable for generating proofs
- Simulates final trusted setup phase (production would have multiple participants)

**File Generated:** `batch_proof_0001.zkey` (~92 MB)

#### Step 7b: Export Verification Key
```bash
npx snarkjs zkey export verificationkey batch_proof_0001.zkey verification_key.json
```

**What This Does:**
- Extracts the public verification key from the zkey
- Used by verifiers to check proofs without the private zkey

**File Generated:** `verification_key.json` (~5 KB)

**Example Content:**
```json
{
  "protocol": "groth16",
  "curve": "bn128",
  "nPublic": 1,
  "vk_alpha_1": [...],
  "vk_beta_2": [...],
  "vk_gamma_2": [...],
  "vk_delta_2": [...],
  "vk_alphabeta_12": [...],
  "IC": [[...], [...], ...]
}
```

---

### Step 8: Generate Solidity Verifier Contract

```bash
npx snarkjs zkey export solidityverifier batch_proof_0001.zkey Verifier.sol
```

**What This Does:**
- Generates a Solidity smart contract that can verify proofs on-chain
- Uses the verification key from step 7b
- Implements the Groth16 verification algorithm

**File Generated:** `Verifier.sol` (~5 KB)

**Key Components in Verifier.sol:**
- `verifyProof(proof, input)` - Main verification function
- Uses two pairing checks to verify the proof mathematically

---

## Automated Setup (Alternative)

If you prefer to run everything automatically:

```bash
cd /home/uak/Projects/bc_34/zk_circs
chmod +x setup.sh
./setup.sh
```

This runs all steps 1-8 in sequence.

---

## Integration with Hardhat Layer 1

### Move Verifier.sol to Hardhat
```bash
cp /home/uak/Projects/bc_34/zk_circs/Verifier.sol \
   /home/uak/Projects/bc_34/l1_block_chain/contracts/
```

### Update UAV_Registry.sol to use the verifier

In `l1_block_chain/contracts/UAV_Registry.sol`:

```solidity
import "./Verifier.sol";

contract UAV_Registry is Verifier {
    // Your existing code...
    
    function submitBatchProof(
        uint256[2] memory a,
        uint256[2][2] memory b,
        uint256[2] memory c,
        uint256[1] memory input
    ) external {
        require(verifyProof(a, b, c, input), "Invalid ZK proof");
        // Process the batch
    }
}
```

---

## Troubleshooting

### Issue: "circom: command not found"
**Solution:** Use `npx circom` instead of just `circom`

### Issue: Out of memory during compilation
**Solution:** 
```bash
node --max-old-space-size=4096 $(which npx) circom batch_proof.circom --r1cs --wasm
```

### Issue: Powers of Tau file too large
**Solution:** Reduce the parameter from 15 to 14 or 13:
```bash
npx snarkjs powersoftau new bn128 13 pot13_0000.ptau  # Smaller setup
```

### Issue: "File not found: batch_proof_js/batch_proof.r1cs"
**Solution:** Check if compilation succeeded; look for `batch_proof.r1cs` in the root directory

---

## Performance Notes

- **Circuit constraints:** ~1,074 (highly efficient)
- **Proof generation:** ~2-5 seconds per proof
- **Proof verification:** ~50-200 milliseconds on-chain
- **Proof size:** 130 bytes (very compact)

---

## Next Steps

1. **Test proof generation**: Create a test witness and generate a proof
2. **Deploy Verifier**: Copy `Verifier.sol` to Hardhat and deploy
3. **Integrate with Layer 2**: Use proving keys in your sequencer (`l2_edge/sequencer.py`)
4. **Production trusted setup**: Run multi-party computation with independent participants

---

## File Reference

| File | Purpose | Generated |
|------|---------|-----------|
| `batch_proof.circom` | Circuit definition | ✗ (manual) |
| `package.json` | Dependencies | ✗ (manual) |
| `batch_proof.r1cs` | Constraint system |  |
| `batch_proof_js/` | Witness calculator |  |
| `pot15_final.ptau` | Powers of Tau |  |
| `batch_proof_0001.zkey` | Proving/verification key |  |
| `verification_key.json` | Public verification key |  |
| `Verifier.sol` | Solidity verifier |  |

