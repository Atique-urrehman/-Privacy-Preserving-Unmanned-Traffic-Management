from pathlib import Path
import os
from typing import Dict, Optional

from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization, hashes as crypto_hashes
from Crypto.Hash import SHA256
from web3 import Web3
import subprocess
import json
from pathlib import Path


class Sequencer:
    def __init__(self):
        self.active_airspace: Dict[str, str] = {}

        self._private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        public_key = self._private_key.public_key()

        pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        out_path = Path(__file__).parent / "public_key.pem"
        out_path.write_bytes(pem)

    def receive_payload(self, encrypted_box: bytes) -> str:
        try:
            plaintext = self._private_key.decrypt(
                encrypted_box,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=crypto_hashes.SHA256()),
                    algorithm=crypto_hashes.SHA256(),
                    label=None,
                ),
            )
        except Exception as e:
            return f"REJECTED (DecryptionError: {e})"

        try:
            text = plaintext.decode("utf-8")
            pos_part, epoch_part, uav_id = text.split("|")
            x, y, z = pos_part.split(",")
        except Exception as e:
            return f"REJECTED (MalformedPayload: {e})"

        key = f"{x}-{y}-{z}-{epoch_part}"
        if key in self.active_airspace:
            return "REJECTED (Collision)"

        self.active_airspace[key] = uav_id
        return "APPROVED"

    def generate_batch(self) -> Optional[str]:
        if not self.active_airspace:
            return None

        leaves = []
        for key, uav in self.active_airspace.items():
            try:
                x, y, z, epoch = key.split("-")
            except ValueError:
                continue

            concat = f"{x}{y}{z}{epoch}{uav}"
            h = SHA256.new(concat.encode("utf-8")).digest()
            leaves.append(h)

        if not leaves:
            return None

        # Normalize leaves to exactly 8 elements (circuit expects 8 leaves)
        if len(leaves) > 8:
            leaves = leaves[:8]
        while len(leaves) < 8:
            leaves.append(leaves[-1])

        level = leaves
        while len(level) > 1:
            if len(level) % 2 == 1:
                level.append(level[-1])

            next_level = []
            for i in range(0, len(level), 2):
                combined = level[i] + level[i + 1]
                node_hash = SHA256.new(combined).digest()
                next_level.append(node_hash)
            level = next_level

        root = level[0]
        root_hex = root.hex()

        # Also produce field elements for each leaf (as decimal strings)
        # Convert SHA256 digest bytes to field elements by reducing mod BN128 field
        field_modulus = 21888242871839275222246405745257275088548364400416034343698204186575808495617
        leaves_field = [str(int.from_bytes(h, byteorder='big') % field_modulus) for h in leaves]

        self.active_airspace.clear()
        return root_hex, leaves_field


def submit_to_l1(new_root_hex, contract_address, contract_abi, leaves_field=None):
    # Attempt to run the ZK proof pipeline before submitting to L1.
    # This will generate inputs.json, witness.wtns, proof.json and public.json
    # in the `zk_circs` folder so you can see the Circom/snarkjs outputs.
    zk_dir = Path(__file__).resolve().parents[1] / "zk_circs"
    strict = os.getenv("STRICT_PROOFS", "1") not in ("0", "false", "False")
    pipeline_failed = False
    try:
        # If caller supplied leaves, write them to seed_leaves.json so the Node generator
        # will use the actual Sequencer leaves for Poseidon hashing and inputs.json.
        if leaves_field is not None:
            seed_path = zk_dir / "seed_leaves.json"
            seed_path.write_text(json.dumps({"leaves": leaves_field}, indent=2), encoding="utf-8")

        # Run the Poseidon input generator (it will use seed_leaves.json if present)
        subprocess.run(["node", "generate_poseidon_inputs.js"], cwd=str(zk_dir), check=True)

        # Generate witness using the generated WASM
        gw = zk_dir / "batch_proof_js" / "generate_witness.js"
        wasm = zk_dir / "batch_proof_js" / "batch_proof.wasm"
        if gw.exists() and wasm.exists():
            subprocess.run(["node", str(gw), str(wasm), str(zk_dir / "inputs.json"), str(zk_dir / "witness.wtns")], cwd=str(zk_dir), check=True)

        # Generate proof using zkey
        zkey = zk_dir / "batch_proof_0001.zkey"
        if zkey.exists():
            subprocess.run(["npx", "snarkjs", "groth16", "prove", str(zkey), str(zk_dir / "witness.wtns"), str(zk_dir / "proof.json"), str(zk_dir / "public.json")], cwd=str(zk_dir), check=True)

        # Read proof artifact to attach to transaction (as JSON bytes)
        proof_bytes = None
        proof_path = zk_dir / "proof.json"
        if proof_path.exists():
            proof_json = json.loads(proof_path.read_text(encoding="utf-8"))
            proof_bytes = json.dumps(proof_json).encode("utf-8")
        else:
            pipeline_failed = True
            print("[sequencer] proof.json not found after pipeline")
    except subprocess.CalledProcessError as e:
        pipeline_failed = True
        print(f"[sequencer] ZK pipeline failed: {e}")

    if pipeline_failed:
        if strict:
            raise RuntimeError("ZK proof generation failed and STRICT_PROOFS is enabled; aborting L1 submission")
        else:
            print("[sequencer] Warning: proof generation failed; proceeding with mock proof due to non-strict mode")
            proof_bytes = b"mock_proof"

    w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:8545"))
    if not w3.is_connected():
        raise ConnectionError("Unable to connect to local Hardhat node at http://127.0.0.1:8545")

    account = w3.eth.account.from_key(
        "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
    )
    contract = w3.eth.contract(address=Web3.to_checksum_address(contract_address), abi=contract_abi)

    root_bytes = bytes.fromhex(new_root_hex[2:] if new_root_hex.startswith("0x") else new_root_hex)
    if len(root_bytes) != 32:
        raise ValueError("new_root_hex must decode to exactly 32 bytes")

    tx = contract.functions.submitBatch(proof_bytes, root_bytes).build_transaction(
        {
            "from": account.address,
            "nonce": w3.eth.get_transaction_count(account.address),
            "gas": 300000,
            "gasPrice": w3.eth.gas_price,
            "chainId": w3.eth.chain_id,
        }
    )

    signed_tx = account.sign_transaction(tx)
    tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    print(f"L1 submit transaction hash: {tx_hash.hex()}")
    print(f"L1 receipt status: {receipt.status}")
    return receipt


if __name__ == "__main__":
    s = Sequencer()
    print(f"Sequencer started, public key written to: {Path(__file__).parent / 'public_key.pem'}")
