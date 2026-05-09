import os
import random
from pathlib import Path
from typing import Tuple, List
import json

from cryptography.hazmat.primitives import serialization, hashes as crypto_hashes
from cryptography.hazmat.primitives.asymmetric import padding

from sequencer import Sequencer, submit_to_l1


UAV_REGISTRY_ABI = [
    {
        "type": "function",
        "name": "submitBatch",
        "stateMutability": "nonpayable",
        "inputs": [
            {"name": "proof", "type": "bytes"},
            {"name": "newRoot", "type": "bytes32"},
        ],
        "outputs": [],
    }
]

DEPLOYMENT_ADDRESS_FILE = (
    Path(__file__).resolve().parent.parent
    / "l1_block_chain"
    / "deployments"
    / "localhost"
    / "UAV_Registry.address.json"
)


def load_public_key(path: Path):
    data = path.read_bytes()
    return serialization.load_pem_public_key(data)


def random_uav_id() -> str:
    prefix = random.choice(["UAV-Amazon", "UAV-DroneX", "UAV-Scout", "UAV-Alpha"])
    return f"{prefix}-{random.randint(1,999)}"


def generate_coordinates(n: int) -> List[Tuple[int, int, int]]:
    coords = []
    for _ in range(n - 1):
        coords.append((random.randint(1, 20), random.randint(1, 20), random.randint(1, 20)))

    if coords:
        coords.append(coords[0])
    else:
        coords.append((1, 1, 1))

    random.shuffle(coords)
    return coords


def encrypt_message(public_key, message: str) -> bytes:
    return public_key.encrypt(
        message.encode("utf-8"),
        padding.OAEP(
            mgf=padding.MGF1(algorithm=crypto_hashes.SHA256()),
            algorithm=crypto_hashes.SHA256(),
            label=None,
        ),
    )


def load_contract_address() -> str | None:
    if DEPLOYMENT_ADDRESS_FILE.exists():
        payload = json.loads(DEPLOYMENT_ADDRESS_FILE.read_text(encoding="utf-8"))
        address = payload.get("address")
        if address:
            return address

    return os.getenv("UAV_REGISTRY_ADDRESS")


def run_simulation():
    seq = Sequencer()

    pk_path = Path(__file__).parent / "public_key.pem"
    public_key = load_public_key(pk_path)

    coords = generate_coordinates(10)

    for idx, (x, y, z) in enumerate(coords, start=1):
        epoch = random.randint(1, 100)
        uav_id = random_uav_id()
        payload = f"{x},{y},{z}|{epoch}|{uav_id}"
        encrypted = encrypt_message(public_key, payload)

        result = seq.receive_payload(encrypted)
        print(f"Request {idx}: {payload} -> {result}")

    result = seq.generate_batch()
    if result is None:
        print("No approved flights in this epoch; no Merkle root generated.")
    else:
        merkle_root, leaves_field = result
        print(f"Merkle Root: {merkle_root}")

        contract_address = load_contract_address()
        if contract_address:
            receipt = submit_to_l1(merkle_root, contract_address, UAV_REGISTRY_ABI, leaves_field)
            print(f"L1 submission confirmed in block {receipt.blockNumber}")
        else:
            print(
                f"Skipping L1 submission: deploy the registry first so {DEPLOYMENT_ADDRESS_FILE} exists, or set UAV_REGISTRY_ADDRESS manually."
            )


if __name__ == "__main__":
    run_simulation()
