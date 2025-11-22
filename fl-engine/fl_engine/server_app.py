import subprocess
import re
import hashlib
import json
import os
from pathlib import Path

from dotenv import load_dotenv
import torch

from fl_engine.storage.factory import get_storage_backend

from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from .task import Net
from .onchain_logger import OnchainFLLogger


# Load environment variables from .env file
env_path = Path(__file__).parent.parent / ".env"
if not env_path.exists():
    env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)


app = ServerApp()


def hash_arrays(arrays: ArrayRecord) -> bytes:
    """Compute a SHA-256 hash of the final global model."""
    m = hashlib.sha256()
    for _, tensor in arrays.to_torch_state_dict().items():
        m.update(tensor.cpu().numpy().tobytes())
    return m.digest()


def print_verification_summary(chain_env: str, rpc_url: str,
                               contract_address: str, storage_root: str | None):
    """Human-friendly validation instructions after training."""

    print("\n========== FEDERATED LEARNING VERIFICATION SUMMARY ==========\n")

    print(f"• Chain environment: {chain_env}")
    print(f"• Contract address: {contract_address}")

    # ---- Explorer links ----
    explorer = None
    if "evmrpc-testnet.0g.ai" in rpc_url:
        explorer = "https://chainscan-galileo.0g.ai"
    elif "evmrpc.0g.ai" in rpc_url:
        explorer = "https://chainscan.0g.ai"

    if explorer:
        print("\n📡 Explorer Links:")
        print(f"  - Contract:       {explorer}/address/{contract_address}")
        print(f"  - Transactions:   {explorer}/address/{contract_address}?tab=transaction")
    else:
        print("\n(No block explorer URL — likely Hardhat localhost)")

    # ---- Storage links ----
    if storage_root:
        indexer = os.getenv("ZEROG_INDEXER", "https://indexer-storage-testnet-turbo.0g.ai").rstrip("/")

        print("\n📦 0G Storage Verification:")
        print(f"  - Root Hash: {storage_root}")

        print("\n🔽 Direct HTTP Download (browser-friendly):")
        print(f"    {indexer}/file?root={storage_root}&name=final_model.pt")

        print("\n🔧 CLI Restore Command:")
        print(f"    0g-storage-client download \\")
        print(f"      --indexer {indexer}/ \\")
        print(f"      --root {storage_root} \\")
        print(f"      --file restored_final_model.pt --proof")

    print("\n==============================================================\n")


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # 1. FL config
    fraction_train = context.run_config["fraction-train"]
    num_rounds = context.run_config["num-server-rounds"]
    lr = context.run_config["lr"]

    # 2. Model
    arrays = ArrayRecord(Net().state_dict())

    # 3. Strategy
    strategy = FedAvg(fraction_train=fraction_train)

    # 4. Training run
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
    )

    # 5. Save final model
    print("\nSaving final model to disk…")
    model_path = "final_model.pt"
    torch.save(result.arrays.to_torch_state_dict(), model_path)

    # 6. Upload ONCE to selected backend (0G / filecoin / dummy)
    storage = get_storage_backend()
    try:
        storage_root = storage.upload(model_path)
    except Exception as e:
        print(f"[storage ERROR] {e}")
        storage_root = None

    # 7. On-chain logging
    try:
        artifact_path = (
            Path(__file__).parent.parent.parent
            / "artifacts"
            / "contracts"
            / "FederatedLearningLog.sol"
            / "FederatedLearningLog.json"
        )

        with open(artifact_path, "r") as f:
            abi = json.load(f)["abi"]

        rpc_url = os.getenv("RPC_URL")
        contract_address = os.getenv("CONTRACT_ADDRESS")
        owner_pk = os.getenv("OWNER_PRIVATE_KEY")
        server_pk = os.getenv("SERVER_PRIVATE_KEY")

        logger = OnchainFLLogger(
            rpc_url=rpc_url,
            contract_address=contract_address,
            owner_private_key=owner_pk,
            server_private_key=server_pk,
            abi=abi,
        )

        next_round = logger.get_latest_round_id() + 1
        model_hash = hash_arrays(result.arrays)

        server_addr = logger.server.address
        logger.ensure_client_registered(server_addr)

        logger.record_round(
            round_id=next_round,
            model_hash=model_hash,
            artifact_cid=storage_root or "NO-STORAGE",
            client_addresses=[server_addr],
            samples=[100],
            scores=[1000000],
        )

        # 8. Show verification summary
        chain_env = os.getenv("CHAIN_ENV", "unknown")
        print_verification_summary(
            chain_env=chain_env,
            rpc_url=rpc_url,
            contract_address=contract_address,
            storage_root=storage_root,
        )

    except Exception as e:
        print(f"[on-chain ERROR] {e}")