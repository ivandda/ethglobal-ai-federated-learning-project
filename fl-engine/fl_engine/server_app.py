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
# Look for .env in fl-engine directory or parent directory
env_path = Path(__file__).parent.parent / ".env"
if not env_path.exists():
    env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)

# Create ServerApp
app = ServerApp()


def hash_arrays(arrays: ArrayRecord) -> bytes:
    """Compute a SHA-256 hash of the final global model."""
    state_dict = arrays.to_torch_state_dict()
    m = hashlib.sha256()
    for _, tensor in state_dict.items():
        m.update(tensor.cpu().numpy().tobytes())
    return m.digest()  # 32 bytes, for bytes32 in Solidity


def print_verification_summary(
    chain_env: str,
    rpc_url: str,
    contract_address: str,
    storage_root: str | None,
) -> None:
    """Print human-friendly verification hints for 0G / local modes."""

    print("\n=== Verification Summary ===")

    # 1) Contract / explorer
    contract_address = contract_address.strip()
    print(f"- Contract address: {contract_address}")

    explorer_base = None
    if "evmrpc-testnet.0g.ai" in rpc_url:
        explorer_base = "https://chainscan-galileo.0g.ai"
    elif "evmrpc.0g.ai" in rpc_url:
        explorer_base = "https://chainscan.0g.ai"

    if explorer_base:
        print(f"- 0G explorer (address): {explorer_base}/address/{contract_address}")
        print(
            f"- Transactions tab:      {explorer_base}/address/{contract_address}?tab=transaction"
        )
    else:
        print("- No public explorer configured (probably local Hardhat).")

    # 2) Storage verification + HTTP download
    if storage_root:
        print(f"- 0G storage root: {storage_root}")
        indexer = os.getenv(
            "ZEROG_INDEXER", "https://indexer-storage-testnet-turbo.0g.ai/"
        ).rstrip("/")

        # CLI verification
        print("- To verify/download from 0G storage via CLI:")
        print(f"    0g-storage-client download \\")
        print(f"      --indexer {indexer}/ \\")
        print(f"      --root {storage_root} \\")
        print(f"      --file restored_final_model.pt --proof")

        # HTTP gateway link
        http_url = f"{indexer}/file?root={storage_root}&name=final_model.pt"
        print("- Direct HTTP download URL:")
        print(f"    {http_url}")

    print("=== End of Summary ===\n")



@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # --- 1) Read FL config from pyproject.toml ---
    fraction_train: float = context.run_config["fraction-train"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["lr"]

    # --- 2) Build global model and wrap as ArrayRecord ---
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # --- 3) Create FedAvg strategy (unchanged from tutorial) ---
    strategy = FedAvg(fraction_train=fraction_train)

    # --- 4) Run strategy ---
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
    )

    # --- 5) Save final model to disk (tutorial behavior) ---
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    model_path = "final_model.pt"
    torch.save(state_dict, model_path)

    # --- 6) Upload model to Filecoin (with graceful fallback) ---
    storage = get_storage_backend()
    artifact_id = None
    try:
        artifact_id = storage.upload(model_path)
    except Exception as e:
        print(f"[storage ERROR] {e}")

    try:
        artifact_cid = storage.upload(model_path)
    except Exception as e:
        print(f"[storage ERROR] {e}")
        artifact_cid = f"bafyDUMMYcid-final-round-{num_rounds}"

    # --- 7) On-chain logging (simple POC) ---
    try:
        # Load ABI from Hardhat artifacts
        artifact_path = (
            Path(__file__).parent.parent.parent
            / "artifacts"
            / "contracts"
            / "FederatedLearningLog.sol"
            / "FederatedLearningLog.json"
        )
        if not artifact_path.exists():
            # Try alternative path (if running from different directory)
            artifact_path = Path(
                "artifacts/contracts/FederatedLearningLog.sol/FederatedLearningLog.json"
            )

        if artifact_path.exists():
            with open(artifact_path, "r") as f:
                artifact = json.load(f)
                FEDLEARN_LOG_ABI = artifact["abi"]
        else:
            raise FileNotFoundError(f"Contract artifact not found at {artifact_path}")

        # Load configuration from environment variables
        rpc_url = os.getenv("RPC_URL", "http://127.0.0.1:8545")
        contract_address = os.getenv("CONTRACT_ADDRESS")
        owner_private_key = os.getenv("OWNER_PRIVATE_KEY")
        server_private_key = os.getenv("SERVER_PRIVATE_KEY")

        # Validate required environment variables
        if not contract_address:
            raise ValueError(
                "CONTRACT_ADDRESS not found in environment variables. Please set it in .env file."
            )
        if not owner_private_key:
            raise ValueError(
                "OWNER_PRIVATE_KEY not found in environment variables. Please set it in .env file."
            )
        if not server_private_key:
            raise ValueError(
                "SERVER_PRIVATE_KEY not found in environment variables. Please set it in .env file."
            )

        logger = OnchainFLLogger(
            rpc_url=rpc_url,
            contract_address=contract_address,
            owner_private_key=owner_private_key,
            server_private_key=server_private_key,
            abi=FEDLEARN_LOG_ABI,
        )

        model_hash = hash_arrays(result.arrays)

        # Get the next sequential round ID from the contract
        next_round_id = logger.get_latest_round_id() + 1

        # For this POC, we'll use a placeholder client (server address)
        placeholder_client = logger.server.address
        logger.ensure_client_registered(placeholder_client)

        logger.record_round(
            round_id=next_round_id,
            model_hash=model_hash,
            artifact_cid=artifact_cid,
            client_addresses=[placeholder_client],
            samples=[100],      # Placeholder
            scores=[1000000],   # Placeholder
        )

        # --- 8) Print verification summary ---
        chain_env = os.getenv("CHAIN_ENV", "unknown")
        rpc_url = os.getenv("RPC_URL", "")
        contract_address = os.getenv("CONTRACT_ADDRESS", "")
        print_verification_summary(
            chain_env=chain_env,
            rpc_url=rpc_url,
            contract_address=contract_address,
            storage_root=artifact_id,
        )

    except Exception as e:
        print(f"[on-chain ERROR] Failed to log final round: {e!r}")
