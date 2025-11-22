import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from fl_engine.task import Net

from typing import Any
import hashlib
import json
import os
from pathlib import Path
from dotenv import load_dotenv

import torch

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
    torch.save(state_dict, "final_model.pt")

    # --- 6) On-chain logging (simple POC) ---
    try:
        # Load ABI from Hardhat artifacts
        artifact_path = Path(__file__).parent.parent.parent / "artifacts" / "contracts" / "FederatedLearningLog.sol" / "FederatedLearningLog.json"
        if not artifact_path.exists():
            # Try alternative path (if running from different directory)
            artifact_path = Path("artifacts/contracts/FederatedLearningLog.sol/FederatedLearningLog.json")
        
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
            raise ValueError("CONTRACT_ADDRESS not found in environment variables. Please set it in .env file.")
        if not owner_private_key:
            raise ValueError("OWNER_PRIVATE_KEY not found in environment variables. Please set it in .env file.")
        if not server_private_key:
            raise ValueError("SERVER_PRIVATE_KEY not found in environment variables. Please set it in .env file.")
        
        logger = OnchainFLLogger(
            rpc_url=rpc_url,
            contract_address=contract_address,
            owner_private_key=owner_private_key,
            server_private_key=server_private_key,
            abi=FEDLEARN_LOG_ABI,
        )

        model_hash = hash_arrays(result.arrays)
        # For now, just a dummy CID; later replace with real Filecoin CID
        artifact_cid = f"bafyDUMMYcid-final-round-{num_rounds}"

        # Get the next sequential round ID from the contract
        next_round_id = logger.get_latest_round_id() + 1
        
        # For this POC, we'll use a placeholder client (server address) since we don't track
        # individual client contributions in this simple setup
        # In a real implementation, you'd track client participation during training
        placeholder_client = logger.server.address
        logger.ensure_client_registered(placeholder_client)
        
        # Log the round with placeholder data
        # Note: In production, you'd collect actual client addresses, samples, and scores
        logger.record_round(
            round_id=next_round_id,  # Use sequential ID from contract
            model_hash=model_hash,
            artifact_cid=artifact_cid,
            client_addresses=[placeholder_client],
            samples=[100],  # Placeholder sample count
            scores=[1000000],  # Placeholder score
        )

    except Exception as e:
        print(f"[on-chain ERROR] Failed to log final round: {e!r}")

