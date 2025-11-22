import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from fl_engine.task import Net

from typing import Any
import hashlib
import json
from pathlib import Path

import torch

from .task import Net
from .onchain_logger import OnchainFLLogger


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

        logger = OnchainFLLogger(
            rpc_url="http://127.0.0.1:8545",
            contract_address="0x5FbDB2315678afecb367f032d93F642f64180aa3",  # update from last deploy
            owner_private_key="0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
            server_private_key="0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d",
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

