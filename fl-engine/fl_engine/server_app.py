import hashlib
import json
import os
import time
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
import torch

from fl_engine.storage.factory import get_storage_backend

from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from .task import Net
from .onchain_logger import OnchainFLLogger

# Global list to store round metrics during training
_round_metrics: List[Dict[str, Any]] = []

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


def print_verification_summary(
    chain_env: str,
    rpc_url: str,
    contract_address: str,
    storage_root: str | None,
) -> None:
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
        indexer = os.getenv(
            "ZEROG_INDEXER",
            "https://indexer-storage-testnet-turbo.0g.ai",
        ).rstrip("/")

        print("\n📦 0G Storage Verification:")
        print(f"  - Root Hash: {storage_root}")

        print("\n🔽 Direct HTTP Download (browser-friendly):")
        print(f"    {indexer}/file?root={storage_root}&name=final_model.pt")

        print("\n🔧 CLI Restore Command:")
        print("    0g-storage-client download \\")
        print(f"      --indexer {indexer}/ \\")
        print(f"      --root {storage_root} \\")
        print("      --file restored_final_model.pt --proof")

    print("\n==============================================================\n")


def write_run_summary(
    result,
    artifact_id: str | None,
    rpc_url: str,
    contract_address: str,
    last_round_id: int,
    num_server_rounds: int,
    context: Context,
) -> None:
    """Dump a machine-readable summary for the LLM helper."""
    
    # Extract metrics from result if available
    global_metrics = {}
    if hasattr(result, 'metrics') and result.metrics:
        # Try to extract metrics from result object
        try:
            if hasattr(result.metrics, 'to_dict'):
                global_metrics = result.metrics.to_dict()
            elif isinstance(result.metrics, dict):
                global_metrics = result.metrics
        except Exception as e:
            print(f"[summary] Warning: Could not extract metrics: {e}")
    
    # Extract round-by-round data if available
    rounds_data = []
    
    # First, try to use our captured round metrics
    if _round_metrics:
        rounds_data = _round_metrics.copy()
    # Otherwise, try to extract from result history
    elif hasattr(result, 'history') and result.history:
        try:
            for i, hist_entry in enumerate(result.history):
                round_info = {
                    "round": i + 1,
                    "metrics": {}
                }
                # Try to extract metrics from history entry
                if hasattr(hist_entry, 'metrics') and hist_entry.metrics:
                    if hasattr(hist_entry.metrics, 'to_dict'):
                        round_info["metrics"] = hist_entry.metrics.to_dict()
                    elif isinstance(hist_entry.metrics, dict):
                        round_info["metrics"] = hist_entry.metrics
                rounds_data.append(round_info)
        except Exception as e:
            print(f"[summary] Warning: Could not extract round history: {e}")
    
    # Dataset information (CIFAR-10 based on task.py)
    dataset_info = {
        "name": "CIFAR-10",
        "type": "image_classification",
        "num_classes": 10,
        "description": "CIFAR-10 is a dataset of 60,000 32x32 color images in 10 classes",
        "source": "uoft-cs/cifar10",
        "partitioning": "IID (Independent and Identically Distributed)",
    }
    
    # Training configuration
    training_config = {
        "learning_rate": context.run_config.get("lr", 0.01),
        "local_epochs": context.run_config.get("local-epochs", 1),
        "fraction_train": context.run_config.get("fraction-train", 0.5),
        "batch_size": 32,
        "optimizer": "Adam",
        "loss_function": "CrossEntropyLoss",
    }
    
    # Model information
    model_info = {
        "architecture": "CNN",
        "description": "Simple CNN with 2 convolutional layers and 3 fully connected layers",
        "input_size": "32x32x3 (RGB images)",
        "output_size": 10,
    }
    
    summary = {
        "timestamp": int(time.time()),
        "num_server_rounds": num_server_rounds,
        "dataset": dataset_info,
        "model": model_info,
        "training_config": training_config,
        "global_metrics": global_metrics,
        "rounds": rounds_data,
        "storage": {
            "backend": os.getenv("STORAGE_BACKEND", "unknown"),
            "zero_g_root": artifact_id,
            "zero_g_indexer": os.getenv("ZEROG_INDEXER"),
        },
        "chain": {
            "rpc_url": rpc_url,
            "contract_address": contract_address,
            "last_round_id": last_round_id,
        },
    }

    # Write local file first
    out_path = Path("run_summary.json")
    summary_json_str = json.dumps(summary, indent=2)
    out_path.write_text(summary_json_str)
    print(f"[summary] Wrote {out_path} for LLM helper")

    # Upload summary data to 0G storage
    try:
        storage = get_storage_backend()
        # Write summary to a temporary file for upload
        temp_summary_path = Path("run_summary_data.json")
        temp_summary_path.write_text(summary_json_str)
        
        summary_root = storage.upload(str(temp_summary_path))
        
        # Update local run_summary.json with the root hash of the data
        summary["summary_data_root"] = summary_root
        summary["summary_data_filename"] = "run_summary_data.json"
        out_path.write_text(json.dumps(summary, indent=2))
        
        # Clean up temp file
        temp_summary_path.unlink()
        
        print(f"[summary] Uploaded summary data to 0G, root = {summary_root}")
    except Exception as e:
        print(f"[summary] Warning: Failed to upload summary to 0G: {e}")
        print("[summary] LLM helper will use local file only")


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # 1. FL config
    fraction_train: float = context.run_config["fraction-train"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["lr"]

    # 2. Model
    arrays = ArrayRecord(Net().state_dict())

    # 3. Strategy with callback to capture metrics
    # Clear previous round metrics
    global _round_metrics
    _round_metrics = []
    
    # Create a custom strategy that captures metrics
    class MetricsCapturingFedAvg(FedAvg):
        def aggregate_fit(self, server_round, results, failures):
            # Call parent aggregation
            aggregated = super().aggregate_fit(server_round, results, failures)
            
            # Capture metrics from this round
            round_metrics = {
                "round": server_round,
                "num_clients": len(results) if results else 0,
                "metrics": {}
            }
            
            # Extract metrics from client results
            if results:
                # Aggregate client metrics
                total_examples = 0
                total_train_loss = 0.0
                for fit_res in results:
                    # Try different ways to access metrics
                    metrics = None
                    if hasattr(fit_res, 'metrics'):
                        metrics = fit_res.metrics
                    elif hasattr(fit_res, 'get_metrics'):
                        metrics = fit_res.get_metrics()
                    elif isinstance(fit_res, tuple) and len(fit_res) > 1:
                        # Some Flower versions return tuples
                        metrics = fit_res[1] if isinstance(fit_res[1], dict) else None
                    
                    if metrics:
                        if isinstance(metrics, dict):
                            num_examples = metrics.get("num-examples", 0)
                            train_loss = metrics.get("train_loss", 0.0)
                        else:
                            # Try to convert to dict
                            try:
                                if hasattr(metrics, 'to_dict'):
                                    metrics = metrics.to_dict()
                                num_examples = metrics.get("num-examples", 0) if isinstance(metrics, dict) else 0
                                train_loss = metrics.get("train_loss", 0.0) if isinstance(metrics, dict) else 0.0
                            except:
                                num_examples = 0
                                train_loss = 0.0
                        
                        total_examples += num_examples
                        total_train_loss += train_loss * num_examples
                
                if total_examples > 0:
                    round_metrics["metrics"] = {
                        "avg_train_loss": total_train_loss / total_examples,
                        "total_examples": total_examples,
                    }
            
            _round_metrics.append(round_metrics)
            return aggregated
        
        def aggregate_evaluate(self, server_round, results, *args, **kwargs):
            # Call parent aggregation - use *args and **kwargs to handle different Flower versions
            # This matches whatever signature the parent class has
            aggregated = super().aggregate_evaluate(server_round, results, *args, **kwargs)
            
            # Extract failures if it was passed
            failures = args[0] if len(args) > 0 else kwargs.get('failures', None)
            
            # Update round metrics with evaluation results (append to existing, don't replace)
            # Data is ADDED to the existing round entry, not replaced
            if server_round <= len(_round_metrics) and len(_round_metrics) > 0:
                round_idx = server_round - 1
                if results:
                    total_examples = 0
                    total_eval_loss = 0.0
                    total_eval_acc = 0.0
                    for eval_res in results:
                        # Try different ways to access metrics
                        metrics = None
                        if hasattr(eval_res, 'metrics'):
                            metrics = eval_res.metrics
                        elif hasattr(eval_res, 'get_metrics'):
                            metrics = eval_res.get_metrics()
                        elif isinstance(eval_res, tuple) and len(eval_res) > 1:
                            metrics = eval_res[1] if isinstance(eval_res[1], dict) else None
                        
                        if metrics:
                            if isinstance(metrics, dict):
                                num_examples = metrics.get("num-examples", 0)
                                eval_loss = metrics.get("eval_loss", 0.0)
                                eval_acc = metrics.get("eval_acc", 0.0)
                            else:
                                try:
                                    if hasattr(metrics, 'to_dict'):
                                        metrics = metrics.to_dict()
                                    num_examples = metrics.get("num-examples", 0) if isinstance(metrics, dict) else 0
                                    eval_loss = metrics.get("eval_loss", 0.0) if isinstance(metrics, dict) else 0.0
                                    eval_acc = metrics.get("eval_acc", 0.0) if isinstance(metrics, dict) else 0.0
                                except:
                                    num_examples = 0
                                    eval_loss = 0.0
                                    eval_acc = 0.0
                            
                            total_examples += num_examples
                            total_eval_loss += eval_loss * num_examples
                            total_eval_acc += eval_acc * num_examples
                    
                    if total_examples > 0:
                        _round_metrics[round_idx]["metrics"].update({
                            "avg_eval_loss": total_eval_loss / total_examples,
                            "avg_eval_accuracy": total_eval_acc / total_examples,
                        })
            
            return aggregated
    
    strategy = MetricsCapturingFedAvg(fraction_train=fraction_train)

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

    # 6. Upload once to selected backend (0G / filecoin / dummy)
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

        rpc_url = os.getenv("RPC_URL", "http://127.0.0.1:8545")
        contract_address = os.getenv("CONTRACT_ADDRESS")
        owner_pk = os.getenv("OWNER_PRIVATE_KEY")
        server_pk = os.getenv("SERVER_PRIVATE_KEY")

        if not contract_address:
            raise RuntimeError("CONTRACT_ADDRESS missing in environment")

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

        # 8. Human-readable verification summary
        chain_env = os.getenv("CHAIN_ENV", "unknown")
        print_verification_summary(
            chain_env=chain_env,
            rpc_url=rpc_url,
            contract_address=contract_address,
            storage_root=storage_root,
        )

        # 9. Machine-readable summary for the 0G LLM helper
        write_run_summary(
            result=result,
            artifact_id=storage_root,
            rpc_url=rpc_url,
            contract_address=contract_address,
            last_round_id=next_round,
            num_server_rounds=num_rounds,
            context=context,
        )

    except Exception as e:
        print(f"[on-chain ERROR] {e}")
