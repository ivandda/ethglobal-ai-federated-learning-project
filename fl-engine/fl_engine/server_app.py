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
    
    # Extract metrics from Flower's result object
    rounds_data = []
    
    # DEBUG: Print what attributes the result object actually has
    print(f"[summary DEBUG] Result type: {type(result)}")
    print(f"[summary DEBUG] Result attributes: {[x for x in dir(result) if not x.startswith('_')]}")
    
    # Access the actual attributes Flower uses (visible in logs)
    train_metrics_dict = getattr(result, 'train_metrics_clientapp', {})
    eval_metrics_dict = getattr(result, 'evaluate_metrics_clientapp', {})
    
    # DEBUG: Print what we got
    print(f"[summary DEBUG] train_metrics_dict type: {type(train_metrics_dict)}")
    print(f"[summary DEBUG] train_metrics_dict: {train_metrics_dict}")
    print(f"[summary DEBUG] eval_metrics_dict type: {type(eval_metrics_dict)}")
    print(f"[summary DEBUG] eval_metrics_dict: {eval_metrics_dict}")
    
    print(f"[summary] Found {len(train_metrics_dict)} train rounds, {len(eval_metrics_dict)} eval rounds")
    
    # Combine train and eval metrics by round
    all_rounds = set()
    if train_metrics_dict:
        all_rounds.update(train_metrics_dict.keys())
    if eval_metrics_dict:
        all_rounds.update(eval_metrics_dict.keys())
    
    print(f"[summary DEBUG] all_rounds: {all_rounds}")
    
    for round_num in sorted(all_rounds):
        round_data = {
            "round": int(round_num),
            "metrics": {}
        }
        
        # Extract train metrics
        if round_num in train_metrics_dict:
            train_m = train_metrics_dict[round_num]
            print(f"[summary DEBUG] Round {round_num} train_m type: {type(train_m)}, value: {train_m}")
            
            # Check if it's a MetricRecord object
            if hasattr(train_m, '__dict__'):
                print(f"[summary DEBUG] train_m attributes: {train_m.__dict__}")
            
            if isinstance(train_m, dict):
                # Convert scientific notation strings to floats
                for key, value in train_m.items():
                    if isinstance(value, str):
                        try:
                            round_data["metrics"][key] = float(value)
                        except:
                            round_data["metrics"][key] = value
                    else:
                        round_data["metrics"][key] = value
            elif hasattr(train_m, 'metrics_dict'):  # Might be a MetricRecord
                for key, value in train_m.metrics_dict.items():
                    if isinstance(value, str):
                        try:
                            round_data["metrics"][key] = float(value)
                        except:
                            round_data["metrics"][key] = value
                    else:
                        round_data["metrics"][key] = value
            elif hasattr(train_m, '__iter__') and not isinstance(train_m, str):
                # Try to iterate if it's iterable
                try:
                    for key, value in train_m.items():
                        if isinstance(value, str):
                            try:
                                round_data["metrics"][key] = float(value)
                            except:
                                round_data["metrics"][key] = value
                        else:
                            round_data["metrics"][key] = value
                except:
                    pass
        
        # Extract eval metrics
        if round_num in eval_metrics_dict:
            eval_m = eval_metrics_dict[round_num]
            print(f"[summary DEBUG] Round {round_num} eval_m type: {type(eval_m)}, value: {eval_m}")
            
            if hasattr(eval_m, '__dict__'):
                print(f"[summary DEBUG] eval_m attributes: {eval_m.__dict__}")
            
            if isinstance(eval_m, dict):
                # Convert scientific notation strings to floats
                for key, value in eval_m.items():
                    if isinstance(value, str):
                        try:
                            round_data["metrics"][key] = float(value)
                        except:
                            round_data["metrics"][key] = value
                    else:
                        round_data["metrics"][key] = value
            elif hasattr(eval_m, 'metrics_dict'):  # Might be a MetricRecord
                for key, value in eval_m.metrics_dict.items():
                    if isinstance(value, str):
                        try:
                            round_data["metrics"][key] = float(value)
                        except:
                            round_data["metrics"][key] = value
                    else:
                        round_data["metrics"][key] = value
            elif hasattr(eval_m, '__iter__') and not isinstance(eval_m, str):
                # Try to iterate if it's iterable
                try:
                    for key, value in eval_m.items():
                        if isinstance(value, str):
                            try:
                                round_data["metrics"][key] = float(value)
                            except:
                                round_data["metrics"][key] = value
                        else:
                            round_data["metrics"][key] = value
                except:
                    pass
        
        print(f"[summary DEBUG] Round {round_num} final metrics: {round_data['metrics']}")
        rounds_data.append(round_data)
    
    print(f"[summary] Extracted {len(rounds_data)} rounds with complete metrics")
    
    # Calculate improvement metrics if we have data
    global_metrics = {}
    if len(rounds_data) > 1:
        first_round = rounds_data[0]["metrics"]
        last_round = rounds_data[-1]["metrics"]
        
        first_acc = first_round.get("eval_acc", 0)
        last_acc = last_round.get("eval_acc", 0)
        first_loss = first_round.get("train_loss", 0)
        last_loss = last_round.get("train_loss", 0)
        
        if first_acc > 0 and last_acc > 0:
            global_metrics["accuracy_improvement"] = {
                "initial_accuracy": first_acc,
                "final_accuracy": last_acc,
                "absolute_improvement": last_acc - first_acc,
                "relative_improvement_pct": ((last_acc - first_acc) / first_acc * 100)
            }
        
        if first_loss > 0 and last_loss > 0:
            global_metrics["loss_reduction"] = {
                "initial_loss": first_loss,
                "final_loss": last_loss,
                "absolute_reduction": first_loss - last_loss,
                "relative_reduction_pct": ((first_loss - last_loss) / first_loss * 100)
            }
    
    # Dataset information (CIFAR-10 based on task.py)
    dataset_info = {
        "name": "CIFAR-10",
        "type": "image_classification",
        "num_classes": 10,
        "classes": ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
        "description": "CIFAR-10 is a dataset of 60,000 32x32 color images in 10 classes, with 6,000 images per class",
        "source": "uoft-cs/cifar10",
        "partitioning": "IID (Independent and Identically Distributed)",
        "image_size": "32x32x3 (RGB)",
    }
    
    # Training configuration
    training_config = {
        "learning_rate": context.run_config.get("lr", 0.01),
        "local_epochs": context.run_config.get("local-epochs", 1),
        "fraction_train": context.run_config.get("fraction-train", 0.5),
        "batch_size": 32,
        "optimizer": "Adam",
        "loss_function": "CrossEntropyLoss",
        "num_server_rounds": num_server_rounds,
    }
    
    # Model information
    model_info = {
        "architecture": "CNN",
        "description": "Simple CNN with 2 convolutional layers and 3 fully connected layers",
        "layers": [
            "Conv2d(3, 6, kernel_size=5)",
            "MaxPool2d(2, 2)",
            "Conv2d(6, 16, kernel_size=5)",
            "MaxPool2d(2, 2)",
            "Linear(16*5*5, 120)",
            "Linear(120, 84)",
            "Linear(84, 10)"
        ],
        "input_size": "32x32x3 (RGB images)",
        "output_size": 10,
        "activation": "ReLU",
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
    # Clear previous round metrics (use global)
    global _round_metrics
    _round_metrics = []  # Reset the list
    
    # Create a custom strategy that captures metrics
    class MetricsCapturingFedAvg(FedAvg):
        def aggregate_fit(self, server_round, results, failures):
            # Access global variable first
            global _round_metrics
            
            # Call parent aggregation first
            aggregated = super().aggregate_fit(server_round, results, failures)
            
            # Capture metrics from this round
            round_metrics = {
                "round": server_round,
                "num_clients": len(results) if results else 0,
                "metrics": {},
                "clients": []  # Track individual client contributions
            }
            
            # Extract metrics from client results
            if results:
                total_examples = 0
                total_train_loss = 0.0
                client_contributions = []
                
                print(f"[metrics] Round {server_round} FIT: Processing {len(results)} client results")
                
                for i, fit_res in enumerate(results):
                    # Flower returns FitRes objects - check what attributes it has
                    metrics_dict = {}
                    num_examples = 0
                    train_loss = 0.0
                    
                    # Debug: print what we're working with (only first client)
                    if i == 0:
                        print(f"[metrics] FitRes type: {type(fit_res)}")
                        print(f"[metrics] FitRes dir: {[x for x in dir(fit_res) if not x.startswith('_')][:10]}")
                    
                    # Try multiple ways to extract metrics
                    # Method 1: Direct metrics attribute
                    if hasattr(fit_res, 'metrics') and fit_res.metrics:
                        try:
                            if hasattr(fit_res.metrics, 'to_dict'):
                                metrics_dict = fit_res.metrics.to_dict()
                            elif isinstance(fit_res.metrics, dict):
                                metrics_dict = fit_res.metrics
                        except Exception as e:
                            print(f"[metrics] Error extracting metrics (method 1): {e}")
                    
                    # Method 2: Check if it's a tuple (parameters, metrics)
                    if not metrics_dict and isinstance(fit_res, tuple) and len(fit_res) > 1:
                        try:
                            if hasattr(fit_res[1], 'to_dict'):
                                metrics_dict = fit_res[1].to_dict()
                            elif isinstance(fit_res[1], dict):
                                metrics_dict = fit_res[1]
                        except Exception as e:
                            print(f"[metrics] Error extracting metrics (method 2): {e}")
                    
                    # Method 3: Check if metrics are in the aggregated result
                    if not metrics_dict and hasattr(aggregated, 'metrics'):
                        try:
                            if hasattr(aggregated.metrics, 'to_dict'):
                                metrics_dict = aggregated.metrics.to_dict()
                        except:
                            pass
                    
                    # Get values from metrics dict
                    num_examples = metrics_dict.get("num-examples", 0)
                    train_loss = metrics_dict.get("train_loss", 0.0)
                    
                    if i == 0:
                        print(f"[metrics] Extracted - samples: {num_examples}, loss: {train_loss}, dict keys: {list(metrics_dict.keys())}")
                    
                    # Track client contribution (even if metrics are 0, we know a client participated)
                    client_contributions.append({
                        "samples": num_examples if num_examples > 0 else 100,  # Default estimate
                        "train_loss": train_loss,
                        "contribution_weight": num_examples if num_examples > 0 else 100
                    })
                    if num_examples > 0:
                        total_examples += num_examples
                        total_train_loss += train_loss * num_examples
                
                # Store aggregated metrics - use aggregated result if we have it
                if hasattr(aggregated, 'metrics') and aggregated.metrics:
                    try:
                        agg_metrics = aggregated.metrics.to_dict() if hasattr(aggregated.metrics, 'to_dict') else {}
                        if 'train_loss' in agg_metrics:
                            round_metrics["metrics"]["avg_train_loss"] = agg_metrics["train_loss"]
                    except:
                        pass
                
                if total_examples > 0:
                    round_metrics["metrics"]["avg_train_loss"] = total_train_loss / total_examples
                    round_metrics["metrics"]["total_examples"] = total_examples
                    round_metrics["metrics"]["num_participating_clients"] = len(client_contributions)
                
                round_metrics["clients"] = client_contributions
                print(f"[metrics] Round {server_round} FIT captured: {round_metrics['metrics']}")
            
            _round_metrics.append(round_metrics)
            print(f"[metrics] Total rounds in history: {len(_round_metrics)}")
            return aggregated
        
        def aggregate_evaluate(self, server_round, results, *args, **kwargs):
            # Call parent aggregation - use *args and **kwargs to handle different Flower versions
            # This matches whatever signature the parent class has
            aggregated = super().aggregate_evaluate(server_round, results, *args, **kwargs)
            
            # Access global variable
            global _round_metrics
            
            # Extract failures if it was passed
            failures = args[0] if len(args) > 0 else kwargs.get('failures', None)
            
            print(f"[metrics] Round {server_round} evaluation: {len(results) if results else 0} results, {len(_round_metrics)} rounds in history")
            
            # Update round metrics with evaluation results (append to existing, don't replace)
            # Data is ADDED to the existing round entry, not replaced
            if server_round <= len(_round_metrics) and len(_round_metrics) > 0:
                round_idx = server_round - 1
                if results:
                    total_examples = 0
                    total_eval_loss = 0.0
                    total_eval_acc = 0.0
                    
                    for i, eval_res in enumerate(results):
                        # Extract metrics from EvaluateRes
                        metrics_dict = {}
                        
                        # Debug first result
                        if i == 0:
                            print(f"[metrics] EvaluateRes type: {type(eval_res)}, has metrics: {hasattr(eval_res, 'metrics')}")
                            if hasattr(eval_res, 'metrics'):
                                print(f"[metrics] Eval metrics type: {type(eval_res.metrics)}")
                        
                        if hasattr(eval_res, 'metrics') and eval_res.metrics:
                            try:
                                if hasattr(eval_res.metrics, 'to_dict'):
                                    metrics_dict = eval_res.metrics.to_dict()
                                elif isinstance(eval_res.metrics, dict):
                                    metrics_dict = eval_res.metrics
                                elif hasattr(eval_res.metrics, 'metrics'):  # Nested?
                                    if hasattr(eval_res.metrics.metrics, 'to_dict'):
                                        metrics_dict = eval_res.metrics.metrics.to_dict()
                                    elif isinstance(eval_res.metrics.metrics, dict):
                                        metrics_dict = eval_res.metrics.metrics
                            except Exception as e:
                                print(f"[metrics] Warning extracting eval metrics: {e}")
                        
                        if i == 0:
                            print(f"[metrics] First eval result - dict: {metrics_dict}")
                        
                        num_examples = metrics_dict.get("num-examples", 0)
                        eval_loss = metrics_dict.get("eval_loss", 0.0)
                        eval_acc = metrics_dict.get("eval_acc", 0.0)
                        
                        total_examples += num_examples
                        total_eval_loss += eval_loss * num_examples
                        total_eval_acc += eval_acc * num_examples
                    
                    if total_examples > 0:
                        eval_metrics = {
                            "avg_eval_loss": total_eval_loss / total_examples,
                            "avg_eval_accuracy": total_eval_acc / total_examples,
                        }
                        _round_metrics[round_idx]["metrics"].update(eval_metrics)
                        print(f"[metrics] Round {server_round} eval metrics added: {eval_metrics}")
                        
                        # Calculate improvement (simple: compare to previous round)
                        if round_idx > 0 and "avg_eval_accuracy" in _round_metrics[round_idx - 1].get("metrics", {}):
                            prev_acc = _round_metrics[round_idx - 1]["metrics"]["avg_eval_accuracy"]
                            curr_acc = eval_metrics["avg_eval_accuracy"]
                            improvement = curr_acc - prev_acc
                            eval_metrics["accuracy_improvement"] = improvement
                            eval_metrics["accuracy_improvement_pct"] = (improvement / prev_acc * 100) if prev_acc > 0 else 0.0
                            _round_metrics[round_idx]["metrics"].update(eval_metrics)
                            print(f"[metrics] Round {server_round} improvement: {improvement:.4f} ({eval_metrics['accuracy_improvement_pct']:.2f}%)")
                    else:
                        print(f"[metrics] Round {server_round} evaluation: No valid metrics (total_examples=0)")
            
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

        # Calculate actual client contributions from collected metrics
        # Note: _round_metrics is already declared as global at the top of main()
        total_samples = 0
        total_contribution_score = 0
        
        # Sum samples and calculate contribution scores from all rounds
        for round_metrics in _round_metrics:
            if "clients" in round_metrics:
                for client_data in round_metrics["clients"]:
                    client_samples = client_data.get("samples", 0)
                    total_samples += client_samples
                    
                    # Calculate contribution score based on samples and training quality
                    # Score = samples * (1 + accuracy_improvement_factor)
                    # For now, use samples as base score, can be enhanced with accuracy metrics
                    train_loss = client_data.get("train_loss", 0.0)
                    # Lower loss = better contribution, so invert it (use 1/loss or similar)
                    # To avoid division by zero, use: score = samples * max(1, 10 - loss)
                    loss_factor = max(1.0, 10.0 - train_loss) if train_loss > 0 else 1.0
                    contribution_score = int(client_samples * loss_factor)
                    total_contribution_score += contribution_score
        
        # Fallback if no metrics collected (shouldn't happen, but safety check)
        if total_samples == 0:
            print("[on-chain WARNING] No client metrics found, using fallback values")
            total_samples = 100
            total_contribution_score = 100000
        
        print(f"[on-chain] Recording round with {total_samples} total samples, score: {total_contribution_score}")

        logger.record_round(
            round_id=next_round,
            model_hash=model_hash,
            artifact_cid=storage_root or "NO-STORAGE",
            client_addresses=[server_addr],
            samples=[total_samples],
            scores=[total_contribution_score],
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
