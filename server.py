from typing import Dict, List, Tuple
import hashlib

import numpy as np
import flwr as fl

from fl_client import FlowerClient
from onchain_logger import OnchainFLLogger
from flwr.common import Context
from flwr.client import Client


def node_id_to_index(node_id: str, num_clients: int) -> int:
    """Convert a node_id string to a deterministic client index in [0, num_clients-1]."""
    # Use SHA256 hash for deterministic mapping
    hash_obj = hashlib.sha256(node_id.encode())
    hash_int = int(hash_obj.hexdigest(), 16)
    return hash_int % num_clients 

# Pool of Ethereum addresses to assign to Flower clients (Hardhat dev node accounts)
CLIENT_ADDRESS_POOL: List[str] = [
    "0x3c44cdddb6a900fa2b585dd299e03d12fa4293bc",  # Account #2
    "0x90f79bf6eb2c4f870365e785982e1f101e93b906",  # Account #3
    # add more if you simulate more clients
]

NUM_SIM_CLIENTS = len(CLIENT_ADDRESS_POOL)


def hash_parameters(parameters) -> bytes:
    """Hash model parameters. Handles both Parameters object and NDArrays list."""
    m = hashlib.sha256()
    
    # Convert Parameters object to NDArrays if needed
    if isinstance(parameters, fl.common.Parameters):
        param_list = fl.common.parameters_to_ndarrays(parameters)
    # Handle list of NDArrays (already converted or older API)
    elif isinstance(parameters, (list, tuple)):
        param_list = parameters
    else:
        raise TypeError(f"Unexpected parameters type: {type(parameters)}")
    
    for p in param_list:
        arr = np.asarray(p)
        m.update(arr.tobytes())
    return m.digest()  # 32 bytes, matches bytes32 in Solidity


class LoggingFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, logger: OnchainFLLogger, **kwargs) -> None:
        super().__init__(**kwargs)
        self.logger = logger

        # maps Flower client id (e.g. "13094099085729097055") -> Ethereum address
        self.cid_to_eth: Dict[str, str] = {}

        # pool of ETH addresses we can assign
        self._eth_pool: List[str] = CLIENT_ADDRESS_POOL.copy()

    def _get_eth_address_for_cid(self, cid: str) -> str:
        # Return existing mapping if we have one
        if cid in self.cid_to_eth:
            return self.cid_to_eth[cid]

        # Otherwise assign a new ETH address from the pool
        if not self._eth_pool:
            raise RuntimeError("No more ETH addresses available for new clients")

        eth_addr = self._eth_pool.pop(0)
        self.cid_to_eth[cid] = eth_addr
        print(f"[mapping] Assigned ETH address {eth_addr} to Flower cid {cid}")
        return eth_addr


    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures,
    ):
        # First, do the normal FedAvg aggregation
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is None:
            return None, aggregated_metrics

        clients_eth: List[str] = []
        samples: List[int] = []
        scores: List[int] = []
        SCALE = 10**6

        for client_proxy, fit_res in results:
            cid = client_proxy.cid  # now something like "client:1", etc.
            metrics = fit_res.metrics or {}
            loss_before = float(metrics.get("loss_before", 0.0))
            loss_after = float(metrics.get("loss_after", loss_before))
            num_samples = int(metrics.get("num_samples", fit_res.num_examples))

            delta = max(0.0, loss_before - loss_after)
            score = int(delta * num_samples * SCALE)

            eth_addr = self._get_eth_address_for_cid(cid)
            clients_eth.append(eth_addr)
            samples.append(num_samples)
            scores.append(score)

        # Hash global model parameters
        model_hash = hash_parameters(aggregated_parameters)
        artifact_cid = f"bafyDUMMYcid-round-{server_round}"

        # ---- Safe on-chain logging ----
        try:
            # Register clients (if not already)
            for eth_addr in clients_eth:
                self.logger.ensure_client_registered(eth_addr)

            # Record the round
            self.logger.record_round(
                round_id=server_round,
                model_hash=model_hash,
                artifact_cid=artifact_cid,
                client_addresses=clients_eth,
                samples=samples,
                scores=scores,
            )
        except Exception as e:
            print(f"[on-chain ERROR] Failed to log round {server_round}: {e!r}")

        return aggregated_parameters, aggregated_metrics



def client_fn(context: Context) -> Client:
    """Create a Flower client instance.

    Flower now passes a Context; we can use its 'node_id' as a stable ID.
    """
    node_id_str = str(context.node_id)
    client_index = node_id_to_index(node_id_str, NUM_SIM_CLIENTS)
    
    client_id = node_id_str
    num_clients = NUM_SIM_CLIENTS

    numpy_client = FlowerClient(client_id, num_clients=num_clients, client_index=client_index)
    return numpy_client.to_client()



def main():
    # Create logger wired to your Hardhat dev node / contract
    logger = OnchainFLLogger(
        rpc_url="http://127.0.0.1:8545",
        contract_address="0x5FbDB2315678afecb367f032d93F642f64180aa3",  # from deploy_local_node
        owner_private_key="0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",  # Account #0
        server_private_key="0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d",  # Account #1
    )

    strategy = LoggingFedAvg(logger=logger, fraction_fit=1.0)

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_SIM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=2),
        strategy=strategy,
    )



if __name__ == "__main__":
    main()
