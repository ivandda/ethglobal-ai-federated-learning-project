# onchain_logger.py

from typing import List
from web3 import Web3
import json
from pathlib import Path


class OnchainFLLogger:
    def __init__(
        self,
        rpc_url: str,
        contract_address: str,
        owner_private_key: str,
        server_private_key: str,
        artifact_path: str = "artifacts/contracts/FederatedLearningLog.sol/FederatedLearningLog.json",
    ) -> None:
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        assert self.w3.is_connected(), "Web3 is not connected"

        artifact = json.loads(Path(artifact_path).read_text())
        abi = artifact["abi"]

        self.contract = self.w3.eth.contract(
            address=self.w3.to_checksum_address(contract_address),
            abi=abi,
        )

        self.owner = self.w3.eth.account.from_key(owner_private_key)
        self.server = self.w3.eth.account.from_key(server_private_key)

    def _send_tx(self, account, tx):
        tx.update(
            {
                "nonce": self.w3.eth.get_transaction_count(account.address),
                "gas": 8000000,
                "maxFeePerGas": self.w3.to_wei("2", "gwei"),
                "maxPriorityFeePerGas": self.w3.to_wei("1", "gwei"),
                "chainId": self.w3.eth.chain_id,
            }
        )
        signed = account.sign_transaction(tx)
        tx_hash = self.w3.eth.send_raw_transaction(signed.rawTransaction)
        return self.w3.eth.wait_for_transaction_receipt(tx_hash)

    def ensure_client_registered(self, client_address: str) -> None:
        addr = self.w3.to_checksum_address(client_address)
        if self.contract.functions.isClient(addr).call():
            return
        tx = self.contract.functions.registerClient(addr).build_transaction(
            {"from": self.owner.address}
        )
        receipt = self._send_tx(self.owner, tx)
        print("Client registered on-chain:", addr, "tx:", receipt.transactionHash.hex())

    def record_round(
        self,
        round_id: int,
        model_hash: bytes,
        artifact_cid: str,
        client_addresses: List[str],
        samples: List[int],
        scores: List[int],
    ) -> None:
        addrs = [self.w3.to_checksum_address(a) for a in client_addresses]

        tx = self.contract.functions.recordRound(
            round_id,
            model_hash,
            artifact_cid,
            addrs,
            samples,
            scores,
        ).build_transaction(
            {
                "from": self.server.address,
            }
        )
        receipt = self._send_tx(self.server, tx)
        print(
            f"[on-chain] Round {round_id} recorded, tx: {receipt.transactionHash.hex()}"
        )
