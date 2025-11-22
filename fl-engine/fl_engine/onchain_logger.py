from dataclasses import dataclass
from typing import List

from web3 import Web3


@dataclass
class OnchainFLLogger:
    rpc_url: str
    contract_address: str
    owner_private_key: str
    server_private_key: str
    abi: list

    def __post_init__(self) -> None:
        self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))
        if not self.w3.is_connected():
            raise RuntimeError("Web3 not connected")

        self.contract = self.w3.eth.contract(
            address=self.w3.to_checksum_address(self.contract_address),
            abi=self.abi,
        )

        self.owner = self.w3.eth.account.from_key(self.owner_private_key)
        self.server = self.w3.eth.account.from_key(self.server_private_key)

    def get_latest_round_id(self) -> int:
        """Get the current latestRoundId from the contract."""
        try:
            return self.contract.functions.latestRoundId().call()
        except Exception as e:
            print(f"[on-chain] Warning: Could not get latestRoundId: {e}")
            return 0  # Default to 0 if contract doesn't have this function or on error

    def ensure_client_registered(self, client_address: str) -> None:
        """Register a client if not already registered."""
        addr = self.w3.to_checksum_address(client_address)
        try:
            is_registered = self.contract.functions.isClient(addr).call()
            if not is_registered:
                tx = self.contract.functions.registerClient(addr).build_transaction({
                    "from": self.owner.address,
                    "nonce": self.w3.eth.get_transaction_count(self.owner.address),
                    "gas": 100_000,
                    "maxFeePerGas": self.w3.to_wei("4", "gwei"),
                    "maxPriorityFeePerGas": self.w3.to_wei("2", "gwei"),
                })
                signed = self.owner.sign_transaction(tx)
                raw_tx = getattr(signed, 'raw_transaction', None) or getattr(signed, 'rawTransaction', None)
                if raw_tx:
                    tx_hash = self.w3.eth.send_raw_transaction(raw_tx)
                    receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
                    tx_hash_hex = getattr(receipt, 'transactionHash', None) or getattr(receipt, 'transaction_hash', None)
                    if tx_hash_hex:
                        tx_hash_str = tx_hash_hex.hex() if hasattr(tx_hash_hex, 'hex') else str(tx_hash_hex)
                    else:
                        tx_hash_str = tx_hash.hex() if hasattr(tx_hash, 'hex') else str(tx_hash)
                    print(f"[on-chain] Registered client {addr}, tx: {tx_hash_str}")
        except Exception as e:
            print(f"[on-chain] Warning: Could not register client {addr}: {e}")

    # For now, a very simple "round recorded" call.
    # You can match this to your existing Solidity contract signature.
    def record_round(
        self,
        round_id: int = None,  # If None, will use next sequential ID
        model_hash: bytes = None,
        artifact_cid: str = None,
        client_addresses: List[str] = None,
        samples: List[int] = None,
        scores: List[int] = None,
    ) -> None:
        # Get the next sequential round ID if not provided
        if round_id is None:
            round_id = self.get_latest_round_id() + 1
        
        # Validate required parameters
        if model_hash is None:
            raise ValueError("model_hash is required")
        if artifact_cid is None or len(artifact_cid) == 0:
            raise ValueError("artifact_cid is required and cannot be empty")
        if client_addresses is None:
            client_addresses = []
        if samples is None:
            samples = []
        if scores is None:
            scores = []
        
        # Validate that round_id is sequential
        latest = self.get_latest_round_id()
        if round_id != latest + 1:
            raise ValueError(f"Round ID must be {latest + 1} (got {round_id}). Round IDs must increment sequentially.")
        
        # Validate arrays are not empty (contract requirement)
        if not client_addresses or len(client_addresses) == 0:
            raise ValueError("client_addresses cannot be empty. Contract requires at least one client.")
        
        if len(client_addresses) != len(samples) or len(client_addresses) != len(scores):
            raise ValueError("client_addresses, samples, and scores must have the same length.")
        
        # Ensure all clients are registered
        for addr in client_addresses:
            self.ensure_client_registered(addr)
        
        # Convert addresses to checksum format
        checksum_addresses = [self.w3.to_checksum_address(addr) for addr in client_addresses]
        
        tx = self.contract.functions.recordRound(
            round_id,
            model_hash,
            artifact_cid,
            checksum_addresses,
            samples,
            scores,
        ).build_transaction(
            {
                "from": self.server.address,
                "nonce": self.w3.eth.get_transaction_count(self.server.address),
                "gas": 5_000_000,
                "maxFeePerGas": self.w3.to_wei("4", "gwei"),
                "maxPriorityFeePerGas": self.w3.to_wei("2", "gwei"),
            }
        )
        signed = self.server.sign_transaction(tx)
        # Handle both old and new Web3.py API (rawTransaction vs raw_transaction)
        raw_tx = getattr(signed, 'raw_transaction', None) or getattr(signed, 'rawTransaction', None)
        if raw_tx is None:
            raise AttributeError("Signed transaction has neither 'raw_transaction' nor 'rawTransaction' attribute")
        tx_hash = self.w3.eth.send_raw_transaction(raw_tx)
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
        
        # Check if transaction reverted (status can be 0, '0x0', or False)
        status = getattr(receipt, 'status', None)
        if status == 0 or status == '0x0' or status is False:
            raise RuntimeError(f"Transaction reverted. Receipt status: {status}. Check Hardhat node logs for revert reason.")
        
        # Handle both old and new Web3.py API (transactionHash vs transaction_hash)
        tx_hash_hex = getattr(receipt, 'transactionHash', None) or getattr(receipt, 'transaction_hash', None)
        if tx_hash_hex:
            tx_hash_str = tx_hash_hex.hex() if hasattr(tx_hash_hex, 'hex') else str(tx_hash_hex)
        else:
            tx_hash_str = tx_hash.hex() if hasattr(tx_hash, 'hex') else str(tx_hash)
        print(f"[on-chain] Round {round_id} recorded, tx: {tx_hash_str}")