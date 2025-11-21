from web3 import Web3
import json
from pathlib import Path

# 1) Connect to your running Hardhat node
w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:8545"))
assert w3.is_connected(), "Web3 is not connected to Hardhat node"

# 2) Load contract ABI from Hardhat artifacts
artifact_path = Path("artifacts/contracts/FederatedLearningLog.sol/FederatedLearningLog.json")
artifact = json.loads(artifact_path.read_text())
abi = artifact["abi"]

# 3) Use Hardhat accounts as owner and server (from hardhat node output)
OWNER_PRIVATE_KEY = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
SERVER_PRIVATE_KEY = "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d"

owner = w3.eth.account.from_key(OWNER_PRIVATE_KEY)
server = w3.eth.account.from_key(SERVER_PRIVATE_KEY)

# 4) Contract address (from deploy_local_node logs)
CONTRACT_ADDRESS = Web3.to_checksum_address("0x5FbDB2315678afecb367f032d93F642f64180aa3")
contract = w3.eth.contract(address=CONTRACT_ADDRESS, abi=abi)

def send_tx(account, tx):
    tx.update({
        "nonce": w3.eth.get_transaction_count(account.address),
        "gas": 8000000,
        "maxFeePerGas": w3.to_wei("2", "gwei"),
        "maxPriorityFeePerGas": w3.to_wei("1", "gwei"),
        "chainId": w3.eth.chain_id,
    })
    signed = account.sign_transaction(tx)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    return w3.eth.wait_for_transaction_receipt(tx_hash)

# 5) Pick a client address (any other Hardhat node account)
CLIENT_ADDRESS = Web3.to_checksum_address("0x3c44cdddb6a900fa2b585dd299e03d12fa4293bc")

# Owner registers client
tx = contract.functions.registerClient(CLIENT_ADDRESS).build_transaction({
    "from": owner.address,
})
receipt = send_tx(owner, tx)
print("Client registered, tx:", receipt.transactionHash.hex())
print("isClient:", contract.functions.isClient(CLIENT_ADDRESS).call())

# 6) Record a dummy round from the server
round_id = 1
model_hash = Web3.to_bytes(hexstr="0x" + "11" * 32)  # fake 32-byte hash
artifact_cid = "bafyFAKEcidForRound1"  # placeholder Filecoin CID
clients = [CLIENT_ADDRESS]
samples = [100]
scores = [123456]

tx = contract.functions.recordRound(
    round_id,
    model_hash,
    artifact_cid,
    clients,
    samples,
    scores,
).build_transaction({
    "from": server.address,
})
receipt = send_tx(server, tx)
print("Round recorded, tx:", receipt.transactionHash.hex())

# 7) Read back on-chain data
stored_round = contract.functions.rounds(round_id).call()
print("Stored round:", stored_round)

client_info = contract.functions.clientRounds(round_id, CLIENT_ADDRESS).call()
print("Client round info:", client_info)
