# On-Chain Federated Learning Verification

> **Proof of Concept** - Decentralized audit trail for federated learning via blockchain smart contracts and decentralized storage.

## Core Innovation

This POC demonstrates **on-chain verification** of federated learning training rounds through:

- **Smart Contract Verification**: Immutable audit log on blockchain (0G/EVM) recording:
  - Training round metadata (model hashes, timestamps)
  - Client participation and contributions
  - Model artifact references (CIDs)
  
- **Decentralized Storage**: Model artifacts stored on **0G Storage** or **Filecoin**:
  - Model weights stored off-chain (too large for blockchain)
  - On-chain CIDs provide verifiable references
  - Enables integrity verification and retrieval

## Architecture

TODO

## ✨ Key Features

- **Immutable Audit Trail**: Every training round permanently recorded on-chain
- **Verifiable Integrity**: Model hashes enable verification of stored artifacts
- **Contribution Tracking**: Client participation and contribution scores logged
- **Decentralized Storage**: Flexible backend (0G Storage / Filecoin)
- **LLM Integration**: Query training results via 0G compute network
- **Production Framework**: Built on [Flower](https://flower.ai/) - industry-standard federated learning

## Quick Start

### Prerequisites

- Python 3.12+
- Node.js & pnpm (for smart contract deployment)
- Environment variables configured (see `.env.example`)

### Installation

```bash
# Install Python dependencies
pip install -r requirements.txt
cd fl-engine && pip install -e .

# Install Node dependencies
pnpm install
```

### Run Training

```bash
cd fl-engine
flwr run .
```

This will:
1. Train a federated learning model (CIFAR-10 example)
2. Upload model to decentralized storage (0G/Filecoin)
3. Record training round on-chain via smart contract
4. Generate verification summary

### Query Results

```bash
pnpm ask-run "How did the model improve over training rounds?"
```

Uses 0G compute network to query training results from decentralized storage.

## 📋 What's On-Chain vs Off-Chain

| Component | Location | Reason |
|-----------|----------|--------|
| Round metadata | ✅ On-chain | Immutable audit log |
| Model hash | ✅ On-chain | Integrity verification |
| Client contributions | ✅ On-chain | Participation tracking |
| Model weights | ❌ Off-chain (0G/Filecoin) | Too large for blockchain |
| Training data | ❌ Client-side | Privacy-preserving |
| Training process | ❌ Flower framework | Off-chain computation |

## Tech Stack

- **Federated Learning**: [Flower](https://flower.ai/) - Production-ready framework
- **Blockchain**: 0G Network / EVM-compatible chains
- **Smart Contracts**: Solidity (Hardhat)
- **Storage**: 0G Storage / Filecoin
- **ML Framework**: PyTorch
- **LLM Integration**: 0G Compute Network

## Project Structure

```
.
├── contracts/              # Solidity smart contracts
│   └── FederatedLearningLog.sol
├── fl-engine/             # Flower federated learning app
│   ├── fl_engine/
│   │   ├── server_app.py  # Server coordination
│   │   ├── client_app.py  # Client training
│   │   ├── onchain_logger.py  # Blockchain integration
│   │   └── storage/       # Decentralized storage backends
│   └── pyproject.toml
├── scripts/               # Deployment & utilities
└── artifacts/             # Compiled contract ABIs
```

## 🎓 Training Example

The included example trains a CNN on **CIFAR-10** using federated learning:
- Simple but demonstrates full pipeline
- Uses Flower's production-ready infrastructure
- Supports multiple clients, rounds, and aggregation strategies
- Easily extensible to other datasets/models

## 🔍 Verification

After training, verify on-chain records:
- Check smart contract events for round records
- Verify model hash matches stored artifact
- Download model from decentralized storage using CID
- Query training metrics via LLM integration


## ⚠️ POC Disclaimer

This is a **Proof of Concept** demonstrating:
- On-chain verification of federated learning
- Integration between FL framework and blockchain
- Decentralized storage for model artifacts

Not production-ready. Some hardcoded values and simplified implementations exist for hackathon demonstration purposes.

---

**Built for ETHGlobal AI Hackathon** | Combining federated learning, blockchain, and decentralized storage
