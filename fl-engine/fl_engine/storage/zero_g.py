import os
import subprocess
import re
from .base import DecentralizedStorage

class ZeroGStorage(DecentralizedStorage):
    def __init__(self):
        self.rpc = os.getenv("ZEROG_RPC", "https://evmrpc-testnet.0g.ai/")
        self.key = os.getenv("ZEROG_PRIVATE_KEY")
        self.indexer = os.getenv(
            "ZEROG_INDEXER",
            "https://indexer-storage-testnet-turbo.0g.ai/",
        )
        # 👇 This is the important line
        self.cli = os.getenv("ZEROG_CLI", "0g-storage-client")

        if not self.key:
            raise ValueError("ZEROG_PRIVATE_KEY must be set in .env")

    def upload(self, path: str) -> str:
        cmd = [
            self.cli,
            "upload",
            "--url", self.rpc,
            "--key", self.key,
            "--indexer", self.indexer,
            "--file", path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        print("[0g stdout]", result.stdout)
        print("[0g stderr]", result.stderr)

        if result.returncode != 0:
            raise RuntimeError("0G upload failed")

        # Try to extract the final "file uploaded" root line
        # Example line:
        # INFO ... file uploaded, root = 0x09d2ab...
        m = re.search(r"file uploaded,\s*root\s*=\s*(0x[0-9a-fA-F]+)", result.stderr)
        if not m:
            # Fallback: match any "root=0x..." in stderr
            m = re.search(r"root\s*=\s*(0x[0-9a-fA-F]+)", result.stderr)

        if not m:
            raise RuntimeError("Could not extract 0G Root Hash")

        root_hash = m.group(1)
        print(f"[0g] Uploaded {path}, root hash = {root_hash}")
        return root_hash
