import subprocess
import re
from .base import DecentralizedStorage

class FilecoinPinStorage(DecentralizedStorage):
    def upload(self, path: str) -> str:
        cmd = ["filecoin-pin", "add", path, "--auto-fund"]

        result = subprocess.run(cmd, capture_output=True, text=True)
        print("[filecoin-pin stdout]", result.stdout)
        print("[filecoin-pin stderr]", result.stderr)

        if result.returncode != 0:
            raise RuntimeError(
                f"filecoin-pin failed with exit code {result.returncode}: {result.stderr}"
            )

        m = re.search(r"Root CID:\s+(\S+)", result.stdout)
        if not m:
            raise RuntimeError("Could not find Root CID in filecoin-pin output")

        return m.group(1)
