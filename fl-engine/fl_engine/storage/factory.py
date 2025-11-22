import os
from .filecoin_pin import FilecoinPinStorage
from .zero_g import ZeroGStorage

def get_storage_backend():
    backend = os.getenv("STORAGE_BACKEND", "filecoin").lower()

    if backend == "filecoin":
        return FilecoinPinStorage()
    elif backend == "0g" or backend == "zero-g":
        return ZeroGStorage()
    else:
        raise ValueError(f"Unknown storage backend: {backend}")
