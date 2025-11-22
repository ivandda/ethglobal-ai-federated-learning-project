from abc import ABC, abstractmethod

class DecentralizedStorage(ABC):
    @abstractmethod
    def upload(self, path: str) -> str:
        """Upload a file and return a CID / root hash"""
        pass
    
    @abstractmethod
    def download(self, root_hash: str, filename: str) -> bytes:
        """Download a file from storage using root hash and filename, return file contents as bytes"""
        pass