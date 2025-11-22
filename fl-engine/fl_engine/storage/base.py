from abc import ABC, abstractmethod

class DecentralizedStorage(ABC):
    @abstractmethod
    def upload(self, path: str) -> str:
        """Upload a file and return a CID / root hash"""
        pass
