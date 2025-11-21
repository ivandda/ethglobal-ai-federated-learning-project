from typing import Tuple
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms


class SimpleNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


def get_dataloaders_for_client(
    num_clients: int,
    cid: int,
    batch_size: int = 32,
) -> Tuple[DataLoader, DataLoader]:
    """Split MNIST into num_clients non-overlapping chunks and return train loader for client cid."""
    transform = transforms.Compose([transforms.ToTensor()])

    full_train = datasets.MNIST(
        "./data", train=True, download=True, transform=transform
    )
    full_test = datasets.MNIST(
        "./data", train=False, download=True, transform=transform
    )

    # split train dataset into equal chunks for clients
    # Ensure cid is in valid range
    cid = max(0, min(cid, num_clients - 1))
    
    subset_size = len(full_train) // num_clients
    if subset_size == 0:
        raise ValueError(f"Cannot split dataset: {len(full_train)} samples for {num_clients} clients")
    
    indices = list(range(len(full_train)))
    start = cid * subset_size
    end = (cid + 1) * subset_size if cid < num_clients - 1 else len(full_train)
    client_indices = indices[start:end]
    
    if len(client_indices) == 0:
        raise ValueError(f"Client {cid} got empty dataset (start={start}, end={end}, dataset_size={len(full_train)})")
    
    client_train = Subset(full_train, client_indices)

    train_loader = DataLoader(client_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(full_test, batch_size=128, shuffle=False)
    return train_loader, test_loader
