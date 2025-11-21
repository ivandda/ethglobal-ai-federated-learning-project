from typing import Dict, Tuple, List
import torch
from torch import nn
from torch.utils.data import DataLoader
import flwr as fl

from model_and_data import SimpleNet, get_dataloaders_for_client


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        total_samples += x.size(0)
    return total_loss / total_samples


def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = loss_fn(logits, y)
            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
    return total_loss / total_samples, correct / total_samples


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid: str, num_clients: int, client_index: int) -> None:
        self.cid = cid
        self.model = SimpleNet().to(DEVICE)
        self.loss_fn = nn.CrossEntropyLoss()
        # Ensure client_index is in valid range
        client_index = max(0, min(client_index, num_clients - 1))
        
        self.train_loader, self.test_loader = get_dataloaders_for_client(
            num_clients, client_index
        )

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List):
        state_dict = self.model.state_dict()
        for (k, v), p in zip(state_dict.items(), parameters):
            state_dict[k] = torch.tensor(p)
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        # loss BEFORE training
        loss_before, _ = evaluate(self.model, self.train_loader, self.loss_fn)

        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        train_loss = train_one_epoch(
            self.model, self.train_loader, optimizer, self.loss_fn
        )

        # loss AFTER training
        loss_after, _ = evaluate(self.model, self.train_loader, self.loss_fn)

        num_examples = len(self.train_loader.dataset)
        metrics = {
            "loss_before": float(loss_before),
            "loss_after": float(loss_after),
            "num_samples": num_examples,
        }

        return self.get_parameters({}), num_examples, metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, acc = evaluate(self.model, self.test_loader, self.loss_fn)
        num_examples = len(self.test_loader.dataset)
        return float(loss), num_examples, {"accuracy": float(acc)}
