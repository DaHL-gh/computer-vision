import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

def accuracy_fn(y_pred, y_true):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


def train_step(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optim: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    train_loss = train_acc = 0

    model.to(device).train()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        y_pred = model(X)

        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        train_acc += accuracy_fn(y_pred.argmax(dim=1), y)

        optim.zero_grad()

        loss.backward()

        optim.step()

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)

    return {"loss": train_loss, "acc": train_acc}


def test_step(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    test_loss, test_acc = 0, 0

    model.eval()

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            y_pred = model(X)

            test_loss += loss_fn(y_pred, y).item()
            test_acc += accuracy_fn(y_pred.argmax(dim=1), y)

        test_loss /= len(dataloader)
        test_acc /= len(dataloader)

    return {"loss": test_loss, "acc": test_acc}


def train_model(
    model: nn.Module,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    loss_fn: nn.Module,
    optim: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
) -> dict[str, dict[str, list[float]]]:
    results: dict[str, dict[str, list[float]]] = {
        "train": {"loss": [], "acc": []},
        "test": {"loss": [], "acc": []},
    }

    for epoch in tqdm(range(epochs)):
        # TRAIN
        res = train_step(model, train_dataloader, loss_fn, optim, device)

        for name, num in res.items():
            results["train"][name].append(num)

        # TEST
        res = test_step(model, test_dataloader, loss_fn, device)

        for name, num in res.items():
            results["test"][name].append(num)

    return results
