import torch


def train_single_pass(
    data_loader: torch.data.utils.DataLoader,
    model: torch.nn.Module,
    loss_fn,
    optimizer,
    device,
):
    """Trains the model for a single epoch, and return the average loss over the entire data."""
    model.train()

    loss_hist = []
    for X, y in data_loader:
        # TODO: need to generalize the reshape operation.
        X, y = X.reshape(-1, 784).to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(X, pred)
        loss_hist.append(loss.detach().item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return torch.Tensor(loss_hist).mean()
