import torch


def elbo_loss(
    ground_truth: torch.Tensor,
    pred: tuple,
    beta: float,
    device: torch.device,
):
    """ELBO loss function: reconstruction loss + KLD loss."""
    x_pred, mean, std_dev = pred
    log_var = 2 * torch.log(std_dev).to(device)
    enc_err = -(0.5 * (1 + log_var - mean.pow(2) - log_var.exp())).sum(dim=1).mean()
    loss_fn = torch.nn.MSELoss()
    dec_err = loss_fn(x_pred, ground_truth)
    return enc_err * beta + dec_err
