import torch


def elbo_loss(
    ground_truth: torch.Tensor,
    pred: tuple,
    beta: float,
    device: torch.device,
    **kwargs,
):
    """ELBO loss function: reconstruction loss + KLD loss.
    debug_info: (optional) assumed to be a default dict.
    """
    x_pred, mean, std_dev = pred
    log_var = 2 * torch.log(std_dev).to(device)
    enc_err = -(0.5 * (1 + log_var - mean.pow(2) - log_var.exp())).sum(dim=1).mean()
    loss_fn = torch.nn.MSELoss()
    dec_err = loss_fn(x_pred, ground_truth)

    if 'debug_info' in kwargs:
        kwargs['debug_info']["enc_err"].append(enc_err.detach().item())
        kwargs['debug_info']["dec_err"].append(dec_err.detach().item())

    return enc_err * beta + dec_err
