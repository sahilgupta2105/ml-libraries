import matplotlib.pyplot as plt
import torch


def show_reconstructions(model, data, unpack_out=None, n=5):
    model.eval()

    fig, axes = plt.subplots(3, n, figsize=(n * 2, 4))
    for _, (x, _) in enumerate(data):
        with torch.no_grad():
            x = x[:n].to(next(model.parameters()).device)  # first n images
            # TODO: need to generalize the reshape operation.
            out = model(x.reshape(-1, 784)).cpu()
            if unpack_out:
                out = unpack_out(out)

        for i in range(n):
            # TODO: need to generalize the reshape operation.
            axes[0, i].imshow(x[i].reshape(28, 28).cpu().numpy())
            axes[0, i].axis("off")
            axes[0, i].set_title("Input")

            # reconstruction
            axes[1, i].imshow(out[i].reshape(28, 28).numpy())
            axes[1, i].axis("off")
            axes[1, i].set_title("Recon")

        break

    plt.tight_layout()
    plt.show()
