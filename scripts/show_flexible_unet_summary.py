from monai.networks.nets import FlexibleUNet
from torchinfo import summary
import torch
from torch.nn import Softmax, Sigmoid
import numpy as np
import random

seed = 3
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


def main():
    device = "cpu"

    model = FlexibleUNet(
        in_channels=3,
        out_channels=3,
        backbone="efficientnet-b0",
        pretrained=True,
        is_pad=False,
    ).to(device)

    summary(model, input_size=(1, 3, 480, 640), device=device)

    input_tensor = torch.rand((1, 3, 480, 640))
    output = model(input_tensor)
    print("\nAdditional output information\n")
    print(f"Output shape: {output.shape}")
    print(f"Output max: {output.max():0.4f}")
    print(f"Output min: {output.min():0.4f}")
    i = 40
    j = 40
    print(f"Output at ({i},{j}): {output[0,:,i,j].detach().cpu().numpy()}")
    output_softmax = Softmax(dim=1)(output)
    output_sigmoid = Sigmoid()(output)
    print(f"Output at ({i},{j}) with softmax: {output_softmax[0,:,i,j].detach().cpu().numpy()}")
    print(f"Output at ({i},{j}) with sigmoid: {output_sigmoid[0,:,i,j].detach().cpu().numpy()}")


if __name__ == "__main__":
    main()
