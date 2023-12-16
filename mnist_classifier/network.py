import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(
        self, n_conv_channels: int, n_linear_out_channels: int, learn_bias: bool
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(1, n_conv_channels, 3, bias=learn_bias)
        self.linear1 = nn.Linear(
            n_conv_channels * 13 * 13, n_linear_out_channels, bias=learn_bias
        )
        self.linear2 = nn.Linear(n_linear_out_channels, 10, bias=learn_bias)

        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.xavier_normal_(self.linear2.weight)

    def __call__(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = x.flatten(start_dim=1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x
