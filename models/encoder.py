"""Feature encoder for ATDA.

it's called as `shared network` in the paper.
"""

from torch import nn


class Encoder(nn.Module):
    """Feature encoder class for ATDA."""

    def __init__(self):
        """Init encoder."""
        super(Encoder, self).__init__()

        self.restored = False

        self.encoder = nn.Sequential(
            # 1st conv block
            # input [1 x 28 x 28]
            # output [32 x 12 x 12]
            nn.Conv2d(1, 32, 5, 1, 0, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 2nd conv block
            # input [32 x 12 x 12]
            # output [48 x 4 x 4]
            nn.Conv2d(32, 48, 5, 1, 0, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    def forward(self, input):
        """Forward encoder."""
        out = self.encoder(input)
        return out.view(-1)
