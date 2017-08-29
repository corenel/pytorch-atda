"""Feature Classifier for ATDA.

it's called as `labelling network` or `target specific network` in the paper.
"""

from torch import nn


class Classifier(nn.Module):
    """Feature classifier class for ATDA."""

    def __init__(self,
                 dropout_keep=0.5,
                 input_dims=768,
                 hidden_dims=100,
                 output_dims=10):
        """Init classifier."""
        super(Classifier, self).__init__()

        self.dropout_keep = dropout_keep
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.output_dims = output_dims
        self.restored = False

        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout_keep),
            nn.Linear(self.input_dims, self.hidden_dims),
            nn.BatchNorm1d(self.hidden_dims),
            nn.ReLU(),
            nn.Dropout(self.dropout_keep),
            nn.Linear(self.hidden_dims, self.hidden_dims),
            nn.BatchNorm1d(self.hidden_dims),
            nn.ReLU(),
            nn.Linear(self.hidden_dims, self.output_dims),
            nn.Softmax()
        )

    def forward(self, input):
        """Forward classifier."""
        out = self.classifier(input)
        return out.view()
