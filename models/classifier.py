"""Feature Classifier for ATDA.

it's called as `labelling network` or `target specific network` in the paper.
"""

from torch import nn


class ClassifierA(nn.Module):
    """Feature classifier class for MNIST -> MNIST-M experiment in ATDA."""

    def __init__(self, dropout_keep=0.5, use_BN=True):
        """Init classifier."""
        super(ClassifierA, self).__init__()

        self.dropout_keep = dropout_keep
        self.use_BN = use_BN
        self.restored = False

        if use_BN:
            self.classifier = nn.Sequential(
                nn.Dropout(self.dropout_keep),
                nn.Linear(768, 100),
                nn.BatchNorm1d(100),
                nn.ReLU(),
                nn.Dropout(self.dropout_keep),
                nn.Linear(100, 100),
                nn.BatchNorm1d(100),
                nn.ReLU(),
                nn.Linear(100, 10),
                nn.LogSoftmax()
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(self.dropout_keep),
                nn.Linear(768, 100),
                nn.ReLU(),
                nn.Dropout(self.dropout_keep),
                nn.Linear(100, 100),
                nn.ReLU(),
                nn.Linear(100, 10),
                nn.LogSoftmax()
            )

    def forward(self, input):
        """Forward classifier."""
        out = self.classifier(input)
        return out.view(-1)


class ClassifierB(nn.Module):
    """Feature classifier class for MNIST -> SVHN experiment in ATDA."""

    def __init__(self):
        """Init classifier."""
        super(ClassifierB, self).__init__()

        self.restored = False

        self.classifier = nn.Sequential(
            nn.Linear(3072, 2048),
            nn.ReLU(),
            nn.Linear(2048, 10),
            nn.LogSoftmax()
        )

    def forward(self, input):
        """Forward classifier."""
        out = self.classifier(input)
        return out.view(-1)


class ClassifierC(nn.Module):
    """Feature classifier class for SVHN -> MNIST or SYN Digits -> SVHN."""

    def __init__(self, dropout_keep=0.5, use_BN=True):
        """Init classifier."""
        super(ClassifierA, self).__init__()

        self.dropout_keep = dropout_keep
        self.use_BN = use_BN
        self.restored = False

        if self.use_BN:
            self.classifier = nn.Sequential(
                nn.Dropout(self.dropout_keep),
                nn.Linear(3072, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Dropout(self.dropout_keep),
                nn.Linear(2048, 10),
                nn.BatchNorm1d(10),
                nn.LogSoftmax()
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(self.dropout_keep),
                nn.Linear(3072, 2048),
                nn.ReLU(),
                nn.Dropout(self.dropout_keep),
                nn.Linear(2048, 10),
                nn.LogSoftmax()
            )

    def forward(self, input):
        """Forward classifier."""
        out = self.classifier(input)
        return out.view(-1)
