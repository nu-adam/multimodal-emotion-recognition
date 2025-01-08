import torch.nn as nn


class ProjectionNetwork(nn.Module):
    """
    Reduces the dimensionality of input features.

    Methods:
        forward(x): Projects the input features to a lower-dimensional space.
    """
    def __init__(self, input_dim=512, output_dim=256):
        super(ProjectionNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, output_dim)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        """
        Forward pass through the projection network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).

        Returns:
            torch.Tensor: Projected features of shape (batch_size, seq_len, output_dim).
        """
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        return x
