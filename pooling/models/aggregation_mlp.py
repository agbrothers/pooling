import torch.nn as nn

# Define a simple model
class AggregationMLP(nn.Module):
    def __init__(self, num_vectors, dim_vectors, aggregate_dim):
        super(AggregationMLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_vectors * dim_vectors, 128),
            nn.ReLU(),
            nn.Linear(128, aggregate_dim)
        )

    def forward(self, x):
        return self.fc(x)
