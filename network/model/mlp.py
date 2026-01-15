import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim=1):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()

        current_dim = input_dim
        for h_dim in hidden_layers:
            self.layers.append(nn.Linear(current_dim, h_dim))
            self.layers.append(nn.ReLU())
            current_dim = h_dim

        self.final_layer = nn.Linear(current_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.final_layer(x)
        return self.sigmoid(x)

