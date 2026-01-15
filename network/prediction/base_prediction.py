import torch
import torch.nn as nn

class BasePrediction(nn.Module):
    def __init__(self):
        super(BasePrediction, self).__init__()

    def forward(self, x):
        raise NotImplementedError


class MLPPrediction(BasePrediction):
    def __init__(self, input_dim):
        super(MLPPrediction, self).__init__()
        self.affine_output = nn.Linear(in_features=input_dim, out_features=1)
        self.logistic = nn.Sigmoid()
        self.optimizer = None

    def configure_optimizer(self, config):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config['lr'], weight_decay=config.get('l2_regularization', 0))

    def zero_grad_optimizer(self):
        if self.optimizer:
            self.optimizer.zero_grad()

    def step_optimizer(self):
        if self.optimizer:
            self.optimizer.step()

    def forward(self, x):
        logits = self.affine_output(x)
        rating = self.logistic(logits)
        return rating
