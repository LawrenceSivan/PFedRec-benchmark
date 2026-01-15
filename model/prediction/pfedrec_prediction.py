import torch
import torch.nn as nn
from model.prediction.base_prediction import BasePrediction

class PFedRecPrediction(BasePrediction):
    def __init__(self, input_dim):
        super(PFedRecPrediction, self).__init__()
        self.affine_output = nn.Linear(in_features=input_dim, out_features=1)
        self.logistic = nn.Sigmoid()
        self.optimizer = None

    def configure_optimizer(self, config):
        self.optimizer = torch.optim.SGD(self.parameters(),
                                    lr=config['lr'],
                                    weight_decay=config['l2_regularization'])

    def zero_grad_optimizer(self):
        if self.optimizer is not None:
            self.optimizer.zero_grad()

    def step_optimizer(self):
        if self.optimizer is not None:
            self.optimizer.step()

    def forward(self, x):
        logits = self.affine_output(x)
        rating = self.logistic(logits)
        return rating
