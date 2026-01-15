import torch
from network.prediction.base_prediction import BasePrediction

class PFedRecPrediction(BasePrediction):
    def __init__(self, config):
        super(PFedRecPrediction, self).__init__()
        self.config = config
        input_dim = config['latent_dim']

        model_name = config.get('model_name', 'mlp')
        if model_name == 'mlp':
            from network.model.mlp import MLP
            self.model = MLP(input_dim=input_dim, hidden_layers=[], output_dim=1)

        self.optimizer = None

    def configure_optimizer(self, config):
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                    lr=config['lr'],
                                    weight_decay=config['l2_regularization'])

    def zero_grad_optimizer(self):
        if self.optimizer is not None:
            self.optimizer.zero_grad()

    def step_optimizer(self):
        if self.optimizer is not None:
            self.optimizer.step()

    def forward(self, x):
        return self.model(x)
