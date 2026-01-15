import torch
import torch.nn as nn
from network.prediction.base_prediction import BasePrediction

class GPFedRecPrediction(BasePrediction):
    def __init__(self, config):
        super(GPFedRecPrediction, self).__init__()
        self.latent_dim = config['latent_dim']
        self.embedding_user = nn.Embedding(num_embeddings=1, embedding_dim=self.latent_dim)

        model_name = config.get('model_name', 'mlp')
        if model_name == 'mlp':
            from network.model.mlp import MLP
            if 'layers' in config and len(config['layers']) > 0:
                 hidden_layers = config['layers'][1:]
                 input_dim = 2 * self.latent_dim
                 self.model = MLP(input_dim=input_dim, hidden_layers=hidden_layers, output_dim=1)

        self.optimizer = None
        self.optimizer_u = None # Optimizer for user embedding


    def forward(self, item_embedding):
        batch_size = item_embedding.size(0)
        device = item_embedding.device

        user_indices = torch.zeros(batch_size, dtype=torch.long, device=device)
        user_embedding = self.embedding_user(user_indices)

        vector = torch.cat([user_embedding, item_embedding], dim=-1)

        rating = self.model(vector)
        return rating

    def configure_optimizer(self, config):
        self.optimizer = torch.optim.SGD([
            {'params': self.model.parameters()}
        ], lr=config['lr'], weight_decay=config['l2_regularization'])

        self.optimizer_u = torch.optim.SGD(self.embedding_user.parameters(),
                                           lr=config['lr'] * config['lr_eta'],
                                           weight_decay=config['l2_regularization'])

    def zero_grad_optimizer(self):
        if self.optimizer is not None:
            self.optimizer.zero_grad()
        if self.optimizer_u is not None:
            self.optimizer_u.zero_grad()

    def step_optimizer(self):
        if self.optimizer is not None:
            self.optimizer.step()
        if self.optimizer_u is not None:
            self.optimizer_u.step()
