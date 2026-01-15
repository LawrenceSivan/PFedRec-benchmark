import torch
import torch.nn as nn
from model.prediction.base_prediction import BasePrediction

class GPFedRecPrediction(BasePrediction):
    def __init__(self, config):
        super(GPFedRecPrediction, self).__init__()
        self.latent_dim = config['latent_dim']
        # User embedding is part of prediction/MLP logic in this split
        self.embedding_user = nn.Embedding(num_embeddings=1, embedding_dim=self.latent_dim)

        self.fc_layers = nn.ModuleList()
        # layers config: e.g. [16, 32, 16, 8]
        # But wait, input dim to MLP is latent_dim * 2 (user + item)
        # In MLP.py:
        # for idx, (in_size, out_size) in enumerate(zip(config['layers'][:-1], config['layers'][1:])):
        # But the first layer input size?
        # In original MLP.py:
        # The input vector is concat of user and item embedding. dimension = latent_dim * 2.
        # But config['layers'] normally defines hidden sizes.
        # Let's check typical config structure in GPFedRec-main/train.py or parser.

        # Checking MLP logic again:
        # vector = torch.cat([user_embedding, item_embedding], dim=-1)
        # for idx, _ in enumerate(range(len(self.fc_layers))):
        #    vector = self.fc_layers[idx](vector)

        # This implies the first layer of fc_layers must accept latent_dim * 2.
        # So config['layers'][0] must be latent_dim * 2?
        # OR the loop handles it?
        # zip(config['layers'][:-1], config['layers'][1:])
        # This creates pairs.
        # If layers = [64, 32, 16, 8], then (64,32), (32,16), (16,8).
        # And input 'vector' must be size 64.
        # So latent_dim * 2 == layers[0].

        for idx, (in_size, out_size) in enumerate(zip(config['layers'][:-1], config['layers'][1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))

        self.affine_output = torch.nn.Linear(in_features=config['layers'][-1], out_features=1)
        self.logistic = torch.nn.Sigmoid()

        self.optimizer = None
        self.optimizer_u = None # Optimizer for user embedding

    def configure_optimizer(self, config):
        # In engine.py:
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=config['lr'], weight_decay=config['l2_regularization'])
        # But wait, engine.py splits optimizers:
        # optimizer = torch.optim.Adam([
        #     {'params': self.model.fc_layers.parameters()},
        #     {'params': self.model.affine_output.parameters()}
        # ], lr=config['lr'], weight_decay=config['l2_regularization'])
        # optimizer_u = torch.optim.SGD(self.model.embedding_user.parameters(), lr=config['lr']*config['lr_eta'], weight_decay=config['l2_regularization'])

        # So we need two optimizers or handle them separately.
        # 'optimizer' handles MLP layers. 'optimizer_u' handles user embedding.

        self.optimizer = torch.optim.Adam([
            {'params': self.fc_layers.parameters()},
            {'params': self.affine_output.parameters()}
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

    def forward(self, item_embedding):
        # item_embedding: (batch_size, latent_dim)
        batch_size = item_embedding.size(0)
        device = item_embedding.device

        # User embedding index 0, repeated for batch
        user_indices = torch.zeros(batch_size, dtype=torch.long, device=device)
        user_embedding = self.embedding_user(user_indices)

        vector = torch.cat([user_embedding, item_embedding], dim=-1)

        for idx in range(len(self.fc_layers)):
            vector = self.fc_layers[idx](vector)
            vector = torch.nn.ReLU()(vector)

        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating

