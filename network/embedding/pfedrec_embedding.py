import torch
from network.embedding.base_embedding import BaseEmbedding

class PFedRecEmbedding(BaseEmbedding):
    def __init__(self, num_items, latent_dim):
        super(PFedRecEmbedding, self).__init__(num_embeddings=num_items, embedding_dim=latent_dim)
        self.optimizer = None

    def configure_optimizer(self, config):
        self.optimizer = torch.optim.SGD(self.parameters(),
                                         lr=config['lr'] * config['num_items'] * config['lr_eta'],
                                         weight_decay=config['l2_regularization'])

    def zero_grad_optimizer(self):
        if self.optimizer is not None:
            self.optimizer.zero_grad()

    def step_optimizer(self):
        if self.optimizer is not None:
            self.optimizer.step()
