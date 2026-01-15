import torch
from network.embedding.gpfedrec_embedding import GPFedRecEmbedding
from network.prediction.gpfedrec_prediction import GPFedRecPrediction

class GPFedRecNetwork(torch.nn.Module):
    def __init__(self, config):
        super(GPFedRecNetwork, self).__init__()
        self.config = config
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim']

        self.embedding = GPFedRecEmbedding(num_items=self.num_items, latent_dim=self.latent_dim)

        self.prediction = GPFedRecPrediction(config)

    def forward(self, item_indices):
        item_embedding = self.embedding(item_indices)
        rating = self.prediction(item_embedding)
        return rating

