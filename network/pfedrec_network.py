import torch
from network.embedding.pfedrec_embedding import PFedRecEmbedding
from network.prediction.pfedrec_prediction import PFedRecPrediction

class PFedRecNetwork(torch.nn.Module):
    def __init__(self, config):
        super(PFedRecNetwork, self).__init__()
        self.config = config
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim']

        self.embedding = PFedRecEmbedding(num_items=self.num_items, latent_dim=self.latent_dim)

        self.prediction = PFedRecPrediction(config)

    def forward(self, item_indices):
        item_embedding = self.embedding(item_indices)
        rating = self.prediction(item_embedding)
        return rating

