import torch
from model.embedding.pfedrec_embedding import PFedRecEmbedding
from model.prediction.pfedrec_prediction import PFedRecPrediction

class PFedRecModel(torch.nn.Module):
    def __init__(self, config):
        super(PFedRecModel, self).__init__()
        self.config = config
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim']

        self.embedding = PFedRecEmbedding(num_items=self.num_items, latent_dim=self.latent_dim)
        self.prediction = PFedRecPrediction(input_dim=self.latent_dim)

    def forward(self, item_indices):
        item_embedding = self.embedding(item_indices)
        rating = self.prediction(item_embedding)
        return rating

