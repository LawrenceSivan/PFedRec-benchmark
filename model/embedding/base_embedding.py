import torch.nn as nn

class BaseEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(BaseEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, input_indices):
        return self.embedding(input_indices)

    def configure_optimizer(self, config):
        raise NotImplementedError

    def zero_grad_optimizer(self):
        raise NotImplementedError

    def step_optimizer(self):
        raise NotImplementedError
