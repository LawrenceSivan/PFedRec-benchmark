from clients.base_client import Base_Client
import torch
from data.dataloader import UserItemRatingDataset
from torch.utils.data import DataLoader

class PFedRec_Client(Base_Client):
    def __init__(self, user_id, config, model):
        super().__init__(user_id, config, model)
        self.embedding = self.local_model.embedding
        self.prediction = self.local_model.prediction
        self.crit = torch.nn.BCELoss()

    def set_params(self, model_params):
        if model_params is not None:
             self.embedding.load_state_dict(model_params, strict=False)

    def get_params(self):
        return self.embedding.state_dict()

    def train(self, train_data, global_params):

        self.set_params(global_params)

        user_tensor = torch.LongTensor(train_data[0])
        item_tensor = torch.LongTensor(train_data[1])
        target_tensor = torch.FloatTensor(train_data[2])

        dataset = UserItemRatingDataset(user_tensor=user_tensor,
                                        item_tensor=item_tensor,
                                        target_tensor=target_tensor)
        dataloader = DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True)

        self.prediction.configure_optimizer(self.config)
        self.embedding.configure_optimizer(self.config)

        self.embedding.train()
        self.prediction.train()

        total_loss = 0.0
        sample_num = 0

        for epoch in range(self.config['local_epoch']):
            for batch_id, (u, items, ratings) in enumerate(dataloader):
                if self.config['use_cuda']:
                    items, ratings = items.cuda(), ratings.cuda()

                ratings = ratings.float()

                self.prediction.zero_grad_optimizer()
                emb = self.embedding(items)
                ratings_pred = self.prediction(emb)
                loss = self.crit(ratings_pred.view(-1), ratings)
                loss.backward()
                self.prediction.step_optimizer()

                self.embedding.zero_grad_optimizer()
                emb = self.embedding(items)
                ratings_pred = self.prediction(emb)
                loss_i = self.crit(ratings_pred.view(-1), ratings)
                loss_i.backward()
                self.embedding.step_optimizer()

                total_loss += loss_i.item() * len(items)
                sample_num += len(items)

        avg_loss = total_loss / sample_num if sample_num > 0 else 0.0

        client_item_params = self.get_params()

        return client_item_params, avg_loss

    def evaluate(self, items_tensor, global_item_embedding_state_dict=None):
        if global_item_embedding_state_dict is not None:
             self.embedding.load_state_dict(global_item_embedding_state_dict, strict=False)

        self.embedding.eval()
        self.prediction.eval()

        if self.config['use_cuda']:
            items_tensor = items_tensor.cuda()

        with torch.no_grad():
            emb = self.embedding(items_tensor)
            ratings_pred = self.prediction(emb)

        return ratings_pred.cpu().detach()
