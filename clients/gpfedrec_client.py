from clients.pfedrec_client import PFedRec_Client
import torch
from torch.nn import MSELoss
from data.dataloader import UserItemRatingDataset
from torch.utils.data import DataLoader

def compute_regularization(model, parameter_label):
    current_weight = model.embedding.weight
    reg_fn = MSELoss(reduction='mean')
    reg_loss = reg_fn(current_weight, parameter_label)
    return reg_loss

class GPFedRec_Client(PFedRec_Client):
    def __init__(self, user_id, config, model):
        super().__init__(user_id, config, model)

    def train(self, sample_generator, global_params, reg_item_embedding_state=None):
        """
        Args:
             global_params: The model parameters. In GPFedRec, this updates the local model.
             reg_item_embedding_state: The embedding state used for regularization target.
                                       It corresponds to 'reg_item_embedding' in GPFedRec engine.py
        """
        self.set_params(global_params)

        # Client fetches its own data
        train_data = sample_generator.store_train_data_for_user(self.user_id, self.config['num_negative'])

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

        # Prepare regularization target if available
        reg_target = None
        if reg_item_embedding_state is not None:
             # reg_item_embedding_state is expected to be a tensor (weight matrix) or state_dict
             # In GPFedRec-main engine.py: reg_item_embedding = server_model_param['...'][user].data
             # So it is a Tensor.
             reg_target = reg_item_embedding_state
             if self.config['use_cuda']:
                 reg_target = reg_target.cuda()

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

                # Add Regularization
                if reg_target is not None:
                    # We pass self.embedding (the module) and the target tensor
                    reg_term = compute_regularization(self.embedding, reg_target)
                    loss_i += self.config['reg'] * reg_term

                loss_i.backward()
                self.embedding.step_optimizer()

                total_loss += loss_i.item() * len(items)
                sample_num += len(items)

        avg_loss = total_loss / sample_num if sample_num > 0 else 0.0

        client_item_params = self.get_params()

        return client_item_params, avg_loss

