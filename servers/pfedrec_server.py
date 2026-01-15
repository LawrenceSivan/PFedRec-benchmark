from servers.base_server import Base_Server
import torch
import logging
import random
import os
from clients.pfedrec_client import PFedRec_Client
from model.pfedrec_model import PFedRecModel
from model.embedding.pfedrec_embedding import PFedRecEmbedding
from utils.utils_checkpoint import save_checkpoint
from metrics import MetronAtK

class PFedRec_Server(Base_Server):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.clients = {}
        self.model_template = PFedRecModel(config)
        temp_embedding = PFedRecEmbedding(config['num_items'], config['latent_dim'])
        self.global_item_embeddings = temp_embedding.state_dict()
        self.metron = MetronAtK(top_k=10)

    def aggregate(self, client_params_list):
        if not client_params_list:
             return self.global_item_embeddings

        agg_keys = [k for k in client_params_list[0].keys() if k.startswith('embedding')]

        agg_params = {k: client_params_list[0][k].clone() for k in agg_keys}

        for key in agg_keys:
            client_weights = [client[key].data.float() for client in client_params_list]
            stacked_params = torch.stack(client_weights)
            agg_params[key].data.copy_(torch.mean(stacked_params, dim=0))

        return agg_params

    def evaluate(self, test_data):
        test_users_t = test_data[0]
        test_items_t = test_data[1]
        neg_users_t = test_data[2]
        neg_items_t = test_data[3]

        if self.config['use_cuda']:
            test_users_t = test_users_t.cuda()
            test_items_t = test_items_t.cuda()
            neg_users_t = neg_users_t.cuda()
            neg_items_t = neg_items_t.cuda()

        unique_test_users = torch.unique(test_users_t)

        batch_test_users = []
        batch_test_items = []
        batch_test_scores = []

        batch_neg_users = []
        batch_neg_items = []
        batch_neg_scores = []

        if self.global_item_embeddings:
            self.model_template.embedding.load_state_dict(self.global_item_embeddings, strict=False)

        for u in unique_test_users:
            u_id = u.item()

            # Test Item
            idx_test = (test_users_t == u)
            u_test_items = test_items_t[idx_test]

            # Neg Items
            idx_neg = (neg_users_t == u)
            u_neg_items = neg_items_t[idx_neg]

            if u_id in self.clients:
                client = self.clients[u_id]
                u_test_preds = client.evaluate(u_test_items, self.global_item_embeddings)
                u_neg_preds = client.evaluate(u_neg_items, self.global_item_embeddings)
            else:
                self.model_template.eval()
                with torch.no_grad():
                     u_test_preds = self.model_template(u_test_items)
                     u_neg_preds = self.model_template(u_neg_items)

                u_test_preds = u_test_preds.cpu()
                u_neg_preds = u_neg_preds.cpu()

            batch_test_users.extend([u_id] * len(u_test_items))
            batch_test_items.extend(u_test_items.tolist())
            batch_test_scores.extend(u_test_preds.view(-1).tolist())

            batch_neg_users.extend([u_id] * len(u_neg_items))
            batch_neg_items.extend(u_neg_items.tolist())
            batch_neg_scores.extend(u_neg_preds.view(-1).tolist())

        self.metron.subjects = [batch_test_users, batch_test_items, batch_test_scores,
                           batch_neg_users, batch_neg_items, batch_neg_scores]

        hr = self.metron.cal_hit_ratio()
        ndcg = self.metron.cal_ndcg()

        return hr, ndcg

    def train(self, sample_generator):
        test_data = sample_generator.test_data
        validate_data = sample_generator.validate_data

        best_val_hr = 0.0

        for round_id in range(self.config['num_round']):
            logging.info('-' * 80)
            logging.info(f'Round {round_id} starts !')

            # 1. Sample Participants
            if self.config['clients_sample_ratio'] <= 1:
                num_participants = int(self.config['num_users'] * self.config['clients_sample_ratio'])
                participants = random.sample(range(self.config['num_users']), num_participants)
            else:
                participants = random.sample(range(self.config['num_users']), self.config['clients_sample_num'])

            logging.info(f"Round {round_id}: {len(participants)} participants.")

            # 2. Get Training Data for this round
            all_train_data = sample_generator.store_all_train_data(self.config['num_negative'])

            user_data_map = {}
            for idx, u_list in enumerate(all_train_data[0]):
                if len(u_list) > 0:
                    uid = u_list[0]
                    user_data_map[uid] = idx

            participant_params_list = []
            loss_list = []

            for user_id in participants:
                if user_id not in user_data_map:
                    continue

                if user_id not in self.clients:
                    self.clients[user_id] = PFedRec_Client(user_id, self.config, self.model_template)

                client = self.clients[user_id]

                data_idx = user_data_map[user_id]
                user_data = [all_train_data[0][data_idx], all_train_data[1][data_idx], all_train_data[2][data_idx]]

                client_item_params, loss = client.train(user_data, self.global_item_embeddings)

                participant_params_list.append(client_item_params)
                loss_list.append(loss)

            if loss_list:
                avg_loss = sum(loss_list) / len(loss_list)
                logging.info(f"Round {round_id} Average Loss: {avg_loss:.4f}")
            else:
                 logging.info(f"Round {round_id} No clients trained.")

            # 3. Aggregate
            if participant_params_list:
                self.global_item_embeddings = self.aggregate(participant_params_list)

            # 4. Evaluation
            logging.info("Testing phase...")
            hr, ndcg = self.evaluate(test_data)
            logging.info(f"[Testing Round {round_id}] HR@10: {hr:.4f}, NDCG@10: {ndcg:.4f}")

            logging.info("Validating phase...")
            val_hr, val_ndcg = self.evaluate(validate_data)
            logging.info(f"[Validating Round {round_id}] HR@10: {val_hr:.4f}, NDCG@10: {val_ndcg:.4f}")

            if val_hr > best_val_hr:
                best_val_hr = val_hr
                # best_ndcg = ndcg
                logging.info(f"New best Validation HR: {best_val_hr:.4f}! Saving checkpoint.")
                save_checkpoint(self.global_item_embeddings,
                                os.path.join('checkpoints', f"{self.config['algorithm']}_{self.config['dataset']}_best.pth"))
