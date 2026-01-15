from servers.base_server import Base_Server
import torch
import logging
import random
import os
import copy
import numpy as np
from sklearn.metrics import pairwise_distances
from clients.gpfedrec_client import GPFedRec_Client
from network.gpfedrec_network import GPFedRecNetwork
from network.embedding.gpfedrec_embedding import GPFedRecEmbedding
from utils.utils_checkpoint import save_checkpoint
from metrics import MetronAtK

class GPFedRec_Server(Base_Server):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.clients = {}
        self.model_template = GPFedRecNetwork(config)
        temp_embedding = GPFedRecEmbedding(config['num_items'], config['latent_dim'])

        self.initial_global_embedding = temp_embedding.state_dict()

        self.user_embeddings_map = {}

        self.global_embedding = self.initial_global_embedding

        self.metron = MetronAtK(top_k=10)

    def get_user_embedding(self, user_id):
        if user_id not in self.user_embeddings_map:
            return copy.deepcopy(self.global_embedding)
        return self.user_embeddings_map[user_id]

    def _construct_user_relation_graph(self, round_user_params):
        item_num = self.config['num_items']
        latent_dim = self.config['latent_dim']
        similarity_metric = self.config['similarity_metric']

        num_participants = len(round_user_params)
        item_embedding = np.zeros((num_participants, item_num * latent_dim), dtype='float32')

        for i, (user_id, params) in enumerate(round_user_params.items()):
            weight = params.get('embedding.weight')
            item_embedding[i] = weight.cpu().numpy().flatten()

        adj = pairwise_distances(item_embedding, metric=similarity_metric)
        if similarity_metric == 'cosine':
            return adj
        else:
            return -adj

    def _select_topk_neighborhood(self, user_relation_graph):
        neighborhood_size = self.config['neighborhood_size']
        neighborhood_threshold = self.config['neighborhood_threshold']

        topk_user_relation_graph = np.zeros(user_relation_graph.shape, dtype='float32')
        if neighborhood_size > 0:
            for user in range(user_relation_graph.shape[0]):
                user_neighborhood = user_relation_graph[user]
                topk_indexes = user_neighborhood.argsort()[-neighborhood_size:][::-1]
                for i in topk_indexes:
                    topk_user_relation_graph[user][i] = 1 / neighborhood_size
        else:
            similarity_threshold = np.mean(user_relation_graph) * neighborhood_threshold
            for i in range(user_relation_graph.shape[0]):
                high_num = np.sum(user_relation_graph[i] > similarity_threshold)
                if high_num > 0:
                    for j in range(user_relation_graph.shape[1]):
                        if user_relation_graph[i][j] > similarity_threshold:
                            topk_user_relation_graph[i][j] = 1/high_num
                else:
                    topk_user_relation_graph[i][i] = 1

        return topk_user_relation_graph

    def _mp_on_graph(self, round_user_params, topk_user_relation_graph):
        layers = self.config['mp_layers']
        item_num = self.config['num_items']
        latent_dim = self.config['latent_dim']

        num_participants = len(round_user_params)
        item_embedding = np.zeros((num_participants, item_num*latent_dim), dtype='float32')

        for i, (user_id, params) in enumerate(round_user_params.items()):
            weight = params.get('embedding.weight')
            item_embedding[i] = weight.cpu().numpy().flatten()

        aggregated_item_embedding = np.matmul(topk_user_relation_graph, item_embedding)
        for layer in range(layers-1):
            aggregated_item_embedding = np.matmul(topk_user_relation_graph, aggregated_item_embedding)

        item_embedding_dict = {}
        for i, user_id in enumerate(round_user_params.keys()):
            user_emb_flat = aggregated_item_embedding[i]
            item_embedding_dict[user_id] = torch.from_numpy(user_emb_flat.reshape(item_num, latent_dim))

        item_embedding_dict['global'] = sum(item_embedding_dict.values())/len(round_user_params)

        return item_embedding_dict

    def aggregate(self, client_params_dict):
        """
        Args:
            client_params_dict: {user_id: client_params (state_dict)}
        """
        if not client_params_dict:
             return

        user_relation_graph = self._construct_user_relation_graph(client_params_dict)

        topk_graph = self._select_topk_neighborhood(user_relation_graph)

        updated_embeddings_dict = self._mp_on_graph(client_params_dict, topk_graph)

        global_weight = updated_embeddings_dict.pop('global')
        self.global_embedding = {'embedding.weight': global_weight.cpu()}

        for user_id, weight_tensor in updated_embeddings_dict.items():
            new_state_dict = {'embedding.weight': weight_tensor.cpu()}
            self.user_embeddings_map[user_id] = new_state_dict


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

        for u in unique_test_users:
            u_id = u.item()

            idx_test = (test_users_t == u)
            u_test_items = test_items_t[idx_test]

            idx_neg = (neg_users_t == u)
            u_neg_items = neg_items_t[idx_neg]

            if u_id in self.user_embeddings_map:
                embedding_state = self.user_embeddings_map[u_id]
            else:
                embedding_state = self.global_embedding

            if u_id in self.clients:
                client = self.clients[u_id]
                u_test_preds = client.evaluate(u_test_items, embedding_state)
                u_neg_preds = client.evaluate(u_neg_items, embedding_state)
            else:
                self.model_template.embedding.load_state_dict(embedding_state, strict=False)
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

            # 2. Get Training Data for this round (with negative sampling)
            all_train_data = sample_generator.store_all_train_data(self.config['num_negative'])

            participant_params_dict = {}
            loss_list = []

            for user_id in participants:
                if user_id not in sample_generator.user_pool:
                     continue

                if user_id not in self.clients:
                    self.clients[user_id] = GPFedRec_Client(user_id, self.config, self.model_template)
                client = self.clients[user_id]

                current_embedding = self.get_user_embedding(user_id)
                reg_target = current_embedding.get('embedding.weight')

                # fetch own data
                user_train_data = [all_train_data[0][user_id], all_train_data[1][user_id], all_train_data[2][user_id]]

                client_item_params, loss = client.train(user_train_data, current_embedding, reg_item_embedding_state=reg_target)
                participant_params_dict[user_id] = client_item_params
                loss_list.append(loss)

            if loss_list:
                avg_loss = sum(loss_list) / len(loss_list)
                logging.info(f"Round {round_id} Average Loss: {avg_loss:.4f}")
            else:
                 logging.info(f"Round {round_id} No clients trained.")

            # 3. Aggregate
            if participant_params_dict:
                self.aggregate(participant_params_dict)

            # 4. Evaluation
            logging.info("Testing phase...")
            hr, ndcg = self.evaluate(test_data)
            logging.info(f"[Testing Round {round_id}] HR@10: {hr:.4f}, NDCG@10: {ndcg:.4f}")

            logging.info("Validating phase...")
            val_hr, val_ndcg = self.evaluate(validate_data)
            logging.info(f"[Validating Round {round_id}] HR@10: {val_hr:.4f}, NDCG@10: {val_ndcg:.4f}")

            if val_hr > best_val_hr:
                best_val_hr = val_hr
                logging.info(f"New best Validation HR: {best_val_hr:.4f}! Saving checkpoint.")
                save_checkpoint(self.global_embedding,
                                os.path.join('checkpoints', f"{self.config['algorithm']}_{self.config['dataset']}_best.pth"))

