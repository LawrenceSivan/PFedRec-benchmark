import torch
import copy

class Base_Server(object):
    def __init__(self, config: dict):
        self.config = config

    def aggregate(self, client_params_list):
        if not client_params_list:
            return None

        agg_params = copy.deepcopy(client_params_list[0])
        num_clients = len(client_params_list)

        for key in agg_params.keys():
            if agg_params[key].dtype in [torch.float32, torch.float64, torch.float16]:
                 client_weights = [client[key].data for client in client_params_list]
                 stacked_weights = torch.stack(client_weights)
                 agg_params[key].data.copy_(torch.mean(stacked_weights, dim=0))

        return agg_params
