import copy
import torch

class Base_Client(object):
    def __init__(self, user_id, config, model):
        self.user_id = user_id
        self.config = config
        self.local_model = copy.deepcopy(model)
        if self.config.get('use_cuda', False):
            self.local_model.cuda()

    def set_params(self, model_params):
        if model_params is not None:
             self.local_model.load_state_dict(model_params, strict=False)

    def get_params(self):
        return self.local_model.state_dict()
