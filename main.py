import logging
import datetime
import os
import pandas as pd
import numpy as np

from data.dataloader import SampleGenerator

from servers.pfedrec_server import PFedRec_Server
from servers.gpfedrec_server import GPFedRec_Server
from utils.parser import parse_args

def setup_logging(config):
    log_dir = 'logs/'
    os.makedirs(log_dir, exist_ok=True)
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_file = os.path.join(log_dir, f"{config['algorithm']}_{config['dataset']}_{current_time}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def load_data(config):
    dataset = config['dataset']
    data_dir = os.path.join("data", dataset, "ratings.dat")

    if dataset == 'ml-1m':
        rating = pd.read_csv(data_dir, sep='::', header=None,
                             names=['userId', 'itemId', 'rating', 'timestamp'],
                             engine='python')
        config['num_users'] = 6040
        config['num_items'] = 3706
    elif dataset == 'ml-100k':
        rating = pd.read_csv(data_dir, sep=',', header=None,
                             names=['userId', 'itemId', 'rating', 'timestamp'],
                             engine='python')
        config['num_users'] = 943
        config['num_items'] = 1682
    elif dataset == 'lastfm-2k':
        rating = pd.read_csv(data_dir, sep=',', header=None,
                             names=['userId', 'itemId', 'rating', 'timestamp'],
                             engine='python')
        config['num_users'] = 1892
        config['num_items'] = 17632
    elif dataset == 'amazon':
        rating = pd.read_csv(data_dir, sep=',', header=None,
                             names=['userId', 'itemId', 'rating', 'timestamp'],
                             engine='python')
    else:
         raise ValueError(f"Unknown dataset: {dataset}")

    # Reindex
    user_id = rating[['userId']].drop_duplicates().reindex()
    user_id['new_userId'] = np.arange(len(user_id))
    rating = pd.merge(rating, user_id, on=['userId'], how='left')

    item_id = rating[['itemId']].drop_duplicates()
    item_id['new_itemId'] = np.arange(len(item_id))
    rating = pd.merge(rating, item_id, on=['itemId'], how='left')

    rating = rating[['new_userId', 'new_itemId', 'rating', 'timestamp']]
    rating.columns = ['userId', 'itemId', 'rating', 'timestamp']

    config['num_users'] = rating['userId'].nunique()
    config['num_items'] = rating['itemId'].nunique()

    return SampleGenerator(rating)


if __name__ == '__main__':
    args = parse_args()
    config = vars(args)
    setup_logging(config)

    logging.info(f"Configuration: {config}")

    sample_generator = load_data(config)

    if config['algorithm'] == 'gpfedrec':
        server = GPFedRec_Server(config)
    elif config['algorithm'] == 'pfedrec':
        server = PFedRec_Server(config)
    else:
        raise ValueError(f"Unknown algorithm: {config['algorithm']}")

    server.train(sample_generator)











