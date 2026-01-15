import torch
import os
import logging

def save_checkpoint(state, filename):
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)

    torch.save(state, filename)
    logging.info(f"Checkpoint saved to {filename}")

def load_checkpoint(filename):
    if os.path.isfile(filename):
        logging.info(f"Loading checkpoint from {filename}")
        return torch.load(filename)
    else:
        logging.error(f"No checkpoint found at {filename}")
        return None

