import gc
import os
import sys
import time
import torch
import pickle
import random
import numpy as np
import transformers
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import log_loss
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from transformers import AutoModel, AutoTokenizer, AdamW, \
    get_linear_schedule_with_warmup
import warnings

warnings.simplefilter('ignore')


# utils funcitons
def fetch_essay(essay_id: str):
    """

    :param essay_id:
    :return:
    """
    essay_path = os.path.join('data/train/', essay_id + '.txt')

    essay_text = open(essay_path, 'r').read()

    return essay_text


def seed_everything(seed):
    """

    :param seed:
    :return:
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# configurations
FOLDS = 5
lr = 2e-5
EPOCHS = 2
SEED = 2018
MAX_LEN = 512
BATCH_SIZE = 4  # out of memory error at batch size 8
accumulation_steps = 4
seed_everything(SEED)
