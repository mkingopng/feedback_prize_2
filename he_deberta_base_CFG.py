import os
import gc
import torch
import pickle
import codecs
import gensim
import numpy as np
import pandas as pd
import pickle as pkl
import torch.nn as nn
from tqdm import tqdm
import seaborn as sns
import torch.nn as nn
import lightgbm as lgb
from scipy import sparse
from typing import Tuple
import torch.nn.functional as F
from sklearn.metrics import log_loss
from text_unidecode import unidecode
from typing import Dict, List, Tuple
from transformers import AutoTokenizer
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from transformers import AutoModel, AutoTokenizer, AutoConfig
import warnings

warnings.simplefilter('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

WEIGHTS = [0.20, 0.65, 0.05, 0.1]
MODEL_NAMES = ['deberta', 'deberta_large', 'roberta', 'lgbm']

INPUT_DIR = '../input/feedback-prize-effectiveness/'


class CFG:
    CVs = []
    seed = 42
    lr = 3e-5
    epochs = 3
    n_fold = 5
    apex = True
    fast = True
    AMP = False
    n_splits = 5
    train = True
    wandb = False
    max_len = 512
    dropout = 0.1
    min_lr = 1e-6
    batch_size = 8
    freezing = True
    print_freq = 50
    target_size = 3
    num_workers = 0
    num_cycles = 0.5
    n_accumulate = 1
    scheduler = 'cosine'
    weigth_decay = 0.01
    num_warmup_steps = 0
    trn_fold = [0, 1, 2, 3, 4]
    gradient_checkpointing = True
    model = 'model_zoo/deberta-v3-base/deberta-v3-base'

