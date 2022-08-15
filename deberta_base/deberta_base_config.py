import os
import gc
import math
import time
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from transformers import AutoModel, AutoConfig, AutoTokenizer, AdamW, DataCollatorWithPadding
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

warnings.simplefilter('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

INPUT_DIR = '../data/feedback_prize_with_essay_text/'
OUTPUT_DIR = '../baseline/'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


class CFG:
    wandb = False
    apex = True
    model = 'microsoft/deberta-v3-base'
    seed = 42
    n_splits = 4
    max_len = 512
    dropout = 0.2
    target_size = 3
    n_accumulate = 1
    print_freq = 100
    min_lr = 1e-6
    scheduler = 'cosine'
    batch_size = 8
    num_workers = 3
    lr = 2e-5
    weigth_decay = 0.01
    epochs = 4
    n_fold = 4
    trn_fold = [0, 1, 2, 3]
    train = True
    num_warmup_steps = 0
    num_cycles = 0.5
    debug = False  # switch to True for debugging
    debug_ver2 = False  # alternate debugging mode


if CFG.debug:
    CFG.epochs = 2
    CFG.trn_fold = [0, 1]
    CFG.print_freq = 10

if CFG.debug_ver2:
    CFG.epochs = 1
    CFG.trn_fold = [0, 1]

train = pd.read_csv(INPUT_DIR + 'train_all.csv')
test = pd.read_csv(INPUT_DIR + 'test_all.csv')
