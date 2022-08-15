import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
import shutil
from torch.utils.data import DataLoader, Dataset
import datasets, transformers
from transformers import TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
import os
import gc
import re
import ast
import sys
import copy
import json
import time
import math
import shutil
import string
import pickle
import random
import joblib
import itertools
from pathlib import Path
import warnings
# from ipython import display
import scipy as sp
from tqdm.auto import tqdm
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
import torch
from torch.cuda.amp import autocast
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader, Dataset
# os.system('pip uninstall -y transformers')
# os.system('pip uninstall -y tokenizers')
# os.system('python -m pip install --no-index --find-links=../input/pppm-pip-wheels transformers')
# os.system('python -m pip install --no-index --find-links=../input/pppm-pip-wheels tokenizers')
import tokenizers
import transformers
from text_unidecode import unidecode
from typing import Dict, List, Tuple
import codecs
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

print(f"tokenizers.__version__: {tokenizers.__version__}")
print(f"transformers.__version__: {transformers.__version__}")

TOKENIZERS_PARALLELISM = True

gc.collect()
warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"torch.__version__: {torch.__version__}")

INPUT_DIR = '../data/'


class CFG:
    num_workers = 1
    path = "model_zoo/feedback-deberta-large-051/"
    config_path = path + 'config.pth'
    model = "microsoft/deberta-large"
    batch_size = 16
    fc_dropout = 0.2
    target_size = 3
    max_len = 512
    seed = 42
    n_fold = 4
    trn_fold = [i for i in range(n_fold)]
    gradient_checkpoint = False


FOLDS = 5
lr = 2e-5
EPOCHS = 2
SEED = 2018
MAX_LEN = 512
BATCH_SIZE = 8
accumulation_steps = 4
model_path = 'data/robertalarge/'
y_cols = ['discourse_effectiveness']