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
import scipy as sp
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast
import tokenizers
import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from text_unidecode import unidecode
from typing import Dict, List, Tuple
import codecs
import wandb

warnings.filterwarnings("ignore")

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

print(f"torch.__version__: {torch.__version__}")
print(f"tokenizers.__version__: {tokenizers.__version__}")
print(f"transformers.__version__: {transformers.__version__}")

# TOKENIZERS_PARALLELISM=true

gc.collect()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


