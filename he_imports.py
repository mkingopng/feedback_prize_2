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

pd.set_option('display.precision', 4)

cm = sns.light_palette('green', as_cmap=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

COMP_DIR = 'data'
