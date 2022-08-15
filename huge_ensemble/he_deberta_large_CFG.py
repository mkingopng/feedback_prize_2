from he_imports import *


class CFG:
    seed = 42
    n_fold = 4
    max_len = 512
    batch_size = 32
    num_workers = 2
    model = "microsoft/deberta-large"
    path = "model_zoo/feedback-deberta-large-051/"
    config_path = "model_zoo/feedback-deberta-large-051/" + 'config.pth'


