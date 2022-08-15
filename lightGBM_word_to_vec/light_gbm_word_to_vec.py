import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import log_loss
import gensim
from scipy import sparse
import lightgbm as lgb
import warnings


warnings.filterwarnings('ignore')


class CFG:
    seed = 42
    n_folds = 4


INPUT_DIR = "../data/"


def get_train_essay(essay_id):
    essay_path = os.path.join(INPUT_DIR, f'train/{essay_id}.txt')
    essay_text = open(essay_path, 'r').read()
    return essay_text


def get_test_essay(essay_id):
    essay_path = os.path.join(INPUT_DIR, f'test/{essay_id}.txt')
    essay_text = open(essay_path, 'r').read()
    return essay_text


train = pd.read_csv(INPUT_DIR + 'train.csv')
test = pd.read_csv(INPUT_DIR + 'test.csv')
train['essay_text'] = train['essay_id'].apply(get_train_essay)
test['essay_text'] = test['essay_id'].apply(get_test_essay)
print(train.head())
print(test.head())


# seed
def set_seed(seed=42):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


set_seed(CFG.seed)

# cv
effectiveness_map = {
    'Ineffective': 0,
    'Adequate': 1,
    'Effective': 2
}

train['target'] = train['discourse_effectiveness'].map(effectiveness_map)

sgkf = StratifiedGroupKFold(
    n_splits=CFG.n_folds,
    shuffle=True,
    random_state=CFG.seed
)

for fold, (_, val_idx) in enumerate(sgkf.split(X=train, y=train['target'], groups=train.essay_id)):
    train.loc[val_idx, 'kfold'] = fold

print(train.head())

train.groupby('kfold')['discourse_effectiveness'].value_counts()

# word to vec
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(
    'model_zoo/google_news/GoogleNews-vectors-negative300.bin',
    binary=True
)

print(word2vec_model.vectors.shape)


def avg_feature_vector(sentence, model, num_features):
    words = sentence.replace('\n', " ").replace(',', ' ').replace('.', " ").split()
    feature_vec = np.zeros((num_features,), dtype="float32")
    i = 0
    for word in words:
        try:
            feature_vec = np.add(feature_vec, model[word])
        except KeyError as error:
            feature_vec
            i = i + 1
    if len(words) > 0:
        feature_vec = np.divide(feature_vec, len(words) - i)
    return feature_vec


# LightGBM

params = {
    "objective": 'multiclass',
    'metric': 'multi_logloss',
    'boosting': 'gbdt',
    'num_class': 3,
    'is_unbalance': True,
    'learning_rate': 0.05,
    'lambda_l2': 0.0256,
    'num_leaves': 52,
    'max_depth': 10,
    'feature_fraction': 0.503,
    'bagging_fraction': 0.741,
    'bagging_freq': 8,
    'bagging_seed': 10,
    'min_data_in_leaf': 10,
    'verbosity': -1,
    'random_state': 42
}

num_rounds = 5000


if __name__ == "__main__":

    oof_score = 0
    y_test_pred = np.zeros((test.shape[0], 3))

    for fold in range(CFG.n_folds):
        print(f'=============fold:{fold}==================')
        train_fold = train[train['kfold'] != fold].reset_index(drop=True)
        valid_fold = train[train['kfold'] == fold].reset_index(drop=True)

        # word2vec

        # discourse_text
        word2vec_train_disc_text = np.zeros((len(train_fold.index), 300), dtype="float32")
        word2vec_valid_disc_text = np.zeros((len(valid_fold.index), 300), dtype="float32")
        word2vec_test_disc_text = np.zeros((len(test.index), 300), dtype="float32")
        for i in range(len(train_fold.index)):
            word2vec_train_disc_text[i] = avg_feature_vector(train_fold["discourse_text"][i], word2vec_model, 300)
        for i in range(len(valid_fold.index)):
            word2vec_valid_disc_text[i] = avg_feature_vector(valid_fold["discourse_text"][i], word2vec_model, 300)
        for i in range(len(test.index)):
            word2vec_test_disc_text[i] = avg_feature_vector(test["discourse_text"][i], word2vec_model, 300)

        # essay_text
        word2vec_train_essay_text = np.zeros((len(train_fold.index), 300), dtype="float32")
        word2vec_valid_essay_text = np.zeros((len(valid_fold.index), 300), dtype="float32")
        word2vec_test_essay_text = np.zeros((len(test.index), 300), dtype="float32")
        for i in range(len(train_fold.index)):
            word2vec_train_essay_text[i] = avg_feature_vector(train_fold["essay_text"][i], word2vec_model, 300)
        for i in range(len(valid_fold.index)):
            word2vec_valid_essay_text[i] = avg_feature_vector(valid_fold["essay_text"][i], word2vec_model, 300)
        for i in range(len(test.index)):
            word2vec_test_essay_text[i] = avg_feature_vector(test["essay_text"][i], word2vec_model, 300)

        # OneHot
        ohe = OneHotEncoder()
        train_type_ohe = sparse.csr_matrix(ohe.fit_transform(train_fold['discourse_type'].values.reshape(-1, 1)))
        valid_type_ohe = sparse.csr_matrix(ohe.transform(valid_fold['discourse_type'].values.reshape(-1, 1)))
        test_type_ohe = sparse.csr_matrix(ohe.transform(test['discourse_type'].values.reshape(-1, 1)))

        # merge
        Xtrain_word2vec = sparse.hstack((train_type_ohe, word2vec_train_disc_text, word2vec_train_essay_text))
        Xvalid_word2vec = sparse.hstack((valid_type_ohe, word2vec_valid_disc_text, word2vec_valid_essay_text))
        test_word2vec = sparse.hstack((test_type_ohe, word2vec_test_disc_text, word2vec_test_essay_text))

        # lgbm
        lgtrain = lgb.Dataset(Xtrain_word2vec, label=train_fold['target'].ravel())
        lgvalidation = lgb.Dataset(Xvalid_word2vec, label=valid_fold['target'].ravel())

        model = lgb.train(
            params,
            lgtrain,
            num_rounds,
            valid_sets=[lgtrain, lgvalidation],
            early_stopping_rounds=100,
            verbose_eval=100
        )

        y_pred = model.predict(Xvalid_word2vec, num_iteration=model.best_iteration)

        y_test_pred += model.predict(test_word2vec, num_iteration=model.best_iteration)

        score = log_loss(valid_fold['target'], y_pred)

        oof_score += score

        print(f'Fold:{fold},valid score:{score}')

    y_test_pred = y_test_pred / float(CFG.n_folds)
    oof_score /= float(CFG.n_folds)
    print("Aggregate OOF Score: {}".format(oof_score))
