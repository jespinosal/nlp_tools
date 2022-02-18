import os
import numpy as np
import torch
import random
from sklearn.model_selection import StratifiedKFold
from collections import Counter


def set_seed(seed=42):
    """
    Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY
    :param seed:
    :return:
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_device():
    """

    :return:
    """
    return torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')


def set_cv_dataset_partitions(df, stratify_column='y', k_folds=10, seed=100):
    """

    :param df:
    :param stratify_column:
    :param k_folds:
    :param seed:
    :return:
    """
    df = df.copy()
    df = df.reset_index(drop=True)
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    for k_fold, (train_index, test_index) in enumerate(skf.split(X=df, y=df[stratify_column])):
        df.loc[test_index, 'k_fold'] = int(k_fold)
        df.loc[test_index, 'index_id'] = test_index
    df['k_fold'] = df['k_fold'].astype(int)
    return df
