import os
import pandas as pd
import numpy as np
import torch
import random
from sklearn.model_selection import StratifiedKFold
from config import Config
from torch.utils.data import Dataset
from transformers import AutoTokenizer


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


def get_data_loaders(df: pd.DataFrame, df_torch_parser: Dataset, config:Config,
                     tokenizer: AutoTokenizer, kfold: int, text_column: str):

    df_torch_test = df_torch_parser(df=df[df[config.PARITION_COLUMN]==kfold],
                                    tokenizer=tokenizer,
                                    config=config,
                                    text_column=text_column)

    df_torch_train = df_torch_parser(df=df[df[config.PARITION_COLUMN]!=kfold],
                                     tokenizer=tokenizer,
                                     config=config,
                                     text_column=text_column)

    data_loader_train = torch.utils.data.DataLoader(df_torch_train,
                                                    batch_size=config.TRAIN_BATCH_SIZE,
                                                    num_workers=2,
                                                    shuffle=True,
                                                    pin_memory=True,
                                                    drop_last=False)

    data_loader_test = torch.utils.data.DataLoader(df_torch_test,
                                                   batch_size=config.TEST_BATCH_SIZE,
                                                   num_workers=2,
                                                   shuffle=True,
                                                   pin_memory=True,
                                                   drop_last=False)

    return data_loader_train, data_loader_test