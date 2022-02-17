import os
import numpy as np
import torch
import random
from sklearn.model_selection import StratifiedKFold
from collections import Counter

def set_seed(seed=42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
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
    return torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')


def set_cv_dataset_partitions(df, n_folds, seed, stratify_column):
    df = df.copy()
    skf = StratifiedKFold(n_splits=n_folds,
                          shuffle=True,
                          random_state=seed)

    for k_fold, (index_train_, index_test_) in enumerate(skf.split(X=df, y=df[stratify_column])):
        df.loc[index_test_ , "kfold"] = int(k_fold)

    df["kfold"] = df["kfold"].astype(int)
    print(f"Dataset size lenght {len(df)}")
    print(f"Test label size for k fold partitions {Counter(df['kfold'])}." )
    return df