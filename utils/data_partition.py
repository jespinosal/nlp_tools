from sklearn.model_selection import StratifiedKFold


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
