import ast
import numpy as np
import pandas as pd
import cleanlab


def get_k_fold_class_probs(df, transformation, baseline_model, k_folds=10,
                           text_column='comment_text', label_column='sentiment') -> pd.Dataframe:
    """

    :param df:
    :param transformation:
    :param baseline_model:
    :param k_folds:
    :param text_column:
    :param label_column:
    :return:
    """

    # Training with noise
    df = df.copy()
    transformation.fit(df.comment_text)
    for k_fold in range(k_folds):
        # Prepare k fold data
        X_train = transformation.transform(df.loc[df.k_fold != k_fold, text_column])
        y_train = df.loc[df.k_fold != k_fold, label_column]
        X_test = transformation.transform(df.loc[df.k_fold == k_fold, text_column])
        y_test = df.loc[df.k_fold == k_fold, label_column]

        # Train k fold partition
        model = baseline_model()
        model.fit(X_train, y_train)

        # Perform k fold inference
        predictions_labels = model.predict(X_test)
        predictions_probs = model.predict_proba(X_test)

        # Save probs
        df.loc[df.k_fold == k_fold, 'predictions_probs'] = [str(prob) for prob in predictions_probs.tolist()]
        df.loc[df.k_fold == k_fold, 'predictions_labels'] = predictions_labels

    df['predictions_probs'] = df['predictions_probs'].apply(lambda x: ast.literal_eval(x))
    return df


def get_noisy_labels(pyx, s, multi_label):

    # Estimate the  P(s,y) counting of the latent joint distribution of true and noisy labels using observed s and
    # predicted probabilities psx
    # cj refers to confident join subset of the joint distribution of noisy and true labels P_{s,y} it's estimated by
    # compute_confident_joint by counting confident examples.
    cj, cj_only_label_error_indices = cleanlab.latent_estimation.compute_confident_joint(s=s, psx=pyx, return_indices_of_off_diagonals=True,
                                                                                         thresholds=None, calibrate=True, multi_label=multi_label)

    # If you want to get label errors using cj_only method --> The most restrictive case that involve deleting all the "error" cases.
    cj_only_bool_mask = np.zeros(len(s), dtype=bool)
    for idx in cj_only_label_error_indices:
        cj_only_bool_mask[idx] = True

    #  To get all the "noisy" samples sorted according the noyse probs calculated on cj matrix.
    #  Internally it use np.argsort -> default is ascending,
    # 'prob_given_margin' the worst cases are the first (with less correct label probability)
    # 'normalized_margin' the worst cases are the first because the margin with negative ones represen wors scenario->
    # self_confidence - psx[label_errors_bool].max(axis=1)
    label_errors_idx_sorted = cleanlab.pruning.order_label_errors(label_errors_bool=cj_only_bool_mask,
                                                                  psx=pyx,
                                                                  labels=s,
                                                                  sorted_index_method='normalized_margin')

    # To filter the noisiest "noisy" samples according the defined criteria (Using Freq, Ratios, or custome methods)
    label_errors_idx = cleanlab.pruning.get_noise_indices(
        s=s,
        psx=pyx,
        confident_joint=cj,
        prune_method='both',
        sorted_index_method='normalized_margin',  # ['prob_given_label', 'normalized_margin']
        frac_noise=1.0,
        multi_label=False)

    return cj_only_bool_mask, label_errors_idx_sorted, label_errors_idx