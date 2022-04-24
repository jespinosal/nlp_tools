from sklearn.metrics import classification_report, multilabel_confusion_matrix, confusion_matrix, accuracy_score, \
    recall_score, precision_score, f1_score
import pandas as pd
import numpy as np


def default_classification_report(targets,
                                  outputs,
                                  labels,
                                  report_type='multilabel'):
    """
    Generate default sklearn classification report for multilabel or multiclass data
    :param targets:
    :param outputs:
    :param labels:
    :param report_type:
    :return:
    """

    matrix_test_result_dict = classification_report(y_true=targets,
                                                  y_pred=outputs,
                                                  target_names=labels,
                                                  output_dict=True)
    labels_encoded = list(range(len(labels)))

    report = pd.DataFrame(matrix_test_result_dict).T

    if report_type == 'multilabel':
        n_confusion_matrix = multilabel_confusion_matrix(y_true=targets,
                                                         y_pred=outputs,
                                                         labels=labels_encoded)
    else:
        n_confusion_matrix = confusion_matrix(y_true=targets,
                                              y_pred=outputs,
                                              labels=labels_encoded)
    return report, n_confusion_matrix


def custom_classification_report(targets,
                                 outputs,
                                 labels,
                                 report_type='multilabel',
                                 summary=True):
    """
    Generate default sklearn classification report for multilabel or multiclass data
    :param targets:
    :param outputs:
    :param labels:
    :param report_type:
    :param summary:
    :return:
    """
    # Class Metric
    targets_ = np.array(targets)
    outputs_ = np.array(outputs)
    labels_encoded = list(range(len(labels)))

    class_metric_results = []

    if report_type == 'multilabel':
        total_samples = np.sum(targets)  # np.sum(targets) =! len(total_samples)
        n_confusion_matrix = multilabel_confusion_matrix(y_true=targets,
                                                         y_pred=outputs,
                                                         labels=labels_encoded)
    else:
        total_samples = len(targets_)
        n_confusion_matrix = confusion_matrix(y_true=targets,
                                              y_pred=outputs,
                                              labels=labels_encoded)

    for label_index, (label, label_matrix) in enumerate(zip(labels,
                                                          n_confusion_matrix)):
        targets_label = targets_[:, label_index]
        outputs_label = outputs_[:, label_index]
        false_positives = label_matrix[0][1]
        false_negatives = label_matrix[1][0]
        true_positives = label_matrix[1][1]
        true_negatives = label_matrix[0][0]
        support = np.sum(targets_label)
        support_ratio = support/total_samples
        f1_score_ = f1_score(targets_label, outputs_label, average='binary')
        precision_score_ = precision_score(targets_label, outputs_label)
        recall_score_ = recall_score(targets_label, outputs_label)
        accuracy_score_ = accuracy_score(targets_label, outputs_label)
        class_metric_results.append({
                                    'label': label,
                                    'support': support,
                                    'support_ratio': support_ratio,
                                    'f1_score': f1_score_,
                                    'precision_score': precision_score_,
                                    'recall_score': recall_score_,
                                    'accuracy_score': accuracy_score_,
                                    'tp': true_positives,
                                    'tn': true_negatives,
                                    'fp': false_positives,
                                    'fn': false_negatives
                                    })
        results = pd.DataFrame(class_metric_results)
        results = results.set_index('label')
        if summary:
            results_ = results.copy()
            results.loc['sum'] = results_.sum(axis=0).values.tolist()
            results.loc['avg'] = results_.mean(axis=0).values.tolist()
            results.loc['avg_weighted'] = [(results_['support_ratio']*results_[column]).sum() for column in
                                           results_.columns]
            results.loc['std'] = results_.std(axis=0).values.tolist()
            results.loc['min'] = results_.min(axis=0).values.tolist()
            results.loc['max'] = results_.max(axis=0).values.tolist()

    return results


if __name__ == "__name__":

    targets_ = [[1, 1, 0], [1, 1, 1], [0, 1, 1]]
    outputs_ = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]

    labels = ['a', 'b', 'c']

    report_defaults, cm = default_classification_report(targets=targets_,
                                                        outputs=outputs_,
                                                        labels=labels)

    report_custom = custom_classification_report(report_type='multilabel',
                                                  targets=targets_,
                                                  outputs=outputs_,
                                                  labels=labels,
                                                  summary=True)