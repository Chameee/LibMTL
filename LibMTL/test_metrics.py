from sklearn.metrics import roc_auc_score, precision_score, accuracy_score, f1_score, recall_score


def calculate_f1_score(y_true, y_pred):
    return f1_score(y_true, y_pred)


def calculate_accuracy_score(y_true, y_pred):
    return accuracy_score(y_true, y_pred)
