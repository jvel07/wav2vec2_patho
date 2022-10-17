import numpy as np
import sklearn as sk
from sklearn import svm
from sklearn.model_selection import LeaveOneOut


def train_linearsvm_cpu(X, y, X_eval, c):
    svc = sk.svm.LinearSVC(class_weight="balanced", C=c, max_iter=100000)
    svc.fit(X, y)
    y_prob = svc._predict_proba_lr(X_eval)
    return y_prob


def train_linearsvm_gpu(X, y, X_eval, c):
    svc = thunder(kernel='linear', C=c, class_weight='balanced',
                  probability=True, max_iter=100000, gpu_id=0)
    svc.fit(X, y)
    y_prob = svc.predict_proba(X_eval)
    return y_prob


def train_RBF_cpu(X, y, X_eval, c):
    svc = sk.svm.SVC(kernel='rbf', gamma='scale', C=c, probability=True, verbose=0, max_iter=100000,
                     class_weight='balanced')
    svc.fit(X, y)
    y_prob = svc.predict_proba(X_eval)
    return y_prob


def train_loocv_svm(X, y, c):
    loo = LeaveOneOut()
    svc = svm.LinearSVC(C=c, class_weight='balanced', max_iter=100000)
    array_posteriors = np.zeros((len(y), len(np.unique(y))))

    list_trues = []
    list_preds = []

    for train_index, test_index in loo.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        svc.fit(X_train, y_train)

        y_prob = svc._predict_proba_lr(X_test)
        array_posteriors[test_index] = y_prob
        # preds = np.argmax(array_posteriors, axis=1)
        preds = array_posteriors[:, 1]
        list_preds.append(preds.round())
        list_trues.append(y_test)

    return list_preds, np.squeeze(list_trues), array_posteriors


# Classifier switcher
def train_svm(svm_type, X, y, X_eval, C):
    switcher = {
        'linear': lambda: train_linearsvm_cpu(X, y, X_eval, C),
        'linear-gpu': lambda: train_linearsvm_gpu(X, y, X_eval, C),
        'rbf': lambda: train_RBF_cpu(X, y, X_eval, C)
    }
    return switcher.get(svm_type, lambda: "Error {} is not an option! Choose from linear and rbf.".format(svm_type))()
