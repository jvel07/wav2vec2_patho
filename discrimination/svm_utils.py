import numpy as np
import sklearn as sk
from dwd.socp_dwd import DWD
from sklearn import svm, preprocessing
from sklearn.model_selection import LeaveOneOut


def standardize_data_jo(x_train, x_test):
    std_scaler = preprocessing.MinMaxScaler()
    x_train = std_scaler.fit_transform(x_train)
    x_test = std_scaler.transform(x_test)
    return x_train, x_test


def train_linearsvm_cpu(X, y, X_eval, c):
    svc = sk.svm.LinearSVC(class_weight="balanced", C=c, max_iter=100000)
    svc.fit(X, y)
    y_prob = svc._predict_proba_lr(X_eval)
    return y_prob


# def train_linearsvm_gpu(X, y, X_eval, c):
#     svc = thunder(kernel='linear', C=c, class_weight='balanced',
#                   probability=True, max_iter=100000, gpu_id=0)
#     svc.fit(X, y)
#     y_prob = svc.predict_proba(X_eval)
#     return y_prob


def train_RBF_cpu(X, y, X_eval, c):
    svc = sk.svm.SVC(kernel='rbf', gamma='scale', C=c, probability=True, verbose=0, max_iter=100000,
                     class_weight='balanced')
    svc.fit(X, y)
    y_prob = svc.predict_proba(X_eval)
    return y_prob


def train_loocv_svm(x, y, c):
    loo = LeaveOneOut()
    svc = svm.LinearSVC(C=c, class_weight='balanced', max_iter=100000)
    # svc = sk.svm.SVC(kernel='linear', C=c, probability=True, verbose=0, max_iter=100000, class_weight='balanced')

    array_preds = np.zeros((len(y),))
    array_trues = np.zeros((len(y),))
    array_probs = np.zeros((len(y), len(np.unique(y))))

    for train_index, test_index in loo.split(X=x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        x_train, x_test = standardize_data_jo(x_train, x_test)
        svc.fit(x_train, y_train)

        pred = svc.predict(x_test)
        # y_prob = svc.predict_proba(x_test)
        y_prob = svc._predict_proba_lr(x_test)

        array_preds[test_index] = pred
        array_trues[test_index] = y_test
        array_probs[test_index] = y_prob

    return array_preds, array_trues, array_probs


def train_loocv_dwd(x, y, c):
    loo = LeaveOneOut()
    # svc = svm.NuSVC(kernel='poly', class_weight='balanced', max_iter=100000, probability=True)
    svc = sk.svm.SVC(kernel='linear', C=c, probability=True, verbose=0, max_iter=100000, class_weight='balanced')
    # svc = DWD(C=c, solver_kws=)

    array_preds = np.zeros((len(y),))
    array_trues = np.zeros((len(y),))
    array_probs = np.zeros((len(y), len(np.unique(y))))

    for train_index, test_index in loo.split(X=x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        x_train, x_test = standardize_data_jo(x_train, x_test)
        svc.fit(x_train, y_train)

        pred = svc.predict(x_test)
        # y_prob = svc.predict_proba(x_test)
        # y_prob = svc.pre(x_test)

        array_preds[test_index] = pred
        array_trues[test_index] = y_test
        # array_probs[test_index] = y_prob

    return array_preds, array_trues



# Classifier switcher
def train_svm(svm_type, C, X, y, X_eval=None):
    switcher = {
        'linear': lambda: train_linearsvm_cpu(X, y, X_eval, C),
        # 'linear-gpu': lambda: train_linearsvm_gpu(X, y, X_eval, C),
        'rbf': lambda: train_RBF_cpu(X, y, X_eval, C),
        'linear-loocv': lambda: train_loocv_svm(X, y, C),
        'rbf-loocv': lambda: train_loocv_dwd(X, y, C)
    }
    return switcher.get(svm_type, lambda: "Error {} is not an option! Choose from: \n {}.".format(svm_type,
                                                                                                  switcher.keys()))()
