import numpy as np
import sklearn as sk
from dwd.socp_dwd import DWD
from scipy.stats import stats
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


def test_linearsvm_cpu(X, y, X_eval, best_c):
    svc = sk.svm.LinearSVC(class_weight="balanced", C=best_c, max_iter=100000)
    svc.fit(X, y)
    y_prob = svc._predict_proba_lr(X_eval)
    return y_prob


def train_linearSVR_cpu(X, y, X_eval, c):
    svc = svm.NuSVR(kernel='linear', C=c, verbose=0, max_iter=100000)
    svc.fit(X, y)
    y_prob = svc.predict(X_eval)
    return y_prob


def test_linearSVR_cpu(X, y, X_eval, best_c):
    svc = svm.NuSVR(kernel='linear', C=best_c, verbose=0, max_iter=100000)
    svc.fit(X, y)
    y_prob = svc.predict(X_eval)
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



def feat_selection_spearman(x, y, keep_feats):
    corr_list = []
    for idx_column_feature in range(len(x[1])):
        corr, _ = stats.pearsonr(y, x[:, idx_column_feature])  # take corr
        # print("y", y.shape)
        # print("x", x[:, idx_column_feature].shape)
        corr_list.append(abs(corr))  # collect the corr (abs) values
    ordered_asc = sorted(corr_list, reverse=True)  # sort desc the corr list
    min_corr = ordered_asc[0:keep_feats]  # pick n most correlating # min_corr = # n higher correlated
    indices = [index for index, item in enumerate(corr_list) if
               item in set(min_corr)]  # take the indices that correspond to the min_corr values in the corr_list
    return indices


def loocv_NuSVR_cpu_pearson(X, Y, c, kernel, keep_feats, feat_selection=False):
    svc = svm.NuSVR(kernel=kernel, C=c, verbose=0, max_iter=100000)
    loo = LeaveOneOut()

    array_preds = np.zeros((len(Y),))
    list_trues = np.zeros((len(Y),))

    for train_index, test_index in loo.split(X=X):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        # doing feature selection based on the most correlated features
        if feat_selection:
            selected_idx_train = feat_selection_spearman(x_train, y_train, keep_feats)
            x_train = x_train[:, selected_idx_train]
            x_test = x_test[:, selected_idx_train]
            # keepfeats = find(corr >= min_corr)
            # print(x_train_selected.shape)

        svc.fit(x_train, y_train)
        pred = svc.predict(x_test)
        array_preds[test_index] = pred
        list_trues[test_index] = y_test

    return array_preds, list_trues


# Classifier switcher
def train_svm(svm_type, C, X, y, X_eval=None, **kwargs):
    switcher = {
        'linear': lambda: train_linearsvm_cpu(X, y, X_eval, C),
        # 'linear-gpu': lambda: train_linearsvm_gpu(X, y, X_eval, C),
        'rbf': lambda: train_RBF_cpu(X, y, X_eval, C),
        'linear-loocv': lambda: train_loocv_svm(X, y, C),
        'rbf-loocv': lambda: train_loocv_dwd(X, y, C),
        'nusvr-loocv': lambda: loocv_NuSVR_cpu_pearson(X, y, C, **kwargs),
        'linearsvr': lambda: train_linearSVR_cpu(X, y, X_eval, C),
    }
    return switcher.get(svm_type, lambda: "Error {} is not an option! Choose from: \n {}.".format(svm_type,
                                                                                                  switcher.keys()))()
