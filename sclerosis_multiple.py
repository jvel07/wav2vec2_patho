from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score

from common import utils
import numpy as np
from discrimination.discrimination_utils import load_data, results_to_csv, roc_auc_score_multiclass
from discrimination.svm_utils import train_svm

config = utils.load_config('config/config_sm.yml')  # loading configuration
df = load_data(config=config)  # loading data
x_train, y_train = df.iloc[:, :-1].values, df.label.values  # loading data
# out_results =
# APPLY PCA!

# train
# print("Checkpoint utilized:", config['pretrained_model_details']['checkpoint_path'])
list_c = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1]
for c in list_c:
    array_preds, array_trues, array_probs = train_svm(svm_type='linear-loocv', C=c, X=x_train, y=np.ravel(y_train))

    acc = accuracy_score(array_trues, array_preds)
    # auc = roc_auc_score(array_trues, array_probs, labels=np.unique(y_train))
    auc = roc_auc_score(array_trues, array_probs[:, 1])
    # aucs = roc_auc_score_multiclass(actual_class=array_trues, pred_class=array_preds)

    f1 = f1_score(array_trues, array_preds)
    prec = precision_score(array_trues, array_preds)
    rec = recall_score(array_trues, array_preds)
    # results_to_csv(file_name=out_results,
    #                list_values=,
    #                list_columns=)
    print("with", c, "acc:", acc, " f1:", f1, " prec:", prec, " recall:", rec, 'AUC:', auc)#, 'auc-c0:', aucs[1],
          # 'auc-c1:', aucs[2])


