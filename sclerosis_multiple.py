from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score

from common import utils
import numpy as np
import pandas as pd
from discrimination.discrimination_utils import load_data, results_to_csv, roc_auc_score_multiclass
from discrimination.svm_utils import train_svm
from common.dimension_reduction import ReduceDims

config = utils.load_config('config/config_sm.yml')  # loading configuration
config_bea = utils.load_config('config/config_bea16k.yml')  # loading configuration
shuffle_data = config['shuffle_data']  # whether to shuffle the training data
label_file = config['paths']['to_labels']  # path to the labels of the dataset

data = load_data(config=config)  # loading data
df_labels = pd.read_csv(label_file)  # loading labels
data['label'] = df_labels.label.values  # adding labels to data
# Shuffling data if needed
if shuffle_data:
    data = data.sample(frac=1).reset_index(drop=True)

x_train, y_train = data.iloc[:, :-1].values, data.label.values  # train and labels

pca_flag = False
if pca_flag:
    # APPLY PCA!
    # Train PCA model using embeddings got from bea-train-flat (57k files fo each emb type: convs and hiddens)
    # Transform the dataset using the fitted PCA model
    reduce_dims = ReduceDims(config_bea=config_bea)
    pca = reduce_dims.fit_pca()  # train PCA
    x_train = pca.transform(x_train)  # transform (reduce dimensionality)
    print("New shape:", x_train.shape)

# train
# print("Checkpoint utilized:", config['pretrained_model_details']['checkpoint_path'])
print("Using", config['discrimination']['emb_type'])
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


