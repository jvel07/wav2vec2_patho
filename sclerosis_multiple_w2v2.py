import os

import torch
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from torch import nn, optim
from torch.utils.data import DataLoader

from common import utils, fit_scaler, results_to_csv, DataBuilder
import numpy as np
import pandas as pd

from common.metrics import calculate_eer
from discrimination.discrimination_utils import load_data, roc_auc_score_multiclass, check_model_used, load_joint_embs, \
     load_baseline_feats
from discrimination.svm_utils import train_svm
from common.dimension_reduction import ReduceDims, Autoencoder, train, weights_init_uniform_rule, \
    VariationalAutoencoder, CustomLoss, train_vae

config = utils.load_config('config/config_depression.yml')  # loading configuration
# config_bea = utils.load_config('config/config_bea16k.yml')  # loading configuration for bea dataset (PCA, std)
shuffle_data = config['shuffle_data']  # whether to shuffle the training data
label_file = config['paths']['to_labels']  # path to the labels of the dataset
output_results = config['paths']['output_results']  # path to csv for saving the results

emb_type = config['discrimination']['emb_type']
checkpoint_path = config['pretrained_model_details']['checkpoint_path']
model_used = check_model_used(checkpoint_path)

# data = load_baseline_feats(path='data/10_narrative_recall/x-vectors/xvecs-10_narrative_recall-sre16-mfcc23-aug.txt', delimiter=None)
# data = load_baseline_feats(path='data/10_narrative_recall/compare/features.compare2016.rr45.10_narrative_recall.txt',
#                            delimiter=',')

if config['feature_combination']:
    data = load_joint_embs(config=config)
else:
    data = load_data(config=config)  # loading data

# bea_train_flat = load_data(config=config_bea)  # load bea embeddings
df_labels = pd.read_csv(label_file)  # loading labels
data['label'] = df_labels.label.values  # adding labels to data

# Shuffling data if needed
if shuffle_data:
    data = data.sample(frac=1).reset_index(drop=True)

x_train, y_train = data.iloc[:, :-1].values, data.label.values  # train and labels

# Standardizing data before reducing dimensions
# scaler_type = config_bea['data_scaling']['scaler_type']
scaler_type = config['data_scaling']['scaler_type']
if scaler_type != 'None':
    scaler = fit_scaler(config, x_train)
    # scaler = fit_scaler(config_bea, bea_train_flat)
    # bea_train_flat = scaler.transform(x_train)
    # bea_train_flat = scaler.transform(bea_train_flat)
    x_train = scaler.transform(x_train)
    print("Train data standardized...")

# Only modify this from the config file not here!
# dim_reduction = config_bea['dimension_reduction']['method']  # autoencoder
dim_reduction = config['dimension_reduction']['method']  # autoencoder
size_reduced = 'None'  # new dimension size after reduction
n_epochs = 'None'
variance = 'None'

# Inference with wav2vec2
print("Using", config['discrimination']['emb_type'])
df = pd.DataFrame(columns=['c', 'acc', 'f1', 'prec', 'recall', 'auc', 'eer'])

for c in list_c:
    # TRY ALSO MLP OR KNN!!!!
    array_preds, array_trues, array_probs = train_svm(svm_type='linear-loocv', C=c, X=x_train, y=np.ravel(y_train))
    # array_preds, array_trues, array_probs = train_svm(svm_type='rbf-loocv', C=c, X=x_train, y=y_train)

    acc = accuracy_score(array_trues, array_preds)
    # auc = roc_auc_score(array_trues, array_probs, labels=np.unique(y_train))
    auc = roc_auc_score(array_trues, array_probs[:, 1])
    # aucs = roc_auc_score_multiclass(actual_class=array_trues, pred_class=array_preds)

    f1 = f1_score(array_trues, array_preds)
    prec = precision_score(array_trues, array_preds)
    rec = recall_score(array_trues, array_preds)
    eer = calculate_eer(array_trues, array_preds)
    # data = {'c': c, 'acc': acc, 'f1': f1, 'prec': prec, 'recall': rec, 'auc': auc}
    dict_metrics = {'c': c, 'acc': acc, 'f1': f1, 'prec': prec, 'recall': rec, 'auc': auc, 'EER': eer, 'Embedding': emb_type,
            'Reduction technique': '{0}-{1}'.format(dim_reduction, str(size_reduced)), 'Model used': model_used,
            'std': str(scaler_type), 'n_epochs': n_epochs, 'variance': variance}
    df = df.append(dict_metrics, ignore_index=True)

    print("with", c, "acc:", acc, " f1:", f1, " prec:", prec, " recall:", rec, 'AUC:', auc, 'EER', eer)#, 'auc-c0:', aucs[1],
          # 'auc-c1:', aucs[2])

# Saving results3
best_scores_df = df.iloc[[df['auc'].idxmax()]]  # getting the best scores based on the highest AUC score.
# best_scores_df.to_csv(output_results, mode='a', header=not os.path.exists(output_results), index=False)
print(best_scores_df.values)