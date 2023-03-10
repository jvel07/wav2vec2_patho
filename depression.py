import os

import torch
from scipy.stats import stats
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, mean_squared_error
from torch import nn, optim
from torch.utils.data import DataLoader

from common import utils, fit_scaler, results_to_csv, DataBuilder
import numpy as np
import pandas as pd

from common.metrics import calculate_eer, calculate_sensitivity_specificity
from discrimination.discrimination_utils import load_data, roc_auc_score_multiclass, check_model_used, load_joint_embs, \
    load_baseline_feats, reduce_dimensions_vae, reduce_dimensions_basic_autoencoder
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

if dim_reduction == 'PCA':
    # APPLY PCA!
    # Train PCA model using embeddings got from bea-train-flat (57k files fo each emb type: convs and hiddens)
    # Transform the dataset using the fitted PCA model
    reduce_dims = ReduceDims(config_bea=config)
    # if not scaler_type
    #     scaler = fit_scaler(config_bea, bea_train_flat)
    #     bea_train_flat = scaler.transform(bea_train_flat)

    # pca = reduce_dims.fit_pca(bea_train_flat)  # train PCA
    pca = reduce_dims.fit_pca(x_train)  # train PCA
    x_train = pca.transform(x_train)  # transform (reduce dimensionality)
    print("New shape:", x_train.shape)
    size_reduced = x_train.shape[1]
    variance = config['dimension_reduction']['pca']['n_components']

elif dim_reduction == 'autoencoder':
    x_train = reduce_dimensions_basic_autoencoder(x_train=x_train, config=config)

elif dim_reduction == 'vae':
    x_train = reduce_dimensions_vae(x_train=x_train, bea_train_flat=bea_train_flat, config=config)

else:
    pass

# Train SVM
print("Using", config['discrimination']['emb_type'])
list_c = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1]
df = pd.DataFrame(columns=['c', 'pearsons', 'uar', 'spec', 'sens', 'auc', 'F1', 'RMSE'])

corr_scores = []
uar_scores = []
auc_scores = []
f1_scores = []
spec_scores = []
sens_scores = []
eer_scores = []
for c in list_c:
    # TRY ALSO MLP OR KNN!!!!
    array_preds, array_trues = train_svm(svm_type='nusvr-loocv', X=x_train, y=y_train, C=c,
                                         kernel='linear', keep_feats=None, feat_selection=False)
    # array_preds, array_trues, array_probs = train_svm(svm_type='rbf-loocv', C=c, X=x_train, y=y_train)

    corr, _ = stats.pearsonr(array_trues, array_preds)
    corr_scores.append(corr)

    # binary class
    trues_bin = np.copy(array_trues)
    trues_bin[trues_bin < 13.5] = 0
    trues_bin[trues_bin >= 13.5] = 1
    preds_bin = np.copy(array_preds)
    preds_bin[preds_bin < 13.5] = 0
    preds_bin[preds_bin >= 13.5] = 1

    # metrics
    auc = roc_auc_score(trues_bin, array_preds)
    auc_scores.append(auc)
    uar = recall_score(trues_bin, preds_bin, average='macro')
    uar_scores.append(uar)
    sens_scores.append(recall_score(trues_bin, preds_bin))
    sensitivity, specificity, accuracy = calculate_sensitivity_specificity(trues_bin, preds_bin)
    spec_scores.append(specificity)
    f1 = f1_score(trues_bin, preds_bin)
    f1_scores.append(f1)
    # eer = metrics.calculate_eer(trues_bin, preds_bin)
    eer = mean_squared_error(array_trues, array_preds, squared=False)
    eer_scores.append(eer)

    # data = {'c': c, 'acc': acc, 'f1': f1, 'prec': prec, 'recall': rec, 'auc': auc}
    dict_metrics = {'c': c, 'pearson': corr, "uar": uar, "spec": specificity, "sens": sensitivity,
                    "AUC": auc, "F1": f1, "RMSE": eer, 'Embedding': emb_type,
                    'Reduction technique': '{0}-{1}'.format(dim_reduction, str(size_reduced)), 'Model used': model_used,
                    'std': str(scaler_type), 'n_epochs': n_epochs, 'variance': variance}
    df = df.append(dict_metrics, ignore_index=True)

    print("with {}:".format(c), "corr:", corr, "uar:", uar, "spec:", specificity, "sens:", sensitivity,
          "AUC:", auc, "F1:", f1, "RMSE:", eer)
# Saving results
best_scores_df = df.iloc[[df['pearson'].idxmax()]]  # getting the best scores based on the highest AUC score.
# best_scores_df.to_csv(output_results, mode='a', header=not os.path.exists(output_results), index=False)
print(best_scores_df.values)