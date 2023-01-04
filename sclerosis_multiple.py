import os

import torch
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from torch import nn, optim

from common import utils, fit_scaler, results_to_csv
import numpy as np
import pandas as pd
from discrimination.discrimination_utils import load_data, roc_auc_score_multiclass
from discrimination.svm_utils import train_svm
from common.dimension_reduction import ReduceDims, Autoencoder, train

config = utils.load_config('config/config_sm.yml')  # loading configuration
config_bea = utils.load_config('config/config_bea16k.yml')  # loading configuration for bea dataset (PCA, std)
shuffle_data = config['shuffle_data']  # whether to shuffle the training data
label_file = config['paths']['to_labels']  # path to the labels of the dataset
output_results = config['paths']['output_results']  # path to csv for saving the results

data = load_data(config=config)  # loading data
bea_train_flat = load_data(config=config_bea)  # load bea embeddings
df_labels = pd.read_csv(label_file)  # loading labels
data['label'] = df_labels.label.values  # adding labels to data
# Shuffling data if needed
if shuffle_data:
    data = data.sample(frac=1).reset_index(drop=True)

x_train, y_train = data.iloc[:, :-1].values, data.label.values  # train and labels

# Standardizing data before reducing dimensions
scaler_type = config_bea['data_scaling']['scaler_type']
if scaler_type:
    scaler = fit_scaler(config_bea, bea_train_flat)
    bea_train_flat = scaler.transform(bea_train_flat)
    x_train = scaler.transform(x_train)
    print("Train data standardized...")

dim_reduction = 'autoencoder'  # autoencoder
size_reduced = 'None'  # new dimension size after reduction
if dim_reduction == 'PCA':
    # APPLY PCA!
    # Train PCA model using embeddings got from bea-train-flat (57k files fo each emb type: convs and hiddens)
    # Transform the dataset using the fitted PCA model
    reduce_dims = ReduceDims(config_bea=config_bea)
    # if not scaler_type
    #     scaler = fit_scaler(config_bea, bea_train_flat)
    #     bea_train_flat = scaler.transform(bea_train_flat)

    pca = reduce_dims.fit_pca(bea_train_flat)  # train PCA
    x_train = pca.transform(x_train)  # transform (reduce dimensionality)
    print("New shape:", x_train.shape)
    size_reduced = x_train.shape[1]

elif dim_reduction == 'autoencoder':
    print("\nReducing dimensions using Autoencoder. Initial shape: {}".format(x_train.shape))
    device = ('cuda' if torch.cuda.is_available() else 'cpu')

    enc_shape = config_bea['dimension_reduction']['encoder_size']
    bea_train_flat = torch.from_numpy(bea_train_flat).float().to(device)
    x_train = torch.from_numpy(x_train).float().to(device)
    x_train = torch.from_numpy(x_train).float().to(device)
    y_train = torch.from_numpy(y_train).float().to(device)
    # defining the autoencoder and training
    encoder = Autoencoder(in_shape=x_train.shape[1], enc_shape=enc_shape).double().to(device)
    error = nn.MSELoss()
    optimizer = optim.Adam(encoder.parameters())
    train(encoder, error, optimizer, 100, x_train)

    # reducing the dimensions
    with torch.no_grad():
        encoded = encoder.encode(x_train)
        decoded = encoder.decode(encoded)
        mse = error(decoded, x_train).item()
        x_train = encoded.cpu().detach().numpy()
        dec = decoded.cpu().detach().numpy()

    print("New encoded shape:", x_train.shape)
else:
    pass

# Train SVM
print("Using", config['discrimination']['emb_type'])
list_c = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1]
df = pd.DataFrame(data, columns=['c', 'acc', 'f1', 'prec', 'recall', 'auc'])

emb_type = config['discrimination']['emb_type']
model_used = config['pretrained_model_details']['checkpoint_path'].split('/')[-2]

for c in list_c:
    array_preds, array_trues, array_probs = train_svm(svm_type='linear-loocv', C=c, X=x_train, y=np.ravel(y_train))

    acc = accuracy_score(array_trues, array_preds)
    # auc = roc_auc_score(array_trues, array_probs, labels=np.unique(y_train))
    auc = roc_auc_score(array_trues, array_probs[:, 1])
    # aucs = roc_auc_score_multiclass(actual_class=array_trues, pred_class=array_preds)

    f1 = f1_score(array_trues, array_preds)
    prec = precision_score(array_trues, array_preds)
    rec = recall_score(array_trues, array_preds)
    # data = {'c': c, 'acc': acc, 'f1': f1, 'prec': prec, 'recall': rec, 'auc': auc}
    data = {'c': c, 'acc': acc, 'f1': f1, 'prec': prec, 'recall': rec, 'auc': auc, 'Embedding': emb_type,
            'Reduction technique': '{0}-{1}'.format(dim_reduction, str(size_reduced)), 'Model used': model_used,
            'std': str(scaler_type)}
    df = df.append(data, ignore_index=True)

    print("with", c, "acc:", acc, " f1:", f1, " prec:", prec, " recall:", rec, 'AUC:', auc)#, 'auc-c0:', aucs[1],
          # 'auc-c1:', aucs[2])

# Saving results3
best_scores_df = df.iloc[[df['auc'].idxmax()]]  # getting the best scores based on the highest AUC score.
best_scores_df.to_csv(output_results, mode='a', header=not os.path.exists(output_results), index=False)

