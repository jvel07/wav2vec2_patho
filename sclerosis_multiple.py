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

config = utils.load_config('config/config_sm.yml')  # loading configuration
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
    print("\nReducing dimensions using a basic Autoencoder. Initial shape: {}".format(x_train.shape))
    device = ('cuda' if torch.cuda.is_available() else 'cpu')

    n_epochs = config['dimension_reduction']['autoencoder']['num_epochs']
    enc_shape = config['dimension_reduction']['autoencoder']['encoder_size']
    # bea_train_flat = torch.from_numpy(bea_train_flat).double().to(device)
    x_train = torch.from_numpy(x_train).double().to(device)
    # defining the autoencoder and training
    encoder = Autoencoder(in_shape=x_train.shape[1], enc_shape=enc_shape).double().to(device)
    error = nn.MSELoss()
    optimizer = optim.Adam(encoder.parameters())
    train(encoder, error, optimizer, n_epochs, x_train)

    # reducing the dimensions
    with torch.no_grad():
        encoded = encoder.encode(x_train)
        decoded = encoder.decode(encoded)
        mse = error(decoded, x_train).item()
        x_train = encoded.cpu().detach().numpy()
        dec = decoded.cpu().detach().numpy()

    size_reduced = x_train.shape[1]
    print("New encoded shape:", x_train.shape)

elif dim_reduction == 'vae':
    print("\nReducing dimensions using {0}. Initial shape: {1}".format(dim_reduction, x_train.shape))
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    n_epochs = config['dimension_reduction']['autoencoder']['num_epochs']
    log_interval = 50

    # converting data into dataloader (needed for training)
    data_set = DataBuilder(bea_train_flat)
    train_loader = DataLoader(dataset=data_set, batch_size=32)
    train_data = DataBuilder(x_train)
    x_loader = DataLoader(dataset=train_data, batch_size=32)

    # define params
    D_in = data_set.x.shape[1]
    H = 50
    H2 = 12
    latent_dim = config['dimension_reduction']['autoencoder']['encoder_size']  # output size of the reduced embs
    model = VariationalAutoencoder(D_in, latent_dim, H, H2).to(device)
    model.apply(weights_init_uniform_rule)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_mse = CustomLoss()

    # train
    print("Training the Variational Autoencoder...")
    for epoch in range(1, n_epochs+1):
        model_trained = train_vae(model, train_loader, epoch, device, optimizer, loss_mse, config_bea)

    #  Reducing dimensions of x_train
    mu_output = []
    with torch.no_grad():
        for i, data in enumerate(x_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)

            mu_tensor = mu
            mu_output.append(mu_tensor)
            mu_result = torch.cat(mu_output, dim=0)

    size_reduced = mu_result.shape[1]
    print("New encoded shape:", mu_result.shape)
    x_train = mu_result.detach().cpu().numpy()

else:
    pass

# Train SVM
print("Using", config['discrimination']['emb_type'])
list_c = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1]
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