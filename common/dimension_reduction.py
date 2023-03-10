import os

import numpy as np
import sklearn
import torch
from sklearn.decomposition import PCA
import pickle as pk

from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

# from discrimination.discrimination_utils import load_data


#     pp = utils.PreprocessFunction(processor, label_list, target_sampling_rate)
class ReduceDims:
    def __init__(self, config_bea):
        self.config_bea = config_bea

    def fit_pca(self, bea_train_flat):

        # save_pca = kwargs.get('save_pca', None)
        # out_dir = kwargs.get('out_dir', None)
        # emb_type = kwargs.get('emb_type', None)
        # config_bea = kwargs.get('config_bea', None)
        n_components = self.config_bea['dimension_reduction']['pca']['n_components']
        save_pca = self.config_bea['dimension_reduction']['pca']['save_pca']
        out_dir = self.config_bea['dimension_reduction']['pca']['pca_path']
        emb_type = self.config_bea['discrimination']['emb_type']  # type of embeddings to load

        final_out_path = '{0}_{1}_{2}.pkl'.format(out_dir, str(n_components), emb_type)

        os.makedirs(os.path.dirname(final_out_path), exist_ok=True)
        # if os.path.isfile(final_out_path):
        #     print("Seems like the PCA model was already trained:\n{}. Loading this model...".format(final_out_path))
        #     pca = pk.load(open(final_out_path, 'rb'))
        #     return pca
        # else:
        print("No trained PCA found; starting to train PCA...")
        # bea_train_flat = load_data(config=self.config_bea)  # load bea embeddings
        # train PCA
        pca = PCA(n_components=46)
        pca.fit(bea_train_flat)
        print("PCA fitted...")

        if save_pca:
            pk.dump(pca, open(final_out_path, 'wb'))
            print("PCA model saved to:", final_out_path)
        return pca

    def transform_pca(self, pca_fitted):
        transformed_data = pca_fitted.transform(self.transform_data)
        print("Data transformed, final shape:", transformed_data.shape)

    def get_var_ratio_pca(X):
        pca = PCA(n_components=None)
        pca.fit(X)
        return pca.explained_variance_ratio_

    def sel_pca_comp(self, var_ratio, goal_var: float) -> int:
        # Set initial variance explained so far
        total_variance = 0.0

        # Set initial number of features
        n_components = 0

        # For the explained variance of each feature:
        for explained_variance in var_ratio:

            # Add the explained variance to the total
            total_variance += explained_variance

            # Add one to the number of components
            n_components += 1

            # If we reach our goal level of explained variance
            if total_variance >= goal_var:
                # End the loop
                break

        # Return the number of components
        return n_components


# Autoencoder
class Autoencoder(nn.Module):
    """Makes the main denoising auto

    Parameters
    ----------
    in_shape [int] : input shape
    enc_shape [int] : desired encoded shape
    """

    def __init__(self, in_shape, enc_shape):
        super(Autoencoder, self).__init__()

        self.encode = nn.Sequential(
            nn.Linear(in_shape, 128),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(64, enc_shape),
        )

        self.decode = nn.Sequential(
            nn.BatchNorm1d(enc_shape),
            nn.Linear(enc_shape, 64),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(128, in_shape)
        )

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x


def train(model, error, optimizer, n_epochs, x):
    model.train()
    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()
        output = model(x)
        loss = error(output, x)
        loss.backward()
        optimizer.step()

        if epoch % int(0.1 * n_epochs) == 0:
            print(f'epoch {epoch} \t Loss: {loss.item():.4g}')


######  Variational Auto Encoder  ######
class VariationalAutoencoder(nn.Module):
    def __init__(self, D_in, latent_dim, H=50, H2=12):

        # Encoder
        super(VariationalAutoencoder, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.lin_bn1 = nn.BatchNorm1d(num_features=H)
        self.linear2 = nn.Linear(H, H2)
        self.lin_bn2 = nn.BatchNorm1d(num_features=H2)
        self.linear3 = nn.Linear(H2, H2)
        self.lin_bn3 = nn.BatchNorm1d(num_features=H2)

        #         # Latent vectors mu and sigma
        self.fc1 = nn.Linear(H2, latent_dim)
        self.bn1 = nn.BatchNorm1d(num_features=latent_dim)
        self.fc21 = nn.Linear(latent_dim, latent_dim)
        self.fc22 = nn.Linear(latent_dim, latent_dim)

        #         # Sampling vector
        self.fc3 = nn.Linear(latent_dim, latent_dim)
        self.fc_bn3 = nn.BatchNorm1d(latent_dim)
        self.fc4 = nn.Linear(latent_dim, H2)
        self.fc_bn4 = nn.BatchNorm1d(H2)

        #         # Decoder
        self.linear4 = nn.Linear(H2, H2)
        self.lin_bn4 = nn.BatchNorm1d(num_features=H2)
        self.linear5 = nn.Linear(H2, H)
        self.lin_bn5 = nn.BatchNorm1d(num_features=H)
        self.linear6 = nn.Linear(H, D_in)
        self.lin_bn6 = nn.BatchNorm1d(num_features=D_in)

        self.relu = nn.ReLU()

    def encode(self, x):
        lin1 = self.relu(self.lin_bn1(self.linear1(x)))
        lin2 = self.relu(self.lin_bn2(self.linear2(lin1)))
        lin3 = self.relu(self.lin_bn3(self.linear3(lin2)))

        fc1 = F.relu(self.bn1(self.fc1(lin3)))

        r1 = self.fc21(fc1)
        r2 = self.fc22(fc1)

        return r1, r2

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        fc3 = self.relu(self.fc_bn3(self.fc3(z)))
        fc4 = self.relu(self.fc_bn4(self.fc4(fc3)))

        lin4 = self.relu(self.lin_bn4(self.linear4(fc4)))
        lin5 = self.relu(self.lin_bn5(self.linear5(lin4)))
        return self.lin_bn6(self.linear6(lin5))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        # self.decode(z) ist später recon_batch, mu ist mu und logvar ist logvar
        return self.decode(z), mu, logvar


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")

    # x_recon ist der im forward im Model erstellte recon_batch, x ist der originale x Batch, mu ist mu und logvar ist logvar
    def forward(self, x_recon, x, mu, logvar):
        loss_MSE = self.mse_loss(x_recon, x)
        loss_KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return loss_MSE + loss_KLD


# takes in a module and applies the specified weight initialization
def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0/np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)


def train_vae(model, train_loader, epoch, device, optimizer, loss_mse, config_bea):
    train_losses = []
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_mse(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
#        if batch_idx % log_interval == 0:
#            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                epoch, batch_idx * len(data), len(trainloader.dataset),
#                       100. * batch_idx / len(trainloader),
#                       loss.item() / len(data)))
    if epoch % 200 == 0:
        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(train_loader.dataset)))
        train_losses.append(train_loss / len(train_loader.dataset))
        out_path = "{}_{}_{}_vae".format(config_bea["dimension_reduction"]["autoencoder"]["save_path"],
                                         len(train_loader.dataset), config_bea["data_scaling"]["scaler_type"])
        torch.save(model, out_path)

    return model


######  Variational Auto Encoder   ######



