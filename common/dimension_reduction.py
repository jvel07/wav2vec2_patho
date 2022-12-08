import os

import sklearn
from sklearn.decomposition import PCA
import pickle as pk

from discrimination.discrimination_utils import load_data


#     pp = utils.PreprocessFunction(processor, label_list, target_sampling_rate)
class ReduceDims:
    def __init__(self, config_bea):
        self.config_bea = config_bea

    def fit_pca(self):

        # save_pca = kwargs.get('save_pca', None)
        # out_dir = kwargs.get('out_dir', None)
        # emb_type = kwargs.get('emb_type', None)
        # config_bea = kwargs.get('config_bea', None)
        n_components = self.config_bea['dimension_reduction']['n_components']
        save_pca = self.config_bea['dimension_reduction']['save_pca']
        out_dir = self.config_bea['dimension_reduction']['pca_path']
        emb_type = self.config_bea['discrimination']['emb_type']  # type of embeddings to load

        final_out_path = '{0}_{1}_{2}.pkl'.format(out_dir, str(n_components), emb_type)

        os.makedirs(os.path.dirname(final_out_path), exist_ok=True)
        if os.path.isfile(final_out_path):
            while True:
                reply = input("Seems like the PCA model was already trained:\n{}. \nIf you changed the size of the sets,"
                              "or the value of the variance, then you may want to train the model again.\n"
                              " Do you want to retrain the PCA model? Yes or [No]: ".format(final_out_path) or "no")
                if reply.lower() not in ('yes', 'no'):
                    print("Please, enter either 'yes' or 'no'")
                    continue
                else:
                    if reply.lower() == 'yes':
                        bea_train_flat = load_data(config=self.config_bea)  # load bea embeddings
                        pca = PCA(n_components=n_components)
                        pca.fit(bea_train_flat)
                        print("PCA fitted...")

                        if save_pca:
                            pk.dump(pca, open(final_out_path, 'wb'))
                            print("PCA model saved to:", final_out_path)
                        return pca
                    else:
                        print("You chose {}. Loading the existing PCA model...".format(reply))
                        pca = pk.load(open(final_out_path, 'rb'))
                        return pca
                        # pass
                    # break
        else:
            bea_train_flat = load_data(config=self.config_bea)  # load bea embeddings
            # train PCA
            pca = PCA(n_components=n_components)
            pca.fit(bea_train_flat)
            print("PCA fitted...")

            if save_pca:
                pk.dump(pca, open(final_out_path, 'wb'))
                print("PCA model saved to:", final_out_path)
            return pca

    def transform_pca(self, pca_fitted):
        transformed_data = pca_fitted.transform(self.transform_data)
        print("Data transformed, final shape:", transformed_data.shape)
