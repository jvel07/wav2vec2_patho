import csv
import glob
import os
import sys
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.utils import shuffle
# from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
from tqdm import tqdm

from discrimination.svm_utils import train_svm


# write results to csv
def results_to_csv(file_name, list_columns, list_values):
    if not os.path.isfile(file_name):
        with open(file_name, mode='w') as csv_file:
            file_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            file_writer.writerow(list_columns)
            file_writer.writerow(list_values)
            print("File " + file_name + " created!")
    else:
        with open(file_name, 'a') as csv_file:
            file_writer = csv.writer(csv_file)
            file_writer.writerow(list_values)


# calculating auc score for each class (multiclass problems)
def roc_auc_score_multiclass(actual_class, pred_class, average="macro"):
    # creating a set of all the unique classes using the actual class list
    unique_class = set(actual_class)
    roc_auc_dict = {}
    for per_class in unique_class:
        # creating a list of all the classes except the current class
        other_class = [x for x in unique_class if x != per_class]

        # marking the current class as 1 and all other classes as 0
        new_actual_class = [0 if x in other_class else 1 for x in actual_class]
        new_pred_class = [0 if x in other_class else 1 for x in pred_class]

        # using the sklearn metrics method to calculate the roc_auc_score
        roc_auc = roc_auc_score(new_actual_class, new_pred_class, average=average)
        roc_auc_dict[per_class] = roc_auc

    return roc_auc_dict

# Pooling types
def global_max_pooling(data):
    max_pool = torch.nn.AdaptiveMaxPool1d(output_size=1)
    output = max_pool(torch.Tensor(data))
    output = output.squeeze(2)
    return output


def do_pooling(x_train, x_dev, x_test, pooling_type):
    try:
        print("Using {} as pooling type...".format(pooling_type))
        if pooling_type == 'mean':
            x_train = np.mean(x_train, 1)
            x_dev = np.mean(x_dev, 1)
            x_test = np.mean(x_test, 1)

        elif pooling_type == 'max':
            x_train = np.max(x_train, 1)
            x_dev = np.max(x_dev, 1)
            x_test = np.max(x_test, 1)

        elif pooling_type == 'sum':
            x_train = np.sum(x_train, 1)
            x_dev = np.sum(x_dev, 1)
            x_test = np.sum(x_test, 1)

        elif pooling_type == 'std':
            x_train = np.std(x_train, 1)
            x_dev = np.std(x_dev, 1)
            x_test = np.std(x_test, 1)

        elif pooling_type == 'global_max':
            x_train = global_max_pooling(x_train)
            x_dev = global_max_pooling(x_dev)
            x_test = global_max_pooling(x_test)

        print("Shapes after pooling:", x_train.shape, x_dev.shape, x_test.shape)
        return x_train, x_dev, x_test

    except:
        print("Pooling type not found:", pooling_type)


# Function to encode labels to numbers
def encode_labels(_y, list_labels):
    le = preprocessing.LabelEncoder()
    le.fit(list_labels)
    y = le.transform(_y)
    y = y.reshape(-1, 1)
    return np.squeeze(y), le


# Load data
def load_data_old(task, list_datasets, list_labels, emb_type):
    dict_data = {}
    for item in list_datasets:
        # Set data directories
        file_dataset = "../data/{0}/embeddings/{2}_{1}_wav2vec2.npy".format(task, emb_type, item)
        # print(file_dataset)
        # Load dataset
        dict_data['x_' + item] = np.load(file_dataset)
        # Load labels
        file_lbl_train = '../data/{0}/labels.csv'.format(task)
        df = pd.read_csv(file_lbl_train)
        df_labels = df[df['file_name'].str.match(item)]
        # dict_data['y_'+item] = df_labels.label.values
        dict_data['y_' + item], enc = encode_labels(df_labels.label.values, list_labels)
    print("Data loaded!")
    return dict_data['x_train'], dict_data['x_dev'], dict_data['x_test'], dict_data['y_train'], dict_data['y_dev'], \
           dict_data['y_test']


def load_data(config):
    model_used = config['pretrained_model_details']['checkpoint_path'].split('/')[-2]
    path_embs = os.path.join(config['paths']['out_embeddings'], model_used + '/') #config['discrimination']['emb_type']+'/')
    emb_type = config['discrimination']['emb_type']  # type of embeddings to load

    list_file_embs = glob.glob('{0}{1}*.npy'.format(path_embs, emb_type))
    list_file_embs.sort()
    print(path_embs)
    print("{} files found".format(len(list_file_embs)))
    list_arr_embs = []
    for file in tqdm(list_file_embs, total=len(list_file_embs)):
        utterance_name = os.path.basename(file).split('.')[0]
        list_arr_embs.append(np.load(file))
    # To dataframe
    data = pd.DataFrame(list_arr_embs)
    print("Data loaded!")

    return data


class LoadMulti:
    def __init__(self):

        # load a file and return the contents
        def load_file(filepath):
            # open the file
            with open(filepath, 'r') as handle:
                # return the contents
                handle.read()

        # return the contents of many files
        def load_files(filepaths):
            # create a thread pool
            with ThreadPoolExecutor(len(filepaths)) as exe:
                # load files
                futures = [exe.submit(load_file, name) for name in filepaths]
                # collect data
                data_list = [future.result() for future in futures]
                # return data and file paths
                return (data_list, filepaths)

        # load all files in a directory into memory
        def main(path='tmp'):
            # prepare all the paths
            paths = [os.path.join(path, filepath) for filepath in os.listdir(path)]
            # determine chunksize
            n_workers = 8
            chunksize = round(len(paths) / n_workers)
            # create the process pool
            with ProcessPoolExecutor(n_workers) as executor:
                futures = list()
                # split the load operations into chunks
                for i in range(0, len(paths), chunksize):
                    # select a chunk of filenames
                    filepaths = paths[i:(i + chunksize)]
                    # submit the task
                    future = executor.submit(load_files, filepaths)
                    futures.append(future)
                # process all results
                for future in as_completed(futures):
                    # open the file and load the data
                    _, filepaths = future.result()
                    for filepath in filepaths:
                        # report progress
                        print(f'.loaded {filepath}')
            print('Done')




# Train and test function
def train_test_pooling(task, list_labels, list_datasets, emb_type, std=True, resample=False, pca=False, svd=False,
               save_res=True, svm_type='linear', pooling_type='mean'):
    # Load data
    x_train, x_dev, x_test, y_train, y_dev, y_test = load_data(task=task, list_datasets=list_datasets,
                                                               emb_type=emb_type, list_labels=list_labels)

    # # Binarizing labels
    # if binarize_lbl:
    #     y_train[y_train == 2] = 0
    #     y_dev[y_dev == 2] = 0
    #     y_test[y_test == 2] = 0
    #     print("Data binarized...")

    # Pooling data (from 3d to 2d)
    x_train, x_dev, x_test = do_pooling(x_train, x_dev, x_test, pooling_type=pooling_type)

    # combine train+dev
    x_combined = np.concatenate((x_train, x_dev))
    y_combined = np.concatenate((y_train, y_dev))
    print("train+dev combined:", x_combined.shape)

    # Standardization
    if std and not pooling_type == 'flatten':
        print("Standardizing...")
        std_scaler = preprocessing.RobustScaler()
        x_train = std_scaler.fit_transform(x_train)
        x_dev = std_scaler.transform(x_dev)
        x_combined = std_scaler.fit_transform(x_combined)
        x_test = std_scaler.transform(x_test)

    # PCA
    pca_comp = 'None'
    if pca and not svd:
        pca_comp = 0.95
        print("Reducing dimesions with PCA. Keeping {0} variance...".format(pca_comp))
        #         pca_er = KernelPCA(n_components=512, n_jobs=16)
        pca_er = PCA(n_components=pca_comp)
        x_train = pca_er.fit_transform(x_train)
        x_dev = pca_er.transform(x_dev)
        # Fit PCA on train+dev
        x_combined = pca_er.fit_transform(x_combined)
        x_test = pca_er.transform(x_test)
        print("Shape 'x_train' after PCA:", x_train.shape)
        print("Shape 'x_combined' after PCA:", x_combined.shape)

    # SVD
    svd_com = 'None'
    if svd and not pca:
        if emb_type == 'hiddens':
            svd_com = 442
        else:
            svd_com = 342
        print("Reducing dimensions with SVD...")
        svd_er = TruncatedSVD(n_components=svd_com, n_iter=5, random_state=42)
        x_train = svd_er.fit_transform(x_train)
        x_dev = svd_er.transform(x_dev)
        # On train+dev
        x_combined = svd_er.fit_transform(x_combined)
        x_test = svd_er.transform(x_test)
        print("Shape 'x_train' after SVD:", x_train.shape)
        print("Shape 'x_combined' after SVD:", x_combined.shape)

    # undersampling x_train
    if resample:
        print("Undersampling...")
        rus = RandomUnderSampler(random_state=42)
        x_train, y_train = rus.fit_resample(x_train, y_train)
        x_combined, y_combined = rus.fit_resample(x_combined, y_combined)
        # Shuffling samples
        x_train, y_train = shuffle(x_train, y_train, random_state=42)
        print("Shape 'x_train' after resampling:", x_train.shape)
        print("Shape 'x_combined' after resampling:", x_combined.shape)

    # train on train set and evaluate on dev
    uar_scores = []
    list_c = [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1]
    print("Using {} SVM... \nEvaluation scores: ".format(svm_type))
    for c in list_c:
        dev_post = train_svm(svm_type=svm_type, X=x_train, y=y_train, X_eval=x_dev, C=c)
        pred = np.argmax(dev_post, axis=1)
        uar = recall_score(y_dev, pred, average='macro')
        uar_scores.append(uar)
        print(c, "-->", uar)

    # Grabbing best C value
    best_c = list_c[np.argmax(uar_scores)]
    print("Best C value was:", best_c)

    # validate with best C on test using train+dev
    post_test = train_svm(svm_type=svm_type, X=x_combined, y=y_combined, X_eval=x_test, C=best_c)
    np.savetxt('data/{3}/posteriors/{2}_embeddings_post_test_{0}_{1}.txt'.format(pooling_type, best_c, emb_type, task),
               post_test)
    print("Test posteriors saved...")
    pred_test = np.argmax(post_test, axis=1)
    uar_test = recall_score(y_test, pred_test, average='macro')
    print("Test score with best C={} -->".format(best_c), uar_test)
    if save_res:
        results_to_csv(file_name='data/{0}/results_wav2vec2.csv'.format(task),
                       list_columns=['Embedding', 'best C', 'PCA', 'UND_SAMPLE', 'POOLING', 'std', 'UAR'],
                       list_values=[emb_type, best_c, pca_comp, resample, pooling_type, std, uar_test])


def train_test(task, list_labels, list_datasets, emb_type, std=True, resample=False, pca=False, svd=False,
               save_res=True, svm_type='linear'):
    pooling_type = 'mean'
    # Load data
    x_train, x_dev, x_test, y_train, y_dev, y_test = load_data(task=task, list_datasets=list_datasets,
                                                               emb_type=emb_type, list_labels=list_labels)

    # # Binarizing labels
    # if binarize_lbl:
    #     y_train[y_train == 2] = 0
    #     y_dev[y_dev == 2] = 0
    #     y_test[y_test == 2] = 0
    #     print("Data binarized...")

    # combine train+dev
    x_combined = np.concatenate((x_train, x_dev))
    y_combined = np.concatenate((y_train, y_dev))
    print("train+dev combined:", x_combined.shape)

    # Standardization
    if std:
        print("Standardizing...")
        std_scaler = preprocessing.RobustScaler()
        x_train = std_scaler.fit_transform(x_train)
        x_dev = std_scaler.transform(x_dev)
        x_combined = std_scaler.fit_transform(x_combined)
        x_test = std_scaler.transform(x_test)

    # PCA
    pca_comp = 'None'
    if pca and not svd:
        pca_comp = 0.95
        print("Reducing dimesions with PCA. Keeping {0} variance...".format(pca_comp))
        #         pca_er = KernelPCA(n_components=512, n_jobs=16)
        pca_er = PCA(n_components=pca_comp)
        x_train = pca_er.fit_transform(x_train)
        x_dev = pca_er.transform(x_dev)
        # Fit PCA on train+dev
        x_combined = pca_er.fit_transform(x_combined)
        x_test = pca_er.transform(x_test)
        print("Shape 'x_train' after PCA:", x_train.shape)
        print("Shape 'x_combined' after PCA:", x_combined.shape)

    # SVD
    svd_com = 'None'
    if svd and not pca:
        print("Reducing dimensions with SVD...")
        svd_er = TruncatedSVD(n_components=svd_com, n_iter=5, random_state=42)
        x_train = svd_er.fit_transform(x_train)
        x_dev = svd_er.transform(x_dev)
        # On train+dev
        x_combined = svd_er.fit_transform(x_combined)
        x_test = svd_er.transform(x_test)
        print("Shape 'x_train' after SVD:", x_train.shape)
        print("Shape 'x_combined' after SVD:", x_combined.shape)

    # undersampling x_train
    if resample:
        print("Undersampling...")
        rus = RandomUnderSampler(random_state=42)
        x_train, y_train = rus.fit_resample(x_train, y_train)
        x_combined, y_combined = rus.fit_resample(x_combined, y_combined)
        # Shuffling samples
        x_train, y_train = shuffle(x_train, y_train, random_state=42)
        print("Shape 'x_train' after resampling:", x_train.shape)
        print("Shape 'x_combined' after resampling:", x_combined.shape)

    # train on train set and evaluate on dev
    uar_scores = []
    list_c = [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1]
    print("Using {} SVM... \nEvaluation scores: ".format(svm_type))
    for c in list_c:
        dev_post = train_svm(svm_type=svm_type, X=x_train, y=y_train, X_eval=x_dev, C=c)
        pred = np.argmax(dev_post, axis=1)
        uar = recall_score(y_dev, pred, average='macro')
        uar_scores.append(uar)
        print(c, "-->", uar)

    # Grabbing best C value
    best_c = list_c[np.argmax(uar_scores)]
    print("Best C value was:", best_c)

    # validate with best C on test using train+dev
    post_test = train_svm(svm_type=svm_type, X=x_combined, y=y_combined, X_eval=x_test, C=best_c)
    np.savetxt('data/{3}/posteriors/{2}_embeddings_post_test_{0}_{1}.txt'.format(pooling_type, best_c, emb_type, task),
               post_test)
    print("Test posteriors saved...")
    pred_test = np.argmax(post_test, axis=1)
    uar_test = recall_score(y_test, pred_test, average='macro')
    print("Test score with best C={} -->".format(best_c), uar_test)
    if save_res:
        results_to_csv(file_name='data/{0}/results_wav2vec2.csv'.format(task),
                       list_columns=['Embedding', 'best C', 'PCA', 'UND_SAMPLE', 'POOLING', 'std', 'UAR'],
                       list_values=[emb_type, best_c, pca_comp, resample, pooling_type, std, uar_test])

