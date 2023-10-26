import collections
import csv
import itertools

import evaluate
import librosa
import torch
import yaml
from datasets import Dataset, load_metric
from sklearn.model_selection import train_test_split, GroupShuffleSplit
import glob
import os


import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer, LabelEncoder
from transformers import EvalPrediction
from yaml import SafeLoader
from tqdm import tqdm
import pickle as pk


# from discrimination.discrimination_utils import load_data, check_model_used


def check_model_used(checkpoint_path):
    if "jonatasgrosman" in checkpoint_path:
        model_used = checkpoint_path.split('/')[-1]
    elif "facebook" in checkpoint_path:
        model_used = checkpoint_path.split('/')[-1]
    elif "yangwang825" in checkpoint_path:
        model_used = checkpoint_path.split('/')[-1]
    else:
        model_used = checkpoint_path.split('/')[-2]

    return model_used


def load_config(path_yaml):
    with open(path_yaml) as f:
        data = yaml.load(f, Loader=SafeLoader)
    return data


def save_data_iteratively(file_path, data):
    with open(file_path, 'w') as f:
        np.savetxt(f, data.squeeze(), newline=" ", fmt="%s", )
        f.write('\n')
        f.close()


def get_audio_paths(path, is_sorted=True):
    """Function to read wav files from a given path.

    :param path: str, Path to the folder containing the wav files.
    :param is_sorted: bool, Whether to sort or not the List.
    :return: List, returns the list of wavs read.
    """
    if is_sorted:
        return sorted(glob.glob('{}*.wav'.format(path)))
    else:
        return glob.glob('{}*.wav'.format(path))


def create_csv_bea(audio_list, out_path, split_data):
    """Function to create csv file for the BEA Corpus with labels of the form:
    'file_name', 'label'

    :param split_data: [Optional] int, corresponds to the percentage of the size of the test.txt set.
    :param out_path: string, path to the desired output folder.
    :param audio_list: List, a list containing the wav files paths.
    :return: List, final csv list.
    """

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    final_list = []
    for idx, i in enumerate(audio_list):
        file_name = os.path.basename(i)
        label = file_name[0:6]
        final_list.append(file_name + ',' + label + ',' + i)  # makes labels based on the speaker name
        # final_list.append(file_name + ',' + str(idx).zfill(3))  # makes labels based on the idx of the file
        final_list.sort()

    df = pd.DataFrame([sub.split(",") for sub in final_list], columns=['file_name', 'label', 'path'])

    if os.path.isfile('{}train.csv'.format(out_path)) and os.path.isfile('{}test.csv'.format(out_path)):
        print("Seems like csv data is already created in {}. Continuing to the next step...".format(out_path))
    else:
        train_df, test_df = train_test_split(df, test_size=split_data, random_state=42)
        train_df = train_df.reset_index(drop=True)  # reset idx count
        train_df.to_csv('{}train.csv'.format(out_path), sep=',', index=False)
        test_df = test_df.reset_index(drop=True)
        test_df.to_csv('{}test.csv'.format(out_path), sep=',', index=False)
        print("Train and test.txt data saved to {}".format(out_path))


def crate_csv_bea_from_scp(scp_file, out_path, train_split_data):
    """Function to create csv file for the BEA Corpus based on a 'scp' Kaldi file. Labels of the form:
    'file_name', 'label'

    :param train_split_data: [Optional] int, corresponds to the percentage of the size of the train set.
    :param out_path: string, path to the desired output folder.
    :param scp_file: A Kaldi scp file of the form "[wav_name] [path_to_wav]".
    :return: List, final csv list.
    """

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    train_csv_path = '{}train.csv'.format(out_path)
    test_csv_path = '{}test.csv'.format(out_path)

    # read scp file
    df = pd.read_csv(scp_file, names=['file_name', 'path'], delimiter=' ')
    df['label'] = df.file_name.str[0:6]  # adding label column based on speakers.
    # df['label'] = df.index # adding label column based on utterances.

    if os.path.isfile(train_csv_path) and os.path.isfile(test_csv_path):
        while True:
            reply = input("Seems like csv data is already created in {}. If you changed the size of the sets, "
                          "you may want to generate the data again.\n"
                          " Do you want to overwrite them? Yes or [No]: ".format(out_path) or "no")
            if reply.lower() not in ('yes', 'no'):
                print("Please, enter either 'yes' or 'no'")
                continue
            else:
                if reply.lower() == 'yes':
                    # Create the files again
                    print("Generating new files...")
                    train_df, test_df = train_test_split(df, train_size=train_split_data, random_state=42)
                    train_df = train_df.reset_index(drop=True)  # reset idx count
                    train_df.to_csv(train_csv_path, sep=',', index=False)
                    test_df = test_df.reset_index(drop=True)
                    test_df.to_csv(test_csv_path, sep=',', index=False)
                    print("Train and test data saved to {}".format(out_path))
                else:
                    print("You chose {}. Continuing to the next step...".format(reply))
                    pass
                break
    else:
        # Create the files
        print("Generating new data files...")
        train_df, test_df = train_test_split(df, train_size=train_split_data, random_state=42)
        train_df = train_df.reset_index(drop=True)  # reset idx count
        train_df.to_csv(train_csv_path, sep=',', index=False)
        test_df = test_df.reset_index(drop=True)
        test_df.to_csv(test_csv_path, sep=',', index=False)
        print("Train and test data saved to {}".format(out_path))


def create_csv_bea_base(corpora_path, out_file):
    """Function to create csv file for the BEA Corpus with labels of the form:
    'file_name', 'text'

    :param out_file: string, name of the output file preceded with a desired parent directory. E.g.: a/path/train_set
    :param corpora_path: String, path to the wavs and their corresponding transcriptions.
    :return: List, final csv list.
    """

    if not os.path.exists(os.path.dirname(out_file)):
        os.makedirs(os.path.dirname(out_file))

    if os.path.isfile(out_file):
        print("{} already exists! Skipping generation...".format(out_file))
    else:
        # reading directories
        transcriptions = glob.glob('{}*.txt'.format(corpora_path))
        transcriptions.sort()
        wavs = glob.glob('{}*.wav'.format(corpora_path))
        wavs.sort()

        final_list = []
        for wav_path, text_file in tqdm(zip(wavs, transcriptions), total=len(wavs)):
            with open(text_file, 'r') as f:
                sentence = f.read()
                f.close()
            final_list.append(wav_path + ',' + sentence)

        df = pd.DataFrame([sub.split(",") for sub in final_list], columns=['file', 'sentence'])
        # to_drop = df.shape[0] - int(size_bea)
        # df.drop(df.tail(int(to_drop)).index, inplace=True)  # drop last n rows

        df.to_csv(out_file, sep=',', index=False)
        print("Data saved to {}. Size: {}".format(out_file, df.shape[0]))


def create_csv_sm(in_path, out_file):
    """Function to create csv file for the Sclerosis Multiple Corpus with labels of the form:
    'file_name', 'label'

    :param out_file: string, name of the output file preceded with a desired parent directory. E.g.: a/path/train_set
    :param in_path: string, path to the dataset containing the utterances.
    :return: List, final csv list.
    """

    if not os.path.exists(os.path.dirname(out_file)):
        os.makedirs(os.path.dirname(out_file))

    # reading directories
    audio_list = glob.glob('{}/*.wav'.format(in_path))
    audio_list.sort()

    final_list = []
    for wav_path in tqdm(audio_list, total=len(audio_list)):
        file_name = os.path.basename(wav_path)
        label = file_name[0]
        if label == 'C':
            label = 0
        else:
            label = 1
        final_list.append(wav_path + ',' + str(label))

    df = pd.DataFrame([sub.split(",") for sub in final_list], columns=['file', 'label'])

    df.to_csv(out_file, sep=',', index=False)
    print("Data saved to {}".format(out_file))


def create_csv_compare_23(in_path, out_file):
    """Function to create csv file for the Sclerosis Multiple Corpus with labels of the form:
    'file_name', 'label'

    :param out_file: string, name of the output file preceded with a desired parent directory. E.g.: a/path/train_set
    :param in_path: string, path to the dataset containing the utterances.
    :return: List, final csv list.
    """

    if not os.path.exists(os.path.dirname(out_file)):
        os.makedirs(os.path.dirname(out_file))

    # reading directories
    for folder_name in ['train', 'dev', 'test']:
        audio_list = glob.glob('{0}/{1}/*.wav'.format(in_path, folder_name))
        audio_list.sort()

        final_list = []
        for wav_path in tqdm(audio_list, total=len(audio_list)):
            file_name = os.path.basename(wav_path)
            final_list.append(wav_path + ',' + str(file_name))

        df = pd.DataFrame([sub.split(",") for sub in final_list], columns=['file', 'name'])

        final_name = os.path.join(out_file, folder_name + '.csv')
        df.to_csv(final_name, sep=',', index=False)
        print("Data saved to {}".format(final_name))

def create_csv_depression(in_path, out_file):
    """Function to create csv file for the Sclerosis Multiple Corpus with labels of the form:
    'file', 'label', 'etc...'

    :param out_file: string, name of the output file preceded with a desired parent directory. E.g.: a/path/train_set
    :param in_path: string, path to the dataset containing the utterances.
    :return: List, final csv list.
    """

    if not os.path.exists(os.path.dirname(out_file)):
        os.makedirs(os.path.dirname(out_file))

    # reading original labels
    metadata = pd.read_csv(in_path + '/metadata.csv', sep='\t')
    # reading directories
    audio_list = glob.glob('{}/*/*.wav'.format(in_path))
    print("Found {} utterances in {}".format(len(audio_list), in_path))
    audio_list.sort()

    metadata['file'] = audio_list
    metadata.to_csv(out_file, sep=',', index=False)
    print("Data saved to {}".format(out_file))


def create_csv_eating(in_path, out_file_train, out_file_dev, out_file_test):
    """Function to create csv file for the Eating Corpus with labels of the form:
    'file', 'label', 'etc...'

    :param out_file: string, name of the output file preceded with a desired parent directory. E.g.: a/path/train_set
    :param in_path: string, path to the folder containing the utterances.
    :return: List, final csv list.
    """

    if not os.path.exists(os.path.dirname(out_file_train)):
        os.makedirs(os.path.dirname(out_file_train))

    # Function to generate the path for each filename
    def add_path(filename):
        return os.path.join(in_path, filename+'.wav')

    def add_duration(file_path):
        return sf.info(os.path.join(file_path)).frames

    # reading original labels
    metadata_train = pd.read_csv('../data/eating/eating_train.csv', sep=',')
    metadata_dev = pd.read_csv('../data/eating/eating_dev.csv', sep=',')
    metadata_test = pd.read_csv('../data/eating/eating_test.csv', sep=',')

    # Apply the function to create the new column
    metadata_train['path'] = metadata_train['filename'].apply(add_path)
    metadata_dev['path'] = metadata_dev['filename'].apply(add_path)
    metadata_test['path'] = metadata_test['filename'].apply(add_path)

    # Apply the function to create the new Duration column
    metadata_train['length'] = metadata_train['path'].apply(add_duration)
    metadata_dev['length'] = metadata_dev['path'].apply(add_duration)
    metadata_test['length'] = metadata_test['path'].apply(add_duration)

    metadata_train.to_csv(out_file_train, sep=',', index=False)
    metadata_dev.to_csv(out_file_dev, sep=',', index=False)
    metadata_test.to_csv(out_file_test, sep=',', index=False)
    print("Data saved to {0} and {1}".format(out_file_train, out_file_dev))


"""FUNCTIONS FOR AUDIO PREPROCESSING"""


def label_to_id(label, label_list):
    if len(label_list) > 0:
        return label_list.index(label) if label in label_list else -1

    return label


def speech_to_array(path):
    speech, _ = sf.read(path)
    # batch["speech"] = speech
    return speech


class PreprocessFunction:
    def __init__(self, processor, label_list, target_sampling_rate):
        self.processor = processor
        self.label_list = label_list
        self.target_sampling_rate = target_sampling_rate

    def preprocess_function(self, samples):
        speech_list = [speech_to_array(path) for path in samples["path"]]
        target_list = [label_to_id(label, self.label_list) for label in samples["label"]]

        result = self.processor(speech_list, sampling_rate=self.target_sampling_rate)
        result["labels"] = list(target_list)

        return result


class PreprocessFunctionASR:
    def __init__(self, processor, target_sampling_rate):
        self.processor = processor
        self.target_sampling_rate = target_sampling_rate

    def preprocess_function_asr(self, samples):
        speech_list = [speech_to_array(path) for path in samples["file"]]
        result = self.processor(speech_list, sampling_rate=self.target_sampling_rate)
        with self.processor.as_target_processor():
            result["labels"] = self.processor(samples["sentence"]).input_ids

        return result

    def preprocess_function_tempo(self, samples):
        speech_list = [speech_to_array(path) for path in samples["path"]]
        # target_list = samples["whole_speech"]

        input_values = self.processor(speech_list, sampling_rate=self.target_sampling_rate)
        # result["labels"] = list(target_list)

        return input_values

    def prepare_dataset(self, batch):
        speech_list = [speech_to_array(path) for path in batch["file"]]

        # batched output is "un-batched"
        batch["input_values"] = self.processor(speech_list, sampling_rate=self.target_sampling_rate).input_values[0]

        with self.processor.as_target_processor():
            batch["labels"] = self.processor(batch["sentence"]).input_ids
        return batch


def compute_metrics(p: EvalPrediction, is_regression=False):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
    # predictions = np.argmax(preds, axis=1)
    if is_regression:
        return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
    else:
        return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}
        # return {"accuracy":  accuracy_metric.compute(predictions=preds, references=p.label_ids)}
        # return {"uar": recall_score(p.label_ids, preds, labels=[1, 0], average='macro')}


def compute_metrics_compare(eval_pred):
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")
    """Computes spearman and pearson on predictions"""
    recall = recall_metric.compute(
        predictions=np.argmax(eval_pred.predictions.squeeze(), axis=-1),
        references=eval_pred.label_ids,
        average="macro",
    )
    f1 = f1_metric.compute(
        predictions=np.argmax(eval_pred.predictions.squeeze(), axis=-1),
        references=eval_pred.label_ids,
        average="macro",
    )
    # f1 = f1_metric.compute(predictions=predictions, references=eval_pred.label_ids, average="macro")
    return {**recall, **f1}


def compute_metrics_2(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=-1)
    # precision, recall, f1, _ = precision_recall_fscore_support(p.label_ids.flatten(), preds.flatten(), average='weighted', zero_division=0)
    return {
        'accuracy': (preds == p.label_ids).mean(),
    }

class ComputeMetricsASR:
    def __init__(self, processor, wer_metric):
        self.processor = processor
        self.wer_metric = wer_metric

    def compute_metrics_asr(self, pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = self.processor.tokenizer.pad_token_id

        pred_str = self.processor.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = self.processor.batch_decode(pred.label_ids, group_tokens=False)

        wer = self.wer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}


def map_to_array(batch):
    speech, _ = librosa.load(batch["path"], sr=None)
    batch["speech"] = speech
    return batch


def choose_scaler(scaler_type):
    switcher = {
        'standard': lambda: StandardScaler(),
        'minmax': lambda: MinMaxScaler(),
        'robust': lambda: RobustScaler(),
        'normalizer': lambda: Normalizer()
    }
    return switcher.get(scaler_type, lambda: "Error {} is not an option! Choose from: \n {}.".format(scaler_type,
                                                                                                     switcher.keys()))()


def fit_scaler(config_bea, bea_train_flat):
    save_scaler = config_bea['data_scaling']['save_scaling_model']
    out_dir = config_bea['data_scaling']['scaling_model_path']
    emb_type = config_bea['discrimination']['emb_type']  # type of embeddings to load
    scaler_type = config_bea['data_scaling']['scaler_type']
    size = config_bea['size_bea']

    checkpoint_path = config_bea['pretrained_model_details']['checkpoint_path']
    model_used = check_model_used(checkpoint_path)

    if scaler_type is not None:
        final_out_path = '{0}_{1}_{2}_{3}_{4}.pkl'.format(out_dir, str(scaler_type), emb_type, size, model_used)

        os.makedirs(os.path.dirname(final_out_path), exist_ok=True)
        # if os.path.isfile(final_out_path):
        #     # while True:
        #     #     reply = input("Seems like a {0} scaler model was already trained:\n{1}. \nIf you changed the size of the sets, "
        #     #                   "then you may want to train the model again.\n"
        #     #                   " Do you want to retrain the scaler model? Yes or [No]: ".format(scaler_type, final_out_path) or "no")
        #     #     if reply.lower() not in ('yes', 'no'):
        #     #         print("Please, enter either 'yes' or 'no'")
        #     #         continue
        #     #     else:
        #     #         if reply.lower() == 'yes':
        #     #             print("Starting to train {} scaler...".format(scaler_type))
        #     #             # bea_train_flat = load_data(config=config_bea)  # load bea embeddings
        #     #             scaler = choose_scaler(scaler_type)
        #     #             scaler.fit(bea_train_flat)
        #     #             print("{} scaler fitted...".format(scaler_type))
        #     #
        #     #             if save_scaler:
        #     #                 pk.dump(scaler, open(final_out_path, 'wb'))
        #     #                 print("Scaler model saved to:", final_out_path)
        #     #             return scaler
        #     #         else:
        #     print("Seems like a {0} scaler model was already trained.\n Loading the existing scaler model:{1}".format(
        #         scaler_type, final_out_path))
        #     scaler = pk.load(open(final_out_path, 'rb'))
        #     return scaler
        #     # pass
        #     # break
        # else:
        print("No trained scaler found, starting to train {} scaler...".format(scaler_type))
        # bea_train_flat = load_data(config=config_bea)  # load bea embeddings
        # train Scaler
        scaler = choose_scaler(scaler_type)
        scaler.fit(bea_train_flat)
        print("{} scaler fitted...".format(scaler_type))

        if save_scaler:
            pk.dump(scaler, open(final_out_path, 'wb'))
            print("Scaler model saved to:", final_out_path)
        return scaler
    else:
        print("Skipping data scaling...")


###

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


class DataBuilder(Dataset):
    def __init__(self, data):
        self.x = data
        self.x = torch.from_numpy(self.x)
        self.len = self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index]

    def __len__(self):
        return self.len


def count_chars(file_path):
    with open(file_path, 'r') as f:
        # Use itertools.chain.from_iterable to flatten the list of lines into a single string
        full_text = ''.join(itertools.chain.from_iterable(f))
        text_no_white_spaces = ''.join(full_text.split())
        # Use collections.Counter to count the occurrence of each character
        # char_counts = collections.Counter(text)

        return len(full_text), len(text_no_white_spaces)


def create_csv_speech_tempo(in_path, out_file, size):
    """Function to create csv file for singing corpus with labels of the form:
    37+2/length wav ==> target
    31/len ==> target

    """

    if not os.path.exists(os.path.dirname(out_file)):
        os.makedirs(os.path.dirname(out_file))

    # reading directories
    # bea_train_path = os.path.join(in_path, 'bea-base-train-flat')
    transcriptions_list = glob.glob('{}/*.txt'.format(in_path))
    transcriptions_list.sort()
    wavs_list = glob.glob('{}/*.wav'.format(in_path))
    wavs_list.sort()

    if len(transcriptions_list) <= 0:
        raise ValueError("No transcriptions found in the directory: {}".format(in_path))

    df = pd.DataFrame(columns=['path', 'name', 'whole_speech', 'no_pause_speech', 'length'])

    for transcription, utterance in tqdm(zip(transcriptions_list[0:size], wavs_list[0:size]),
                                         total=len(transcriptions_list[0:size])):  #
        file_name = os.path.basename(utterance).split('.')[0]
        # check wav length
        wav, sr = sf.read(utterance)
        wav_length = len(wav)
        # print(sr)

        # check text length
        number_total_chars, with_no_spaces = count_chars(transcription)
        # print(number_total_chars, with_no_spaces, wav_length)

        # compute speech temporal params
        whole_speech = (number_total_chars + 2) / wav_length
        no_pause_speech = with_no_spaces / wav_length

        # define python dict
        data = {'path': utterance, 'name': file_name, 'whole_speech': whole_speech,
                'no_pause_speech': no_pause_speech, 'length': wav_length / sr}
        # df = df.append(data, ignore_index=True)    # append new data to the librimix csv
        df = df.append(data, ignore_index=True)  # append new data to the librimix csv
    df.to_csv(out_file, sep=',', index=False)
    if os.path.isfile(out_file):
        print("Data saved to {}".format(out_file))
    else:
        print("Error saving data to {}".format(out_file))


# create_csv_speech_tempo('/media/jvel/data/audio/Bea-base/', './test.csv', 5000)

class CustomSubset(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        labels(sequence) : targets as required for the indices. will be the same length as indices
    """

    def __init__(self, dataset, indices, labels):
        self.dataset = torch.utils.data.Subset(dataset, indices)
        self.targets = labels

    def __getitem__(self, idx):
        image = self.dataset[idx][0]
        target = self.targets[idx]
        return (image, target)

    def __len__(self):
        return len(self.targets)


def split_depisda_corpus(data):
    encoder = LabelEncoder()
    data['file_prefix'] = data['filename'].str[:6]
    data['file_prefix_enc'] = encoder.fit_transform(data['file_prefix'])
    n_grupos = data['file_prefix_enc'].values
    X = data.drop(columns=['label'])
    y = data['label']
    gss_train = GroupShuffleSplit(n_splits=1, random_state=42, test_size=0.25, train_size=0.75)
    gss_dev_test = GroupShuffleSplit(n_splits=1, random_state=42, test_size=0.25, train_size=0.75)

    for i, (train_index, temp_index) in enumerate(gss_train.split(X=X, y=y, groups=n_grupos)):
        print(f"Fold {i}:")
        print(f"  Train: index={train_index}, group={n_grupos[train_index]}")
        print(f"  Test:  index={temp_index}, group={n_grupos[temp_index]}")
        x_train, x_temp, y_train, y_temp = X.iloc[train_index], X.iloc[temp_index], y.iloc[train_index], y.iloc[
            temp_index]

    train_df = pd.concat([x_train, y_train], axis=1)

    rest_groups = x_temp['file_prefix_enc'].values
    for i, (dev_index, test_index) in enumerate(gss_dev_test.split(X=x_temp, y=y_temp, groups=rest_groups)):
        print(f"Fold {i}:")
        print(f"  Dev: index={dev_index}, group={rest_groups[dev_index]}")
        print(f"  Test:  index={test_index}, group={rest_groups[test_index]}")
        x_dev, x_test, y_dev, y_test = x_temp.iloc[dev_index], x_temp.iloc[test_index], y_temp.iloc[dev_index], \
        y_temp.iloc[test_index]

    dev_df = pd.concat([x_dev, y_dev], axis=1)
    test_df = pd.concat([x_test, y_test], axis=1)

    train_df.to_csv('./metadata/depression/depression_train.csv', sep=',', index=False)
    dev_df.to_csv('./metadata/depression/depression_dev.csv', sep=',', index=False)
    test_df.to_csv('./metadata/depression/depression_test.csv', sep=',', index=False)

    return train_df, dev_df, test_df


def get_dataset_partitions_pd(df, train_split=0.8, val_split=0.1, test_split=0.1, target_variable=None):
    assert (train_split + test_split + val_split) == 1

    # Only allows for equal validation and test splits
    assert val_split == test_split

    # Shuffle
    df_sample = df.sample(frac=1, random_state=12)

    # Specify seed to always have the same split distribution between runs
    # If target variable is provided, generate stratified sets
    if target_variable is not None:
        grouped_df = df_sample.groupby(target_variable)
        arr_list = [np.split(g, [int(train_split * len(g)), int((1 - val_split) * len(g))]) for i, g in grouped_df]

        train_ds = pd.concat([t[0] for t in arr_list])
        val_ds = pd.concat([t[1] for t in arr_list])
        test_ds = pd.concat([v[2] for v in arr_list])

    else:
        indices_or_sections = [int(train_split * len(df)), int((1 - val_split) * len(df))]
        train_ds, val_ds, test_ds = np.split(df_sample, indices_or_sections)

    return train_ds, val_ds, test_ds



