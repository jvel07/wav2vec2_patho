import csv

import yaml
from sklearn.model_selection import train_test_split
import glob
import os

import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
from transformers import EvalPrediction
from yaml import SafeLoader
from tqdm import tqdm
import pickle as pk

from discrimination.discrimination_utils import load_data


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

    df.to_csv(out_file, sep=',', index=False)
    print("Data saved to {}".format(out_file))


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

    df = pd.DataFrame([sub.split(",") for sub in final_list], columns=['path', 'label'])

    df.to_csv(out_file, sep=',', index=False)
    print("Data saved to {}".format(out_file))


"""FUNCTIONS FOR AUDIO PREPROCESSING"""
def label_to_id(label, label_list):
    if len(label_list) > 0:
        return label_list.index(label) if label in label_list else -1

    return label


def speech_to_array(path):
    speech, _ = sf.read(path)
    #     batch["speech"] = speech
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

    if is_regression:
        return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
    else:
        return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}
        # return {"uar": recall_score(p.label_ids, preds, labels=[1, 0], average='macro')}


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
    speech, _ = sf.read(batch["file"])
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

    if scaler_type is not None:
        final_out_path = '{0}_{1}_{2}.pkl'.format(out_dir, str(scaler_type), emb_type)

        os.makedirs(os.path.dirname(final_out_path), exist_ok=True)
        if os.path.isfile(final_out_path):
            while True:
                reply = input("Seems like a {0} scaler model was already trained:\n{1}. \nIf you changed the size of the sets, "
                              "then you may want to train the model again.\n"
                              " Do you want to retrain the scaler model? Yes or [No]: ".format(scaler_type, final_out_path) or "no")
                if reply.lower() not in ('yes', 'no'):
                    print("Please, enter either 'yes' or 'no'")
                    continue
                else:
                    if reply.lower() == 'yes':
                        print("Starting to train {} scaler...".format(scaler_type))
                        # bea_train_flat = load_data(config=config_bea)  # load bea embeddings
                        scaler = choose_scaler(scaler_type)
                        scaler.fit(bea_train_flat)
                        print("{} scaler fitted...".format(scaler_type))

                        if save_scaler:
                            pk.dump(scaler, open(final_out_path, 'wb'))
                            print("Scaler model saved to:", final_out_path)
                        return scaler
                    else:
                        print("You chose {}. Loading the existing scaler model...".format(reply))
                        scaler = pk.load(open(final_out_path, 'rb'))
                        return scaler
                        # pass
                    # break
        else:
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



