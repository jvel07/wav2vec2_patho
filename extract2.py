import glob
import os
import yaml

import pandas as pd
import soundfile as sf
from datasets import load_dataset, DownloadMode
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

from extract_helper import extract_embeddings
from common import utils


def map_to_array(batch):
    speech, _ = sf.read(batch["path"])
    batch["speech"] = speech
    return batch


# Loading configuration
config = utils.load_config('config.yml')
model_name = config['pretrained_model_details']['checkpoint_path']

# Loading feature extractor and model
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
# feature_extractor = Wav2Vec2Processor.from_pretrained(model_name)  # use this if the model has Wav2Vec2CTCTokenizer
model = Wav2Vec2Model.from_pretrained(model_name)

# load data
task = config['task']  # name of the dataset
audio_path = config['paths']['audio_path']  # path to the audio files of the task
label_file = config['paths']['to_labels']  # path to the labels of the dataset
save_path = config['paths']['to_save_csv']  # path to save the csv file containing info of the dataset

data = pd.read_csv(label_file, header=None, names=['label'])  # reading labels
list_wavs = glob.glob('{}*.wav'.format(audio_path))  # getting audio paths
list_wavs.append(list_wavs[2])  # temp => one file is missing
data['path'] = list_wavs  # getting the paths into the dataframe
# save to csv
os.makedirs(save_path, exist_ok=True)  # creating dir for the csv
data.to_csv(f"{save_path}/train.csv", sep=",", encoding="utf-8", index=False)  # saving csv

# Load data in HF 'datasets' class format
data_files = {
    "train": "data/{}/train.csv".format(task),
}

dataset = load_dataset("csv", data_files=data_files, delimiter=",", cache_dir='/srv/data/egasj/hf_cache/',
                       download_mode=DownloadMode['REUSE_DATASET_IF_EXISTS'])
train_dataset = dataset["train"]
train_dataset = train_dataset.map(map_to_array, batch_size=8, num_proc=8)

# EXTRACT FEATURES
extract_embeddings(dataset_list=[train_dataset], feature_extractor=feature_extractor, batch_size=1, model=model)
