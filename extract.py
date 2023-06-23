import glob
import os

import pandas as pd
import soundfile as sf
from datasets import load_dataset, DownloadMode
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model, Wav2Vec2Processor, Data2VecAudioForCTC

from extract_helper import extract_embeddings, extract_embeddings_original, extract_embeddings_gabor, \
    extract_embeddings_and_save
from common import utils

# Loading configuration
# config = utils.load_config('config/config_bea16k.yml')  # provide the task's yml
config = utils.load_config('config/config_requests.yml')
model_name = config['pretrained_model_details']['checkpoint_path']
task = config['task']  # name of the dataset
audio_path = config['paths']['audio_path']  # path to the audio files of the task
label_path = config['paths']['to_labels']  # path to the labels of the dataset
save_path = config['paths']['to_save_metadata']  # path to save the csv file containing info of the dataset (metadata)
# size_bea = config['size_bea']

# Generating labels (comment this if already generated)
# utils.create_csv_sm(in_path=audio_path, out_file=label_file) # sclerosis multiplie
# utils.create_csv_bea_base(corpora_path=audio_path, out_file=label_file) # BEA
# utils.create_csv_depression(in_path=audio_path, out_file=label_file)  # DESPISA (depression)
utils.create_csv_compare_23(in_path=audio_path, out_file=label_path)  #

# loading data
# data = pd.read_csv(label_file)  # reading labels
# os.makedirs(save_path, exist_ok=True)  # creating dir for the csv
# data.to_csv(f"{save_path}/metadata.csv", sep=",", encoding="utf-8", index=False)  # saving to csv


# Loading feature extractor and model
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
# feature_extractor = Wav2Vec2Processor.from_pretrained(model_name)  # use this if the model has Wav2Vec2CTCTokenizer
model = Wav2Vec2Model.from_pretrained(model_name)
# model = Data2VecAudioForCTC.from_pretrained(model_name)

# Load data in HF 'datasets' class format
data_files = {
    "train": label_path + 'train.csv',  # this is the metadata
    "dev": label_path + 'dev.csv',  # this is the metadata
    "test": label_path + 'test.csv'  # this is the metadata
}

dataset = load_dataset("csv", data_files=data_files, delimiter=",", cache_dir=config['hf_cache_dir'],
                       download_mode=DownloadMode['REUSE_DATASET_IF_EXISTS'])
train_dataset = dataset["train"]
train_dataset = train_dataset.map(utils.map_to_array, num_proc=24)

# EXTRACT FEATURES
# extract_embeddings(dataset_list=[train_dataset], feature_extractor=feature_extractor, model=model, chunk_size=30)
# extract_embeddings_original(dataset_list=[train_dataset], feature_extractor=feature_extractor, model=model)
extract_embeddings_and_save(dataset_list=[train_dataset], feature_extractor=feature_extractor, model=model,
                            chunk_size=30, config=config)
# extract_embeddings_gabor(dataset_list=[train_dataset], feature_extractor=feature_extractor, model=model, chunk_size=30)
