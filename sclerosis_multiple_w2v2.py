import os

import torch
from datasets import DownloadMode, load_dataset, load_metric
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model, Wav2Vec2Processor, AutoConfig

from common import utils, fit_scaler, results_to_csv, DataBuilder, Wav2Vec2ForSpeechClassification
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from discrimination.discrimination_utils import  check_model_used

config = utils.load_config('config/config_sm.yml')  # loading configuration
# config_bea = utils.load_config('config/config_bea16k.yml')  # loading configuration for bea dataset (PCA, std)
shuffle_data = config['shuffle_data']  # whether to shuffle the training data
label_file = config['paths']['to_labels']  # path to the labels of the dataset
output_results = config['paths']['output_results']  # path to csv for saving the results
audio_path = config['paths']['audio_path']  # path to the audio files of the task

checkpoint_path = config['pretrained_model_details']['checkpoint_path']
model_used = check_model_used(checkpoint_path)

# create labels if not already created
utils.create_csv_sm(in_path=audio_path, out_file=label_file) # sclerosis multiplie

# Load data in HF 'datasets' class format
data_files = {
    "train": label_file # this is the metadata
}

dataset = load_dataset("csv", data_files=data_files, delimiter=",", cache_dir=config['hf_cache_dir'],
                       download_mode=DownloadMode['REUSE_DATASET_IF_EXISTS'])
train_dataset = dataset["train"]
train_dataset = train_dataset.map(utils.map_to_array, batched=False, num_proc=8)

# bea_train_flat = load_data(config=config_bea)  # load bea embeddings
df_labels = pd.read_csv(label_file)  # loading labels
# data['label'] = df_labels.label.values  # adding labels to data

# Inference with wav2vec2
print("Using", config['discrimination']['emb_type'])
df = pd.DataFrame(columns=['c', 'acc', 'f1', 'prec', 'recall', 'auc', 'eer'])

# Loading feature extractor and model
model_name = config['pretrained_model_details']['checkpoint_path']
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
# processor = Wav2Vec2Processor.from_pretrained(model_name)  # use this if the model has Wav2Vec2CTCTokenizer
model = Wav2Vec2ForSpeechClassification.from_pretrained(model_name)  # ==> this is for inference (gives logits)
# model = Wav2Vec2Model.from_pretrained(model_name)  # ==> this is for feature extraction (gives embeddings)
# model = Data2VecAudioForCTC.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)

def iterate_utterance(batch):
    sampling_rate = 16000
    chunk_size = 30
    frame_step = chunk_size * sampling_rate

    list_current_utterance_logits = []
    utterance = batch['speech']
    # iterating frames of the utterance
    for frame in (pbar := tqdm(range(0, len(utterance), frame_step), desc="Computing features", position=0)):
        # getting the frame
        current_segment = utterance[frame:frame + frame_step]
        pbar.set_description(f"Computing features for frame {frame} to {frame + frame_step}")
        # if the frame is shorter than the frame length, we pad it with zeros
        if len(current_segment) < frame_step:
            current_segment = np.pad(current_segment, (0, frame_step - len(current_segment)), 'constant')
        # computing features for the segment
        input_values_segment = feature_extractor(current_segment, return_tensors="pt", padding=True,
                                                 feature_size=1, sampling_rate=sampling_rate)
        with torch.no_grad():
            logits_segment = model(input_values_segment.input_values, input_values_segment.attention_mask).logits
        list_current_utterance_logits.append(logits_segment)

    whole_utterance_logits = torch.mean(torch.vstack(list_current_utterance_logits), dim=0)
    return whole_utterance_logits

def predict(batch):


    pred_ids = torch.argmax(logits, dim=-1).detach().cpu().numpy()
    batch["predicted"] = pred_ids
    return batch

label_names = [config.id2label[i] for i in range(config.num_labels)]

f1_scores = []
for i in range(len(train_dataset)):
    wav_logits = iterate_utterance(train_dataset[i]['speech'])
    y_true = train_dataset[i]['label']
    y_pred = torch.argmax(wav_logits, dim=-1).detach().cpu().numpy()
    f1 = f1_score(np.ravel(y_true), np.ravel(y_pred), average='weighted')
    f1_scores.append(f1)
    print(f1)
print("Overall F1 score:", np.mean(f1_scores))







