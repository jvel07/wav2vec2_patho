import glob

import pandas
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification, Wav2Vec2ForPreTraining, Wav2Vec2Processor, Wav2Vec2Model
from datasets import load_dataset, load_metric
import pandas as pd
from transformers import pipeline
import os
import re
import numpy as np
# from common.utils import map_to_array
import soundfile as sf
#%%
# from common import utils

def map_to_array(batch):
    speech, _ = sf.read(batch["path"])
    batch["speech"] = speech
    return batch

pre_trained_model = 'bea16k_1.0_hungarian'
path = '/srv/data/egasj/code/wav2vec2_patho_deep4/'
checkpoint = '1130'
model_name = '{0}runs/{1}/checkpoint-{2}/'.format(path, pre_trained_model, checkpoint)
# model_name = 'jonatasgrosman/wav2vec2-large-xlsr-53-hungarian'
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
# feature_extractor = Wav2Vec2Processor.from_pretrained(model_name)  # use this if the model has Wav2Vec2CTCTokenizer
model = Wav2Vec2Model.from_pretrained(model_name)
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(device)

# load data
task = 'demencia_wav16k_selected225-B'  # name of the dataset
audio_path = "/srv/data/egasj/corpora/{}/".format(task)  # path to the audio files of the task
label_file = path + 'audio/labels/diagnosis-keep-diagA-75.txt'  # path to the labels of the dataset
save_path = '{0}data/{1}'.format(path, task)  # path to save the csv file containing info of the dataset

data = pd.read_csv(label_file, header=None, names=['label'])  # reading labels
list_wavs = glob.glob('{}*.wav'.format(audio_path))  # getting audio paths
list_wavs.append(list_wavs[2])  # temp => one file is missing
data['path'] = list_wavs  # getting the paths into the dataframe
# save to csv
os.makedirs(save_path, exist_ok=True)  # creating dir for the csv
data.to_csv(f"{save_path}/train.csv", sep="\t", encoding="utf-8", index=False)  # saving csv

# Load data in HF 'datasets' class format
data_files = {
    "train": "../data/{}/train.csv".format(task),
}

dataset = load_dataset("csv", data_files=data_files, delimiter="\t", )
train_dataset = dataset["train"]
train_dataset = train_dataset.map(map_to_array, batch_size=8, num_proc=8)

# EXTRACT FEATURES
print("Getting features extracted...")
batch_size = 1
list_convs = []
files = []
final_dict = dict()
list_hidden = []

for index, i2 in enumerate([train_dataset]):

    #     if index == 0:
    #         set_ = 'train'
    # if index == 0:
    #     set_ = 'dev'
    # else:
    #     set_ = 'test.txt'
    # set_='test.txt'

    tot_input_values = feature_extractor(i2['speech'], return_tensors="pt", padding=True,
                                         feature_size=1, sampling_rate=16000 )
    # tot_input_values.to(device)
    print("Features extracted, now computing the outputs!")
    #
    for i in range(0, len(i2), 2):
        #     print(train_dataset[i:i+batch_size]['file_name'])
        #     files.append(train_dataset[i:i+batch_size]['file_name'])
        #     input_values = feature_extractor(train_dataset[i:i+batch_size]["speech"], return_tensors="pt", padding=True,
        #                                      feature_size=1, sampling_rate=16000 )#.input_values  # Batch size 1
        #     outputs = model(**tot_input_values[i:i+batch_size])
        outputs = model(tot_input_values.input_values[i:i+batch_size], tot_input_values.attention_mask[i:i+batch_size])

        # extract features from the last CNN layer
        convs = outputs.extract_features.detach().numpy()
        list_convs.append(convs)

        # extract features corresponding to the sequence of last hidden states
        hidden = outputs.last_hidden_state.detach().numpy()
        list_hidden.append(hidden)

        print(batch_size+i)

    embs = np.asanyarray(np.vstack(list_convs))
    hiddens = np.asanyarray(np.vstack(list_hidden))
    np.save("../data/{0}/embeddings/{1}_convs_wav2vec2".format(task), embs)
    np.save("../data/{0}/embeddings/{1}_hiddens_wav2vec2".format(task), hiddens)
