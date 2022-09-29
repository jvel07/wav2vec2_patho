import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification, Wav2Vec2ForPreTraining, Wav2Vec2Processor, Wav2Vec2Model
from datasets import load_dataset
import pandas as pd
from transformers import pipeline
import os
import re
import numpy as np

#%%
task = 'bea16k_1.0_'
path = '/srv/data/egasj/code/wav2vec2_patho_deep4/'
checkpoint = '860'
model_name = '{0}{1}/checkpoint-{2}/'.format(path, task, checkpoint)
# model_name = 'jonatasgrosman/wav2vec2-large-xlsr-53-hungarian'
feature_extractor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2Model.from_pretrained(model_name)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)


# EXTRACT FEATURES
batch_size = 3
list_convs = []
files = []
final_dict = dict()
list_hidden = []

for index, i in enumerate([train_dataset]):

    #     if index == 0:
    #         set_ = 'train'
    # if index == 0:
    #     set_ = 'dev'
    # else:
    #     set_ = 'test.txt'
    set_='test.txt'

    tot_input_values = feature_extractor(i['speech'], return_tensors="pt", padding=True,
                                         feature_size=1, sampling_rate=16000 )
    # tot_input_values.to(device)
    print("Features extracted, now computing the outputs!")

    for i in range(0, len(i), batch_size):
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
    np.save("data/{0}/embeddings/{1}_convs_wav2vec2_{2}".format(task, set_, label_file_name), embs)
    np.save("data/{0}/embeddings/{1}_hiddens_wav2vec2_{2}".format(task, set_, label_file_name), hiddens)