import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification, Wav2Vec2ForPreTraining, \
    Wav2Vec2Processor, Wav2Vec2Model, EvalPrediction
from datasets import load_dataset
import pandas as pd
from transformers import pipeline
import os
import re
import numpy as np
import glob
from datasets import load_dataset
import soundfile as sf


def read_audio_wavs(path, is_sorted=True):
    """Function to read wav files from a given path.

    :param path: str, Path to the folder containing the wav files.
    :param sorted: bool, Whether to sort or not the List.
    :return: List, returns the list of wavs read.
    """
    if is_sorted:
        return sorted(glob.glob('{}*.wav'.format(path)))
    else:
        return glob.glob('{}*.wav'.format(path))


def create_labels_bea(audio_list, out_path):
    """Function to create csv file for the BEA Corpus with labels of the form:
    'file_name', 'label'

    :param audio_list: List, a list containing the wav files paths.
    # :return: List, final csv list.
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
    df.to_csv('{}labels.csv'.format(out_path), sep=',', index=False)
    # np.savetxt('{}labels.csv'.format(out_path), final_list, fmt="%s", delimiter=' ')
    print("Labels saved to {}".format(out_path))


"""FUNCTIONS FOR AUDIO PREPROCESSING"""
def speech_to_array(path):
    speech, _ = sf.read(path)
    #     batch["speech"] = speech
    return speech


def label_to_id(label, label_list):
    if len(label_list) > 0:
        return label_list.index(label) if label in label_list else -1

    return label


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


def compute_metrics(p: EvalPrediction, is_regression=True):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)

    if is_regression:
        return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
    else:
        return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}