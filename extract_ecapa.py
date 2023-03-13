import torch
import torchaudio


from datasets import load_dataset, DownloadMode
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model, Wav2Vec2Processor, Data2VecAudioForCTC

from extract_helper import extract_embeddings, extract_embeddings_original, extract_embeddings_gabor, \
    extract_embeddings_and_save, Encoder, extract_ecapa_and_save, extract_ecapa_original
from common import utils


# Loading configuration
# config = utils.load_config('config/config_bea16k.yml')  # provide the task's yml
config = utils.load_config('config/config_depression.yml')
model_name = config['pretrained_model_details']['checkpoint_path']
task = config['task']  # name of the dataset
audio_path = config['paths']['audio_path']  # path to the audio files of the task
label_file = config['paths']['to_labels']  # path to the labels of the dataset
save_path = config['paths']['to_save_metadata']  # path to save the csv file containing info of the dataset (metadata)
# size_bea = config['size_bea']

# Generating labels (comment this if already generated)
# utils.create_csv_sm(in_path=audio_path, out_file=label_file)
# utils.create_csv_bea_base(corpora_path=audio_path, out_file=label_file)

# loading data
# data = pd.read_csv(label_file)  # reading labels
# os.makedirs(save_path, exist_ok=True)  # creating dir for the csv
# data.to_csv(f"{save_path}/metadata.csv", sep=",", encoding="utf-8", index=False)  # saving to csv

# Load data in HF 'datasets' class format
data_files = {
    "train": label_file # this is the metadata
}

dataset = load_dataset("csv", data_files=data_files, delimiter=",", cache_dir=config['hf_cache_dir'],
                       download_mode=DownloadMode['REUSE_DATASET_IF_EXISTS'])
train_dataset = dataset["train"]
train_dataset = train_dataset.map(utils.map_to_array)

model = Encoder.from_hparams(
    source=config['pretrained_model_details']['checkpoint_path']
)

extract_ecapa_original([train_dataset], model, config)
