import json
import os
import re
import numpy as np

from datasets import load_dataset, DownloadMode
from transformers import AutoConfig, Wav2Vec2Processor, TrainingArguments, Trainer, Wav2Vec2CTCTokenizer, \
    Wav2Vec2FeatureExtractor, Wav2Vec2Processor

# in-house functions
from common import utils, utils_fine_tune, crate_csv_bea_from_scp, create_csv_bea_base
from common.utils_fine_tune import Wav2Vec2ForSpeechClassification

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb=860'

# inspired by https://colab.research.google.com/github/patrickvonplaten/notebooks/blob/master/Fine_tuning_Wav2Vec2_for_English_ASR.ipynb


# def train():
task = 'Bea_base'

train_path = "/srv/data/egasj/corpora/{}/bea-base-dev-train/".format(task)
out_file_train = "/srv/data/egasj/corpora/{}/metadata/bea-base-train.csv".format(task)

dev_path = "/srv/data/egasj/corpora/{}/bea-base-dev-spont-flat/".format(task)
out_file_dev = "/srv/data/egasj/corpora/{}/metadata/bea-base-dev-spont.csv".format(task)

vocab_path = "/srv/data/egasj/corpora/{}/vocab.json".format(task)

# Getting data info ready
# create_csv_bea_base(corpora_path=train_path, out_file=out_file_train)
# create_csv_bea_base(corpora_path=dev_path, out_file=out_file_dev)

# Loading the dataset into 'load_datasets' class
data_files = {
    'train': out_file_train,
    'validation': out_file_dev
}

bea_set = load_dataset('csv', data_files=data_files, delimiter=',', cache_dir='/srv/data/egasj/hf_cache/',
                       download_mode=DownloadMode['REUSE_DATASET_IF_EXISTS'])
train_set = bea_set['train']
val_set = bea_set['validation']

# Removing special chars
chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'


def remove_special_characters(batch):
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower() + " "
    return batch


train_set = train_set.map(remove_special_characters)
val_set = val_set.map(remove_special_characters)


# concatenate transcriptions into one and transform the string into a set of chars to build the vocabulary.
def extract_all_chars(batch):
    all_text = " ".join(batch["sentence"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}


vocab_train = train_set.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True,
                            remove_columns=train_set.column_names)
vocab_val = val_set.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True,
                        remove_columns=val_set.column_names)

vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_val["vocab"][0]))
vocab_dict = {v: k for k, v in enumerate(vocab_list)}
# manually adding extra letters
vocab_dict['cs'] = 36
vocab_dict['dz'] = 37
vocab_dict['dzs'] = 38
vocab_dict['gy'] = 39
vocab_dict['ly'] = 40
vocab_dict['ny'] = 41
vocab_dict['sz'] = 42
vocab_dict['ty'] = 43
vocab_dict['zs'] = 44

vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]

vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)
print("Length of the vocabulary:", len(vocab_dict))

# saving vocab if it doesn't already exist
if not os.path.isfile(vocab_path):
    with open(vocab_path, 'w') as vocab_file:
        json.dump(vocab_dict, vocab_file)

# instantiate vocabulary into a Wav2Vec2CTCTokenizer class
tokenizer = Wav2Vec2CTCTokenizer(vocab_path, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")

# feature extractor
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                                             do_normalize=True, return_attention_mask=True)

# feature extractor and tokenizer wrapped into a single Wav2Vec2Processor class
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

number_of_training_samples = 10
train_set = train_set.shuffle(seed=42).select(range(number_of_training_samples))

number_of_val_samples = 3
val_set = val_set.shuffle(seed=42).select(range(number_of_val_samples))

# reads wavs, calculates input_values, adds labels
pp = utils.PreprocessFunctionASR(processor, target_sampling_rate=16000)

train_set = train_set.map(
    pp.preprocess_function_asr,
    batch_size=128,
    batched=True,
    num_proc=16
)
val_set = val_set.map(
    pp.preprocess_function_asr,
    batch_size=128,
    batched=True,
    num_proc=16
)

# Checking a random sample
rand_int = np.random.randint(0, len(train_set) - 1)
print("Target text:", train_set[rand_int]["sentence"])
print("Input array shape:", len(train_set[rand_int]["audio"]))
print("File:", train_set[rand_int]["file"])

### DATA PREPROCESSING


# processor = Wav2Vec2Processor.from_pretrained(model_name_or_path)
# target_sampling_rate = processor.feature_extractor.sampling_rate
# print(f"The target sampling rate: {target_sampling_rate}")
#
# pp = utils.PreprocessFunctionASR(processor, target_sampling_rate=16000)
