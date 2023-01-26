import json
import os
import re
import numpy as np

from datasets import load_dataset, DownloadMode, load_metric
from transformers import AutoConfig, Wav2Vec2Processor, TrainingArguments, Trainer, Wav2Vec2CTCTokenizer, \
    Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC

# in-house functions
from common import utils, utils_fine_tune_asr, crate_csv_bea_from_scp, create_csv_bea_base
from common.utils_fine_tune import Wav2Vec2ForSpeechClassification

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb=860'

# inspired by https://colab.research.google.com/github/patrickvonplaten/notebooks/blob/master/Fine_tuning_Wav2Vec2_for_English_ASR.ipynb


# def train():
task = 'CovidSpeech'

audio_base ='/media/jvel/data/audio/{}/'

train_path = "/media/jvel/data/audio/CovidSpeech/".format(task)
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

# Combinations (total of 69176, 4344):
# E.g.: 20% --> tr: ~13800; val: ~900 (rounded)
use_percentage = 35
len_orig_train = (len(train_set))
number_of_training_samples = round((len_orig_train * use_percentage) / 100)
train_set = train_set.shuffle(seed=42).select(range(number_of_training_samples))

len_orig_val = (len(val_set))
number_of_val_samples = round((len_orig_val * use_percentage) / 100)
val_set = val_set.shuffle(seed=42).select(range(number_of_val_samples))

print("Using {}% of the data:\n {}/{} training samples \n {}/{} validation samples.".format(use_percentage,
                                                                                            number_of_training_samples,
                                                                                            len_orig_train,
                                                                                            number_of_val_samples,
                                                                                            len_orig_val))

# reads wavs, calculates input_values, adds labels
pp = utils.PreprocessFunctionASR(processor, target_sampling_rate=16000)

train_set = train_set.map(
    pp.preprocess_function_asr,
    batch_size=128,
    batched=True,
    num_proc=4
)
val_set = val_set.map(
    pp.preprocess_function_asr,
    batch_size=128,
    batched=True,
    num_proc=4
)

# Checking a random sample
rand_int = np.random.randint(0, len(train_set) - 1)
print("Target text:", train_set[rand_int]["sentence"])
print("Input array shape:", len(train_set[rand_int]["input_values"]))
print("File:", train_set[rand_int]["file"])

# # Setting-up the trainer
data_collator = utils_fine_tune_asr.DataCollatorCTCWithPadding(processor=processor, padding=True)
wer_metric = load_metric("wer")

model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-large-xlsr-53",
    attention_dropout=0.1,
    hidden_dropout=0.1,
    feat_proj_dropout=0.0,
    mask_time_prob=0.05,
    layerdrop=0.1,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer)
)

model.freeze_feature_extractor()
model.gradient_checkpointing_enable()

training_args = TrainingArguments(
    # output_dir="/content/gdrive/MyDrive/wav2vec2-large-xlsr-turkish-demo",
    output_dir="./wav2vec2-large-xlsr-beaBase-{}percent".format(use_percentage),
    group_by_length=True,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    evaluation_strategy="steps",
    num_train_epochs=5,
    fp16=True,
    save_steps=100,
    eval_steps=100,
    logging_steps=10,
    learning_rate=3e-4,
    warmup_steps=500,
    save_total_limit=2,
)

mm = utils.ComputeMetricsASR(processor, wer_metric)

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=mm.compute_metrics_asr,
    train_dataset=train_set,
    eval_dataset=val_set,
    tokenizer=processor.feature_extractor,
)

trainer.train()
