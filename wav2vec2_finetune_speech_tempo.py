from datasets import load_dataset, Dataset, load_metric, Audio, concatenate_datasets, DownloadMode
import torch
from transformers import pipeline, AutoFeatureExtractor, AutoTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, \
    AutoModelForAudioClassification, TrainingArguments, Trainer, AutoProcessor
import soundfile as sf
import librosa
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score

from common import utils, create_csv_speech_tempo

# example of increase in mean squared error
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error


def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    if "-superb-" in model_name:
        preds = np.argmax(eval_pred.predictions[0], axis=1)
    else:
        preds = np.argmax(eval_pred.predictions, axis=1)
    # recall = recall_metric.compute(predictions=predictions, references=eval_pred.label_ids, average="macro")
    # f1 = f1_metric.compute(predictions=predictions, references=eval_pred.label_ids, average="macro")
    # return {"f1": f1, "spearmanr": spearmanr}
    score = mean_squared_error(y_true=eval_pred.label_ids, y_pred=preds)
    return score


# https://github.com/aalto-speech/ComParE2022

freeze_feature_extractor = False
freeze_transformer = False

config = utils.load_config('config/config_bea16k.yml')

task = 'bea-base-train-flat'
audio_base ='/srv/data/egasj/corpora/Bea_base/'
size = 5000  # size of the sub-set of the data to use
tempo_target = 'no_pause_speech'
labels_train = 'data/{}/{}_train_{}.csv'.format(task, tempo_target, size)
labels_dev = 'data/{}/{}_dev_{}.csv'.format(task, tempo_target, size)

# getting tempo labels ready (do this for dev as well) --> uncomment if no csv files are available
# create_csv_speech_tempo(in_path='{}bea-base-train-flat'.format(audio_base),
#                         out_file=labels_train, size=size)
# create_csv_speech_tempo(in_path='{}bea-base-dev-spont-flat'.format(audio_base),
#                         out_file=labels_dev, size=size)

# Loading the dataset into 'load_datasets' class
data_files = {
    'train': labels_train,
    'validation': labels_dev
}

dataset = load_dataset('csv', data_files=data_files, delimiter=',', cache_dir=config['hf_cache_dir'],
                        download_mode=DownloadMode['REUSE_DATASET_IF_EXISTS'])
train_set = dataset['train']
dev_set = dataset['validation']

model_name = config['pretrained_model_details']['checkpoint_path']

# define feature extractor
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name, cache_dir=config['hf_cache_dir'])

processor = Wav2Vec2Processor.from_pretrained(model_name)


try:
    processor = AutoProcessor.from_pretrained(model_name, cache_dir=config['hf_cache_dir'])
except (OSError, ValueError) as e:
    print(f"No tokenizer found for {model_name}")
    processor = None


pp = utils.PreprocessFunctionASR(processor, target_sampling_rate=16000)

train_set = train_set.map(
    pp.preprocess_function_tempo,
    batched=True,
    batch_size=1,
)

dev_set = dev_set.map(
    pp.preprocess_function_tempo,
    batched=True,
    batch_size=1,
)

model = AutoModelForAudioClassification.from_pretrained(
    model_name,
    trust_remote_code=True,
    cache_dir=config['hf_cache_dir']
)
model.config.id2label = None
model.config.label2id = None
# check speech_notebook/speech_tempo_labelling.ipynb to see how the labels where created.
# ==>  thresholds = [0.00035, 0.0006, 0.0008, 0.0012]
# == > labels = ['slow', 'mid-slow', 'normal', 'fast']
model.config.num_labels = 4  # 37+2/length wav ==> target; 31/len ==> target
model.classifier = torch.nn.Linear(in_features=256, out_features=4, bias=True)


if freeze_feature_extractor:
    model.freeze_feature_extractor()
if freeze_transformer:
    model.freeze_transformer()

num_train_epochs = 10
out_dir = '/srv/data/egasj/code/wav2vec2_patho_deep4/runs/{0}_{1}_{2}'.format(task, num_train_epochs, )
args = TrainingArguments(
    output_dir=out_dir,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    per_device_eval_batch_size=1,
    num_train_epochs=10,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="recall",
    push_to_hub=False,
    gradient_checkpointing=True,
    save_total_limit=5
)

trainer = Trainer(
    model,
    args,
    train_dataset=train_set,
    eval_dataset=dev_set,
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics,
)
trainer.train()

predictions = trainer.predict(dev_set)
print(compute_metrics(predictions))
