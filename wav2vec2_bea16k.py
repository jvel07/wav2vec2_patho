import os

import torch
from datasets import load_dataset, DownloadMode
from torch.nn import CrossEntropyLoss
from transformers import AutoConfig, Wav2Vec2Processor, TrainingArguments, Trainer, AutoModelForAudioClassification

# in-house functions
from common import utils, utils_fine_tune, crate_csv_bea_from_scp, create_csv_eating
from common.utils_fine_tune import Wav2Vec2ForSpeechClassification


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb=860'

# inspired by https://colab.research.google.com/github/m3hrdadfi/soxan/blob/main/notebooks/Emotion_recognition_in_Greek_speech_using_Wav2Vec2.ipynb#scrollTo=ZXVl9qW1Gw_-

config = utils.load_config('config/config_sm.yml')

task = 'bea-base-train-flat'

# Getting data info ready
save_path = '/srv/data/egasj/corpora/labels/{}/'.format(task)
audio_path = "/srv/data/egasj/corpora/{}/".format(task)
scp_file = "/srv/data/egasj/corpora/labels/{}/wav.txt".format(task)

labels_train = 'data/{}/train.csv'
labels_dev = 'data/{}/dev.csv'

create_csv_eating(audio_path, labels_train, labels_dev)

# Loading the dataset into 'load_datasets' class
data_files = {
    'train': labels_train,
    'validation': labels_dev
}

whole_set = load_dataset('csv', data_files=data_files, delimiter=',', cache_dir=config['hf_cache_dir'],
                         download_mode=DownloadMode['REUSE_DATASET_IF_EXISTS'])
train_set = whole_set['train']
val_set = whole_set['validation']
print("Length of the training set: {}".format(len(train_set)))

# Getting unique labels
label_list = train_set.unique('label')
label_list.sort()
num_labels = len(label_list)

# Configurations
lang = 'english'
model_name_or_path = 'jonatasgrosman/wav2vec2-large-xlsr-53-german'
pooling_mode = "mean"

config = AutoConfig.from_pretrained(
    model_name_or_path,
    num_labels=num_labels,
    label2id={label: i for i, label in enumerate(label_list)},
    id2label={i: label for i, label in enumerate(label_list)},
    finetuning_task="wav2vec2_clf",
    cache_dir=config['hf_cache_dir'],
    problem_type='single_label_classification'
    # loss=CrossEntropyLoss(),
)
setattr(config, 'pooling_mode', pooling_mode)

processor = Wav2Vec2Processor.from_pretrained(model_name_or_path)
target_sampling_rate = processor.feature_extractor.sampling_rate
print(f"The target sampling rate: {target_sampling_rate}")

pp = utils.PreprocessFunction(processor, label_list, target_sampling_rate)

print("Generating the datasets...\n")
# Preprocess data
train_dataset = train_set.map(
    pp.preprocess_function,
    batch_size=16,
    batched=True,
    num_proc=4
    # keep_in_memory=True
)
print("Train dataset generated successfully...\n")

eval_dataset = val_set.map(
    pp.preprocess_function,
    batch_size=16,
    batched=True,
    num_proc=4,
    # keep_in_memory=True
)
print("Validation dataset generated successfully...\n")

# Setting-up the trainer
data_collator = utils_fine_tune.DataCollatorCTCWithPadding(processor=processor, padding=True)

# Define evaluation metrics
is_regression = False

# Load pre-trained model to fine-tune
model = Wav2Vec2ForSpeechClassification.from_pretrained(
    model_name_or_path,
    config=config,
)
#
# model = AutoModelForAudioClassification.from_pretrained(
#     model_name_or_path,
#     config=config,
# )
# model.classifier = torch.nn.Linear(in_features=256, out_features=4, bias=True)

# Freeze CNN blocks
model.freeze_feature_extractor()

# Define trainers and train model

epochs_list = [5.0, 10.0]
for num_train_epochs in epochs_list:
    out_dir = '/srv/data/egasj/code/wav2vec2_patho_deep4/runs/{0}_{1}_{2}'.format(task, num_train_epochs, lang)
    training_args = TrainingArguments(
        output_dir=out_dir,
        # output_dir="/content/gdrive/MyDrive/wav2vec2-xlsr-greek-speech-emotion-recognition"
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        evaluation_strategy="steps",
        num_train_epochs=1.0,
        fp16=True,
        save_steps=10,
        eval_steps=10,
        logging_steps=10,
        learning_rate=1e-4,
        save_total_limit=2,
    )

    trainer = utils_fine_tune.CTCTrainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=utils.compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor.feature_extractor,
    )

    # trainer = Trainer(
    #     model=model,
    #     data_collator=data_collator,
    #     args=training_args,
    #     compute_metrics=utils.compute_metrics,
    #     train_dataset=train_dataset,
    #     eval_dataset=eval_dataset,
    #     tokenizer=processor.feature_extractor,
    # )

    trainer.train()
    trainer.save_model(out_dir)
