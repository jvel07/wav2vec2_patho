import os

import torch.utils.data
from datasets import load_dataset, DownloadMode
from transformers import AutoConfig, Wav2Vec2Processor, TrainingArguments, Trainer, Seq2SeqTrainer

# in-house functions
from common import utils, utils_fine_tune, crate_csv_bea_from_scp
from common.utils_fine_tune import Wav2Vec2ForSpeechClassification

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

config = utils.load_config('config/config_sm.yml')
task = 'bea-base-train-flat'
# Loading the dataset into 'load_datasets' class
size = 5000  # size of the sub-set of the data to use
tempo_target = 'no_pause_speech'
labels_train = 'data/{}/{}_train_{}.csv'.format(task, tempo_target, size)
labels_dev = 'data/{}/{}_dev_{}.csv'.format(task, tempo_target, size)

data_files = {
    'train': labels_train,
    'validation': labels_dev
}

bea16k_set = load_dataset('csv', data_files=data_files, delimiter=',', cache_dir=config['hf_cache_dir'],
                          download_mode=DownloadMode['REUSE_DATASET_IF_EXISTS'])
train_set = bea16k_set['train']
val_set = bea16k_set['validation']

# Getting unique labels
label_list = train_set.unique('speed')
label_list.sort()
num_labels = len(label_list)

# Configurations
# lang = 'english'
model_name_or_path = 'jonatasgrosman/wav2vec2-large-xlsr-53-hungarian'
pooling_mode = "mean"

config = AutoConfig.from_pretrained(
    model_name_or_path,
    num_labels=num_labels,
    label2id={label: i for i, label in enumerate(label_list)},
    id2label={i: label for i, label in enumerate(label_list)},
    finetuning_task="wav2vec2_clf",
    mask_time_length=8,
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
    batch_size=100,
    batched=True,
    num_proc=16
    # keep_in_memory=True
)
print("Train dataset generated successfully...\n")

eval_dataset = val_set.map(
    pp.preprocess_function,
    batch_size=100,
    batched=True,
    num_proc=16
    # keep_in_memory=True
)
print("Validation dataset generated successfully...\n")

pp = utils.PreprocessFunction(processor, label_list, target_sampling_rate)

print("Generating the datasets...\n")
# Preprocess data
train_dataset = train_set.map(
    pp.preprocess_function,
    batch_size=20,
    batched=True,
    num_proc=4
    # keep_in_memory=True
)
print("Train dataset generated successfully...\n")

eval_dataset = val_set.map(
    pp.preprocess_function,
    batch_size=20,
    batched=True,
    num_proc=4
    # keep_in_memory=True
)
print("Validation dataset generated successfully...\n")

# Setting-up the trainer
data_collator = utils_fine_tune.DataCollatorCTCWithPadding(processor=processor, padding=True)

# Load pre-trained model to fine-tune
model = Wav2Vec2ForSpeechClassification.from_pretrained(
    model_name_or_path,
    config=config,
)
model.freeze_feature_extractor()

epochs_list = [5.0, 10.0]
for num_train_epochs in epochs_list:
    out_dir = 'runs/{0}_{1}_{2}'.format(task, num_train_epochs, tempo_target)
    training_args = TrainingArguments(
        output_dir=out_dir,
        # output_dir="/content/gdrive/MyDrive/wav2vec2-xlsr-greek-speech-emotion-recognition"
        per_device_train_batch_size=4,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        evaluation_strategy="steps",
        num_train_epochs=num_train_epochs,
        fp16=False,
        # save_steps=10,
        # eval_steps=10,
        # logging_steps=10,
        learning_rate=3e-5,
        save_total_limit=2,
        prediction_loss_only=True,
    )

    trainer = utils_fine_tune.CTCTrainer(
    # trainer = Seq2SeqTrainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=utils.compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor.feature_extractor,
    )


    trainer.train()
    # trainer.save_model(out_dir)