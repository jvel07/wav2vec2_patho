import os

from datasets import load_dataset, DownloadMode
from transformers import AutoConfig, Wav2Vec2Processor, TrainingArguments, Trainer

# in-house functions
from common import utils, utils_fine_tune, crate_csv_bea_from_scp
from common.utils_fine_tune import Wav2Vec2ForSpeechClassification


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb=860'

# inspired by https://colab.research.google.com/github/m3hrdadfi/soxan/blob/main/notebooks/Emotion_recognition_in_Greek_speech_using_Wav2Vec2.ipynb#scrollTo=ZXVl9qW1Gw_-

config = utils.load_config('config/config_sm.yml')

task = '10_narrative_recall'
audio_base ='/media/jvel/data/audio/SM/{}/'.format(task)
labels = 'data/{}/labels.csv'.format(task)

train_path = os.path.join(task, audio_base, 'train/')
dev_path = os.path.join(task, audio_base, 'dev/')
test_path = os.path.join(task, audio_base, 'test/')

# save_path = 'data/labels/{}/'.format(task)
scp_file = "/srv/data/egasj/corpora/labels/{}/wav.txt".format(task)

# Getting data info ready
# crate_csv_bea_from_scp(scp_file=scp_file, out_path=save_path, train_split_data=0.85)

# Loading the dataset into 'load_datasets' class
data_files = {
    'train': labels,
}

complete_dataset = load_dataset('csv', data_files=data_files, delimiter=',', cache_dir=config['hf_cache_dir'],
                          download_mode=DownloadMode['REUSE_DATASET_IF_EXISTS'])
train_set = complete_dataset['train']

# Getting unique labels
label_list = train_set.unique('label')
label_list.sort()
num_labels = len(label_list)

# Configurations
model_name = config['pretrained_model_details']['checkpoint_path']

config = AutoConfig.from_pretrained(
    model_name_or_path,
    num_labels=num_labels,
    label2id={label: i for i, label in enumerate(label_list)},
    id2label={i: label for i, label in enumerate(label_list)},
    finetuning_task="wav2vec2_clf",
)

setattr(config, 'pooling_mode', 'mean')

processor = Wav2Vec2Processor.from_pretrained(model_name)
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

# Setting-up the trainer
data_collator = utils_fine_tune.DataCollatorCTCWithPadding(processor=processor, padding=True)

# Define evaluation metrics
is_regression = False

# Load pre-trained model to fine-tune
model = Wav2Vec2ForSpeechClassification.from_pretrained(
    model_name_or_path,
    config=config,
)

# Freeze CNN blocks
model.freeze_feature_extractor()

# Define trainers and train model

epochs_list = [1.0, 3.0, 5.0]
for num_train_epochs in epochs_list:
    out_dir = '/srv/data/egasj/code/wav2vec2_patho_deep4/runs/{0}_{1}_{2}'.format(task, num_train_epochs, lang)
    training_args = TrainingArguments(
        output_dir=out_dir,
        # output_dir="/content/gdrive/MyDrive/wav2vec2-xlsr-greek-speech-emotion-recognition"
        per_device_train_batch_size=8,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=1,
        evaluation_strategy="steps",
        num_train_epochs=num_train_epochs,
        fp16=False,
        save_steps=10,
        eval_steps=10,
        logging_steps=10,
        learning_rate=5e-3,
        save_total_limit=2,
        # use_ipex=True
    )

    # trainer = utils_fine_tune.CTCTrainer(
    #     model=model,
    #     data_collator=data_collator,
    #     args=training_args,
    #     compute_metrics=utils.compute_metrics,
    #     train_dataset=train_dataset,
    #     eval_dataset=eval_dataset,
    #     tokenizer=processor.feature_extractor,
    # )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=utils.compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor.feature_extractor,
    )

    trainer.train()
    trainer.save_model(out_dir)


if __name__ == '__main__':
    train()
