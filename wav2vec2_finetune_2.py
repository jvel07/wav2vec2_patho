from datasets import load_dataset, Dataset, load_metric, Audio, concatenate_datasets
import torch
from transformers import pipeline, AutoFeatureExtractor, AutoTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, \
    AutoModelForAudioClassification, TrainingArguments, Trainer, AutoProcessor
import soundfile as sf
import librosa
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score


# https://github.com/aalto-speech/ComParE2022
def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    if "-superb-" in model_link:
        predictions = np.argmax(eval_pred.predictions[0], axis=1)
    else:
        predictions = np.argmax(eval_pred.predictions, axis=1)
    recall = recall_metric.compute(predictions=predictions, references=eval_pred.label_ids, average="macro")
    # f1 = f1_metric.compute(predictions=predictions, references=eval_pred.label_ids, average="macro")
    # return {"f1": f1, "spearmanr": spearmanr}
    return recall


def prepare_example(example):
    if '.FI0' in example["file"]:
        example["speech"], example["sampling_rate"] = sf.read(example["file"], channels=1, samplerate=16000,
                                                              format='RAW', subtype='PCM_16')
    else:
        example["audio"], example["sampling_rate"] = librosa.load(example["file"], sr=16000)
    example["duration_in_seconds"] = len(example["audio"]) / 16000
    return example


def preprocess_function(examples):
    audio_arrays = examples["audio"]
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=feature_extractor.sampling_rate
    )
    return inputs


def map_to_array(example):
    speech, _ = librosa.load(example["file"], sr=16000, mono=True)
    example["speech"] = speech
    return example


def map_to_pred(batch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_values = processor(batch["speech"], sampling_rate=16000, return_tensors="pt", padding="longest").input_values
    with torch.no_grad():
        logits = model_fi(input_values.to(device)).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    batch["probs"] = torch.softmax(logits, dim=-1)
    batch['predictions'] = predicted_ids
    return batch


freeze_feature_extractor = False
freeze_transformer = False
task = "Stuttering"
# task = "Vocalisation"
TRAIN_FINAL = False

# model_checkpoint = "facebook/wav2vec2-base-10k-voxpopuli-ft-de"
# model_checkpoint = "facebook/wav2vec2-large-west_germanic-voxpopuli-v2"
model_checkpoint = "aware-ai/wav2vec2-base-german"
# model_checkpoint = "aware-ai/wav2vec2-xls-r-300m-german"
# model_checkpoint = "aware-ai/wav2vec2-xls-r-1b-5gram-german"
# model_checkpoint = "jonatasgrosman/wav2vec2-large-xlsr-53-german"
# model_checkpoint = "facebook/wav2vec2-xls-r-2b"
# model_checkpoint = "superb/wav2vec2-large-superb-er"
# model_checkpoint = "Aniemore/wav2vec2-xlsr-53-russian-emotion-recognition"
# model_checkpoint = "harshit345/xlsr-wav2vec-speech-emotion-recognition"
# model_checkpoint = "superb/hubert-large-superb-er"

batch_size = 8
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
f1_metric = load_metric("f1")
recall_metric = load_metric("recall")
target_sr = 16000
model_name = model_checkpoint.split("/")[-1]

if task == "Vocalisation":
    categories = {'surprise': 0, 'fear': 1, 'anger': 2, 'pleasure': 3, 'pain': 4, 'achievement': 5, '?': -1}
    path_to_recs = "/teamwork/t40511_asr/c/ComParE_2022/Vocalisation/dist/wav/"
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint,
                                                             cache_dir="/scratch/elec/puhe/p/getmany1/cache")

    try:
        processor = AutoProcessor.from_pretrained(model_checkpoint, cache_dir="/scratch/elec/puhe/p/getmany1/cache")
    except (OSError, ValueError) as e:
        print(f"No tokenizer found for {model_checkpoint}")
        processor = None

    label_base = "/teamwork/t40511_asr/c/ComParE_2022/Vocalisation/dist/lab"
    labels = pd.concat([pd.read_csv(f"{label_base}/{partition}.csv") for partition in ["train", "devel", "test"]])
    train_ids = pd.read_csv(f"{label_base}/train.csv").filename
    dev_ids = pd.read_csv(f"{label_base}/devel.csv").filename
    test_ids = pd.read_csv(f"{label_base}/test.csv").filename

    my_dict_train = {'file': [path_to_recs + item for item in train_ids],
                     'label': [categories[labels[labels.filename == item].label.item()] for item in train_ids]}
    my_dict_dev = {'file': [path_to_recs + item for item in dev_ids],
                   'label': [categories[labels[labels.filename == item].label.item()] for item in dev_ids]}

    if TRAIN_FINAL:
        my_dict_train['file'] += my_dict_dev['file']
        my_dict_train['label'] += my_dict_dev['label']

    my_dict_test = {'file': [path_to_recs + item for item in test_ids],
                    'label': [categories[labels[labels.filename == item].label.item()] for item in test_ids]}

    train_dataset = Dataset.from_dict(my_dict_train)
    dev_dataset = Dataset.from_dict(my_dict_dev)
    test_dataset = Dataset.from_dict(my_dict_test)

    train_dataset = train_dataset.map(prepare_example, remove_columns=['file'])
    dev_dataset = dev_dataset.map(prepare_example, remove_columns=['file'])
    test_dataset = test_dataset.map(prepare_example, remove_columns=['file'])

    train_dataset = train_dataset.map(preprocess_function, batched=True, batch_size=1)
    test_dataset = test_dataset.map(preprocess_function, batched=True, batch_size=1)
    dev_dataset = dev_dataset.map(preprocess_function, batched=True, batch_size=1)

    model = AutoModelForAudioClassification.from_pretrained(
        model_checkpoint,
        trust_remote_code=True,
        cache_dir="/scratch/elec/puhe/p/getmany1/cache"
    )
    model.config.id2label = None
    model.config.label2id = None
    model.config.num_labels = 2 # 37+2/length wav ==> target; 31/len ==> target
    model.classifier = torch.nn.Linear(in_features=256, out_features=2, bias=True)

elif task == "Stuttering":
    path_to_train = '/teamwork/t40511_asr/c/ComParE_2022/Stuttering/compare22-KSF/lab/train.csv'
    path_to_dev = '/teamwork/t40511_asr/c/ComParE_2022/Stuttering/compare22-KSF/lab/devel.csv'
    path_to_test = '/teamwork/t40511_asr/c/ComParE_2022/Stuttering/compare22-KSF/lab/test.csv'

    df_train = pd.read_csv(path_to_train, encoding='utf-8')
    df_dev = pd.read_csv(path_to_dev, encoding='utf-8')
    df_test = pd.read_csv(path_to_test, encoding='utf-8')

    labels = sorted(df_train.label.unique())
    label_dict = {labels[i]: [j for j in range(len(labels))][i] for i in range(len(labels))}
    df_train = df_train.replace({"label": label_dict})
    df_dev = df_dev.replace({"label": label_dict})

    df_train['file'] = '/teamwork/t40511_asr/c/ComParE_2022/Stuttering/compare22-KSF/wav/' + df_train['filename']
    df_dev['file'] = '/teamwork/t40511_asr/c/ComParE_2022/Stuttering/compare22-KSF/wav/' + df_dev['filename']
    df_test['file'] = '/teamwork/t40511_asr/c/ComParE_2022/Stuttering/compare22-KSF/wav/' + df_test['filename']

    train_dataset = Dataset.from_pandas(df_train)
    dev_dataset = Dataset.from_pandas(df_dev)
    test_dataset = Dataset.from_pandas(df_test)

    if TRAIN_FINAL:
        train_dataset = concatenate_datasets([train_dataset, dev_dataset])

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_checkpoint)
    processor = Wav2Vec2Processor.from_pretrained(model_checkpoint)

    train_dataset = train_dataset.map(prepare_example)
    dev_dataset = dev_dataset.map(prepare_example)
    test_dataset = test_dataset.map(prepare_example)

    train_dataset = train_dataset.map(preprocess_function, remove_columns=['filename', 'file'], batched=True,
                                      batch_size=1)
    test_dataset = test_dataset.map(preprocess_function, remove_columns=['filename', 'file'], batched=True,
                                    batch_size=1)
    dev_dataset = dev_dataset.map(preprocess_function, remove_columns=['filename', 'file'], batched=True, batch_size=1)

    model = AutoModelForAudioClassification.from_pretrained(
        model_checkpoint,
        trust_remote_code=True,
        cache_dir="/scratch/elec/puhe/p/getmany1/cache",
        num_labels=len(df_train['label'].unique())
    )

if freeze_feature_extractor:
    model.freeze_feature_extractor()
if freeze_transformer:
    model.freeze_transformer()

args = TrainingArguments(
    "/scratch/elec/puhe/p/getmany1/wav2vec2_compare_stuttering_base_unfrozen_cnn",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=batch_size,
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
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics,
)
trainer.train()

predictions = trainer.predict(dev_dataset)
print(compute_metrics(predictions))
