import os

import math
from sklearn.model_selection import train_test_split

from common import split_depisda_corpus, get_dataset_partitions_pd

# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,7"

from dataclasses import dataclass
from glob import glob

from os import listdir, makedirs
from shutil import rmtree
from os.path import isfile, join
from typing import Dict, List, Optional, Tuple, Union
from sklearn.preprocessing import LabelEncoder
import sys
import librosa
import numpy as np
import pandas as pd
import datetime
import socket
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio
import yaml
import evaluate
import torch.nn.functional as F
from datasets import Audio, Dataset, concatenate_datasets, load_dataset
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    pipeline,
)
from transformers.file_utils import ModelOutput
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(
            self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_features = [
            {"input_values": feature["input_values"]} for feature in features
        ]

        # d_type = torch.long if isinstance(label_features[0], int) else torch.float
        d_type = torch.long

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            truncation=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        if "labels" in features[0]:
            label_features = [feature["labels"] for feature in features]
            batch["labels"] = torch.tensor(label_features, dtype=d_type)

        return batch


def speech_file_to_array_fn(path):
    speech_array, sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
    speech = resampler(speech_array).squeeze().numpy()
    return speech


def preprocess_function(examples):
    speech_list = [speech_file_to_array_fn(path) for path in examples["path"]]
    result = processor(speech_list, sampling_rate=target_sampling_rate)

    if "label" in examples:
        target_list = [
            int(label) for label in examples["label"]
        ]  # Do any preprocessing on your float/integer data
        result["labels"] = list(target_list)

    return result


@dataclass
class SpeechClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class Wav2Vec2ClassificationHead(nn.Module):
    """Head for wav2vec classification task."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class Wav2Vec2ForSpeechClassification(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_mode = config.pooling_mode
        self.config = config

        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = Wav2Vec2ClassificationHead(config)

        self.init_weights()

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def merged_strategy(self, hidden_states, mode="mean"):
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif mode == "max":
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']"
            )

        return outputs

    def forward(
            self,
            input_values,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            labels=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        hidden_states = self.merged_strategy(hidden_states, mode=self.pooling_mode)
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                        labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = F.mse_loss
                # print("logits:", logits.view(-1, self.num_labels).shape, "values:", logits.view(-1, self.num_labels), "dtype:", logits.view(-1, self.num_labels).dtype)
                # print("labels:", labels.unsqueeze(-1).shape, "values:", labels.unsqueeze(-1),
                #       "dtype:", labels.unsqueeze(-1).dtype)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.unsqueeze(-1).float())
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        # print("Problem type:", self.config.problem_type)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SpeechClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def compute_metrics(eval_pred):
    """Computes spearman and pearson on predictions"""
    recall = recall_metric.compute(
        predictions=np.argmax(eval_pred.predictions.squeeze(), axis=-1),
        references=eval_pred.label_ids,
        average="macro",
    )
    f1 = f1_metric.compute(
        predictions=np.argmax(eval_pred.predictions.squeeze(), axis=-1),
        references=eval_pred.label_ids,
        average="macro",
    )
    # f1 = f1_metric.compute(predictions=predictions, references=eval_pred.label_ids, average="macro")
    return {**recall, **f1}


def compute_regression_metrics(eval_pred):
    """Computes spearman and pearson on predictions"""
    preds = eval_pred.predictions[0] if isinstance(eval_pred.predictions, tuple) else eval_pred.predictions
    preds = np.squeeze(preds)

    # pearsons = pearsons_metric.compute(
    #     predictions=preds,
    #     references=eval_pred.label_ids,
    # )
    # mse = mse_metric.compute(
    #     predictions=preds,
    #     references=eval_pred.label_ids,
    # )
    # # f1 = f1_metric.compute(predictions=predictions, references=eval_pred.label_ids, average="macro")
    # # return {**recall, **f1}
    # return {**pearsons, **mse}

    return {"mse": ((preds - eval_pred.label_ids) ** 2).mean().item()}

if __name__ == "__main__":
    with open("config/params.yaml") as f:
        _params = yaml.safe_load(f)
        params = _params["wav2vec"]
        target = _params["target"]
    batch_size = 2

    model_used = _params['wav2vec']["model"].split("/")[-1]
    label_dir = "metadata/depression"
    # result_dir = "results/{}".format(model_used)
    result_dir = "results/depression/chunked4secs_{}_{}batch".format(model_used, str(batch_size))
    logging_dir = "tb_logs/{}".format(model_used)

    model_checkpoint = params["model"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_sr = 16000
    model_name = model_checkpoint.split("/")[-1]

    all_files = list(glob("./data/wav/*.wav", recursive=True))
    label_list = [0]

    makedirs(result_dir, exist_ok=True)
    makedirs(logging_dir, exist_ok=True)

    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")

    pearsons_metric = evaluate.load("pearsonr")
    mse_metric = evaluate.load("mse")

    data = pd.read_csv(f"{label_dir}/chunked_4secs.csv", encoding="utf-8")
    random_numbers = np.random.randint(1, 18, size=data['label'].isna().sum()) # to fill HC NaN values
    data.loc[data['label'].isna(), 'label'] = random_numbers
    data['label'] = data['label'].astype(int)

    # strat group by filename
    df_train, df_dev, df_test = split_depisda_corpus(data)

    # Separate features (X) and labels (y)
    # X = data.drop(columns=['label'])
    # y = data['label']
    #
    # # Perform a stratified split
    # X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
    # X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
    #
    # # Create DataFrames for each split
    # df_train = pd.concat([X_train, y_train], axis=1)
    # df_dev = pd.concat([X_dev, y_dev], axis=1)
    # df_test = pd.concat([X_test, y_test], axis=1)

    # df_train = pd.read_csv(f"{label_dir}/train_chunked_4secs.csv", encoding="utf-8")
    # df_dev = pd.read_csv(f"{label_dir}/dev_chunked_4secs.csv", encoding="utf-8")
    # df_test = pd.read_csv(f"{label_dir}/test.csv", encoding="utf-8")

    # # df_train["filename_full"] = "/srv/data/egasj/corpora/eating-wav-all/train/" + df_train["filename"] + ".wav"
    # df_train["filename_full"] = "/srv/data/egasj/corpora/eating-wav-all/" + df_train["filename"] + ".wav"
    # # df_train = df_train[df_train["filename_full"].isin(all_files)]
    #
    # df_dev["filename_full"] = "/srv/data/egasj/corpora/eating-wav-all/" + df_dev["filename"] + ".wav"
    # # df_dev["filename_full"] = "/srv/data/egasj/corpora/eating-wav-all/dev/" + df_dev["filename"] + ".wav"
    # # df_dev = df_dev[df_dev["filename_full"].isin(all_files)]
    #
    # # df_test["filename_full"] = "./data/wav/" + df_test["filename"]
    # df_test["filename_full"] = "/srv/data/egasj/corpora/eating-wav-all/" + df_test["filename"] + ".wav"
    # # df_test = df_test[df_test["filename_full"].isin(all_files)]

    label_encoder = LabelEncoder()
    df_train["label"] = label_encoder.fit_transform(df_train['label'])
    df_dev["label"] = label_encoder.transform(df_dev['label'])
    # if len(set(df_test[target])) > 1:
    df_test["label"] = label_encoder.transform(df_test['label'])

    train_dataset = Dataset.from_pandas(df_train)
    dev_dataset = Dataset.from_pandas(df_dev)
    test_dataset = Dataset.from_pandas(df_test)

    config = AutoConfig.from_pretrained(
        model_checkpoint,
        # num_labels=len(label_encoder.classes_),
        num_labels=1,  # for regression
        problem_type='regression',
        # problem_type=None,
    )
    setattr(config, "pooling_mode", params["pooling"])

    processor = Wav2Vec2Processor.from_pretrained(model_checkpoint)
    target_sampling_rate = processor.feature_extractor.sampling_rate
    data_collator = DataCollatorCTCWithPadding(
        processor=processor, padding=True, max_length=10 * target_sampling_rate
    )

    train_dataset = train_dataset.map(
        preprocess_function, batch_size=100, batched=True, num_proc=4
    )
    print("processed train with size", len(train_dataset))
    dev_dataset = dev_dataset.map(
        preprocess_function, batch_size=100, batched=True, num_proc=4
    )
    print("processed devel")
    test_dataset = test_dataset.map(
        preprocess_function, batch_size=32, batched=True, num_proc=4
    )
    print("processed test")

    model = Wav2Vec2ForSpeechClassification.from_pretrained(
        model_checkpoint,
        config=config,
    )
    if params["freezeExtractor"]:
        model.freeze_feature_extractor()
    if params["freezeTransformer"]:
        model.freeze_transformer()

    args = TrainingArguments(
        result_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-4,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=params["epochs"],
        warmup_ratio=0.1,
        logging_steps=10,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="pearsonr",
        push_to_hub=False,
        gradient_checkpointing=True,
        save_total_limit=2,
        disable_tqdm=False,
        logging_dir=f"{logging_dir}/{datetime.datetime.now().strftime('%h%d_%Y-%H-%M-%S')}_{socket.gethostname()}",
    )
    print(args.device)

    trainer = Trainer(
        model=model,
        args=args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=processor.feature_extractor,
        compute_metrics=compute_regression_metrics,

    )
    trainer.train()

    dev_predictions = trainer.predict(dev_dataset)
    dev_df = pd.DataFrame(
        {
            "filename": dev_dataset["filename"],
            "prediction": label_encoder.inverse_transform(
                np.argmax(dev_predictions.predictions.squeeze(), axis=-1)
            ),
            "true": label_encoder.inverse_transform(
                dev_predictions.label_ids.squeeze()
            ),
        }
    )
    dev_df.to_csv(join(result_dir, "predictions.devel.csv"), index=False)
    # print("DEV -->", compute_metrics(dev_predictions))
    print("DEV -->", compute_regression_metrics(dev_predictions))

    test_predictions = trainer.predict(test_dataset)
    test_df = pd.DataFrame(
        {
            "filename": test_dataset["filename"],
            "prediction": label_encoder.inverse_transform(
                np.argmax(test_predictions.predictions.squeeze(), axis=-1)
            ),
            "true": label_encoder.inverse_transform(
                test_predictions.label_ids.squeeze()
            )
            # if len(set(test_predictions.label_ids)) > 1
            # else ["?"] * test_predictions.predictions.shape[0],
        }
    )
    test_df.to_csv(join(result_dir, "predictions.test.csv"), index=False)
    print("TEST on chunks-->",compute_regression_metrics(test_predictions))

    # computing test scores on original --non-chunked-- wavs
    data_orig = pd.read_csv(f"{label_dir}/metadata_depisda.csv", encoding="utf-8")
    test_files_set = df_test['filename'].str[:6].unique()
    orig_test_df = data_orig[data_orig['name'].isin(test_files_set)].copy()
    orig_test_df['label'] = orig_test_df['label'].fillna(0)
    orig_test_df["label"] = label_encoder.transform(orig_test_df['label'])

    orig_test_dataset = Dataset.from_pandas(orig_test_df)
    orig_test_dataset = orig_test_dataset.map(
        preprocess_function, batch_size=32, batched=True, num_proc=4
    )

    # print("TEST-->",compute_metrics(test_predictions))
    print("TEST on originals-->",compute_regression_metrics(test_predictions))

# ON EATING CORPUS

# DEV --> {'recall': 0.7204081632653061, 'f1': 0.7241680721926215}
# TEST--> {'recall': 0.7473922902494331, 'f1': 0.743123588831164}

# DEV --> {'recall': 0.761, 'f1': 0.754} --> 2 secs chunks
# TEST--> {'recall': 0.793, 'f1': 0.787} --> 2 secs chunks

# DEV --> {'recall': 0.903, 'f1': 0.903} --> 16 batch 2 secs chunks + handle remaining chunks (2733 samples)
# TEST--> {'recall': 0.845, 'f1': 0.843} --> 16 batch 2 secs chunks

# DEV --> {'recall': 0.904., 'f1': 0.905} --> 16 batch 4 secs chunks + handle remaining chunks (1523 samples)
# TEST--> {'recall': 0.853, 'f1': 0.852} --> 16 batch 4 secs chunks

# DEV --> {'recall': 0.912., 'f1': 0.912} --> 8 batch 4 secs chunks + handle remaining chunks (1523 samples)
# TEST--> {'recall': 0.856, 'f1': 0.855} --> 8 batch 4 secs chunks

# DEV --> {'recall': 0.., 'f1': 0.} --> 6 batch 4 secs chunks + handle remaining chunks (1523 samples)
# TEST--> {'recall': 0., 'f1': 0.} --> 6 batch 4 secs chunks

# train on 6 batch 4 secs chunks and evaluate on non-chunked original eating test data
# TEST--> {'recall': 0.8665532879818595, 'f1': 0.865481555877025}



##### DEPRESSION chunked 4 secs
# DEV --> {'pearsonr': 0.701, 'mse': 83.164}
# TEST--> {'pearsonr': 0.746, 'mse': 87.226}
# TEST--> {'pearsonr': 0.756, 'mse': 88.912}
