import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,3,4,5"

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
    speech_list = [speech_file_to_array_fn(path) for path in examples["filename_full"]]
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
                loss = loss_fct(logits.view(-1, self.num_labels), labels.unsqueeze(-1))
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

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


if __name__ == "__main__":
    with open("config/params.yaml") as f:
        _params = yaml.safe_load(f)
        params = _params["wav2vec"]
        target = _params["target"]

    label_dir = "data/eating"
    result_dir = "results/wav2vec"
    logging_dir = "tb_logs/wav2vec"

    model_checkpoint = params["model"]

    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_sr = 16000
    model_name = model_checkpoint.split("/")[-1]

    all_files = list(glob("./data/wav/*.wav", recursive=True))
    label_list = [0]

    makedirs(result_dir, exist_ok=True)
    makedirs(logging_dir, exist_ok=True)

    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")

    df_train = pd.read_csv(f"{label_dir}/train.csv", encoding="utf-8")
    df_dev = pd.read_csv(f"{label_dir}/dev.csv", encoding="utf-8")
    df_test = pd.read_csv(f"{label_dir}/test.csv", encoding="utf-8")

    # df_train["filename_full"] = "/srv/data/egasj/corpora/eating-wav-all/train/" + df_train["filename"] + ".wav"
    df_train["filename_full"] = "/srv/data/egasj/corpora/eating-wav-all/" + df_train["filename"] + ".wav"
    # df_train = df_train[df_train["filename_full"].isin(all_files)]

    df_dev["filename_full"] = "/srv/data/egasj/corpora/eating-wav-all/" + df_dev["filename"] + ".wav"
    # df_dev["filename_full"] = "/srv/data/egasj/corpora/eating-wav-all/dev/" + df_dev["filename"] + ".wav"
    # df_dev = df_dev[df_dev["filename_full"].isin(all_files)]

    # df_test["filename_full"] = "./data/wav/" + df_test["filename"]
    df_test["filename_full"] = "/srv/data/egasj/corpora/eating-wav-all/" + df_test["filename"] + ".wav"
    # df_test = df_test[df_test["filename_full"].isin(all_files)]

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
        num_labels=len(label_encoder.classes_),
        problem_type=None,
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
    print("processed train")
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
        metric_for_best_model="recall",
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
        compute_metrics=compute_metrics,

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
    print("DEV -->", compute_metrics(dev_predictions))

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
    print("TEST-->",compute_metrics(test_predictions))

