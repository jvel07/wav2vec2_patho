import numpy as np
import pandas as pd
import soundfile as sf
import torch
from sklearn.preprocessing import LabelEncoder
from torch import hub
from torch.utils.data import Dataset, DataLoader
import random as random
import os
import shutil
import zipfile


class ChunkedDataset(Dataset):
    """Dataset class for generating chunks for training.

    Args:
        csv_dir (str): The path to the metadata file.
        sample_rate (int) : The sample rate of the sources and mixtures.
        segment (int, optional) : The desired sources and mixtures length in s.

    """


    def __init__(
        self, csv_dir, sample_rate=16000, segment=3, return_id=False
    ):
        self.csv_dir = csv_dir
        self.return_id = return_id

        self.df = pd.read_csv(csv_dir, encoding="utf-8")

        # self.df["filename_full"] = "/srv/data/egasj/corpora/eating-wav-all/" + self.df["filename"] + ".wav"

        label_encoder = LabelEncoder()
        self.df["label"] = label_encoder.fit_transform(self.df['label'])

        self.segment = segment
        self.sample_rate = sample_rate

        # Get rid of the utterances too short
        if self.segment is not None:
            max_len = len(self.df)
            self.seg_len = int(self.segment * self.sample_rate)  # 2 * 8000 = 16000
            # con segment 2 y sample rate 12000 --> segment_length = 24000
            # con segment 1 y sample rate 12000 --> segment_length = 12000
            # con segment 0.5 y sample rate 12000 --> segment_length = 6000
            # Ignore the file shorter than the desired_length
            # data_set = data_set[utt_duration >= desired_length=2 * 12000 (24000)]
            self.df = self.df[self.df["length"] >= self.seg_len]
            print(
                f"Drop {max_len - len(self.df)} utterances from {max_len} "
                f"(shorter than {segment} seconds)"
            )
        else:
            self.seg_len = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get the row in dataframe
        row = self.df.iloc[idx]
        # Get mixture path
        utterance_path = row["path"]
        self.utterance_path = utterance_path
        # If there is a seg start point is set randomly
        if self.seg_len is not None:
            start = random.randint(0, row["length"] - self.seg_len)
            stop = start + self.seg_len
        else:
            start = 0
            stop = None

        print("start", start, "stop", stop, "mixture_path", utterance_path)

        # Read the mixture
        utterance, _ = sf.read(utterance_path, dtype="float32", start=start, stop=stop)
        print("mixture", utterance.shape)
        # Convert to torch tensor
        mixture = torch.from_numpy(utterance)
        # Stack sources
        # print("source from loader", sources.shape)

        if not self.return_id:
            return mixture
        # 5400-34479-0005_4973-24515-0007.wav
        # id1, id2 = mixture_path.split("/")[-1].split(".")[0].split("_")
        return mixture