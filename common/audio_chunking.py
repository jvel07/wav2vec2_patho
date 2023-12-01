import os
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm

from common import utils


# Function to split audio file into N-second chunks
def split_audio(input_path, output_dir, label, age, sex, smoke, N):
    # print("\n Current audio ", input_path)
    audio, sr = librosa.load(input_path, sr=None)
    duration = librosa.get_duration(y=audio, sr=sr)
    chunk_length = N
    num_chunks = int(duration / chunk_length)

    new_csv_rows = []
    filename = os.path.basename(input_path).split('.')[0]

    for i in range(num_chunks):
        chunk_start = int(i * chunk_length * sr)
        chunk_end = int((i + 1) * chunk_length * sr)
        chunk_audio = audio[chunk_start:chunk_end]
        chunk_filename = f"{filename}_chunk{i + 1}.wav"
        chunk_filepath = os.path.join(output_dir, chunk_filename)
        sf.write(chunk_filepath, chunk_audio, sr)
        # print("wrote chunk: ", chunk_filepath)

        new_row = {
            "filename": chunk_filename,
            "label": label,
            "path": chunk_filepath,
            "length_in_frames": len(chunk_audio),
            "age": age,
            "sex": sex,
            "smoke": smoke,
        }
        new_csv_rows.append(new_row)

    # Handle the remaining audio
    remaining_audio = audio[num_chunks * chunk_length * sr:]
    config = utils.load_config('../config/config_depression_chunked.yml')  # loading configuration
    seg_len = int(config['segment'] * config['sample_rate'])
    # print(len(remaining_audio), seg_len)
    if len(remaining_audio) >= seg_len:
        # print("Remaining audio {} for file {}: ".format(len(remaining_audio), filename))
        r_chunk_filename = f"{filename}_chunk{num_chunks + 1}.wav"
        r_chunk_filepath = os.path.join(output_dir, r_chunk_filename)
        sf.write(r_chunk_filepath, remaining_audio, sr)
        # print("wrote remaining audio: ", r_chunk_filepath)

        new_row = {
            "filename": r_chunk_filename,
            "label": label,
            "path": r_chunk_filepath,
            "length_in_frames": len(remaining_audio),
            "age": age,
            "sex": sex,
            "smoke": smoke,
        }
        new_csv_rows.append(new_row)

    return new_csv_rows


    # if 0 < len(remaining_audio) < seg_len:
    # if len(remaining_audio) > 0 and  seg_len >= len(remaining_audio):
    #     remaining_chunk_filename = f"{filename}_chunk{num_chunks + 1}.wav"
    #     remaining_chunk_filepath = os.path.join(output_dir, remaining_chunk_filename)
    #     sf.write(remaining_chunk_filepath, remaining_audio, sr)
    #
    #     new_row = {
    #         "filename": remaining_chunk_filename,
    #         "label": label,
    #         "path": remaining_chunk_filepath,
    #         "length_in_frames": len(remaining_audio),
    #         "age": age,
    #         "sex": sex,
    #         "smoke": smoke,
    #     }
    #     new_csv_rows.append(new_row)


if __name__ == "__main__":

    # input_folder = "/path/to/your/input/folder"
    N_seconds = 4
    # output_folder = "/srv/data/egasj/corpora/eating-wav-all/{}secs_chunked_2/".format(N_seconds)
    # output_folder = "/srv/data/egasj/corpora/DEPISDA_16k_{}secs_chunked/".format(N_seconds)
    output_folder = "/media/jvel/data/audio/DEPISDA_16k_{}secs_chunked_depured/".format(N_seconds)
    os.makedirs(output_folder, exist_ok=True)
    list_set = ['1']
    # list_set = ["train"]

    for _set in list_set:
        rows_audio_list = []
        csv_path = "../metadata/depression/metadata_depisda_local.csv"
        metadata_df = pd.read_csv(csv_path)
        for index, row in (pbar := (tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Splitting audio files", position=0))):
        # for index, row in metadata_df.iterrows():
            label = row["label"]
            input_path = row["path"]
            sex = row['Sex']
            age = row['Age']
            smoke = row['Smoke']
            rows_audio = split_audio(input_path, output_folder, label, age, sex, smoke, N=N_seconds)
            # pbar.set_description("Splitting audio {}".format(input_path))
            rows_audio_list.append(rows_audio)

        flat_list = [item for sublist in rows_audio_list for item in sublist]
        new_metadata_df = pd.DataFrame(flat_list)

        median_bdi = new_metadata_df['label'].median()
        new_metadata_df['label'].fillna(median_bdi, inplace=True)
        median_age = new_metadata_df['age'].median()
        new_metadata_df['age'].fillna(median_age, inplace=True)
        new_metadata_df['smoke'].fillna(method='ffill', inplace=True)

        # some DE BDI values are below 13.5, which is not correct, change them to the median of DE
        median_DE_label = new_metadata_df.loc[new_metadata_df['filename'].str.contains('DE'), 'label'].median()
        condition = (new_metadata_df['label'] <= 13.5) & (new_metadata_df['filename'].str.contains('DE'))
        new_metadata_df.loc[condition, 'label'] = median_DE_label

        new_metadata_df['label'] = new_metadata_df['label'].astype('int')
        new_metadata_df['age'] = new_metadata_df['age'].astype('int')

        # print(rows_audio_list)

        new_csv_path = "../metadata/depression/depured_complete_depisda16k_chunked_{}secs.csv".format(N_seconds)
        new_metadata_df.to_csv(new_csv_path, index=False)
