import os
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm


# Function to split audio file into N-second chunks
def split_audio(input_path, output_dir, label, N=3):
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

        new_row = {
            "filename": chunk_filename,
            "label": label,
            "path": chunk_filepath,
            "length_in_frames": len(chunk_audio),
        }
        new_csv_rows.append(new_row)

    # Handle the remaining audio
    remaining_audio = audio[num_chunks * chunk_length * sr:]
    if len(remaining_audio) > 0:
        remaining_chunk_filename = f"{filename}_chunk{num_chunks + 1}.wav"
        remaining_chunk_filepath = os.path.join(output_dir, remaining_chunk_filename)
        sf.write(remaining_chunk_filepath, remaining_audio, sr)

        new_row = {
            "filename": remaining_chunk_filename,
            "label": label,
            "path": remaining_chunk_filepath,
            "length_in_frames": len(remaining_audio),
        }
        new_csv_rows.append(new_row)

    return new_csv_rows


if __name__ == "__main__":

    # input_folder = "/path/to/your/input/folder"
    N_seconds = 4
    # output_folder = "/srv/data/egasj/corpora/eating-wav-all/{}secs_chunked_2/".format(N_seconds)
    output_folder = "/srv/data/egasj/corpora/DEPISDA_16k_{}secs_chunked/".format(N_seconds)
    os.makedirs(output_folder, exist_ok=True)
    list_set = ["DE"]
    # list_set = ["train"]

    for _set in list_set:
        rows_audio_list = []
        csv_path = "../data/depression/metadata_depisda.csv"
        metadata_df = pd.read_csv(csv_path)
        for index, row in (pbar := (tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Splitting audio files", position=0))):
            label = row["label"]
            input_path = row["path"]
            rows_audio = split_audio(input_path, output_folder, label, N=N_seconds)
            pbar.set_description("Splitting audio {}".format(input_path))
            rows_audio_list.append(rows_audio)

        flat_list = [item for sublist in rows_audio_list for item in sublist]
        new_metadata_df = pd.DataFrame(flat_list)
        # print(rows_audio_list)

        new_csv_path = "../data/depression/chunked_{}secs.csv".format(N_seconds)
        new_metadata_df.to_csv(new_csv_path, index=False)
