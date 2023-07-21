import os
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm


# Function to split audio file into N-second chunks
def split_audio(input_path, output_dir, filename, label, N=3):
    audio, sr = librosa.load(input_path, sr=None)
    duration = librosa.get_duration(y=audio, sr=sr)
    chunk_length = N
    num_chunks = int(duration / chunk_length)

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

# input_folder = "/path/to/your/input/folder"
N_seconds = 2
output_folder = "/srv/data/egasj/corpora/eating-wav-all/{}secs_chunked/".format(N_seconds)
os.makedirs(output_folder, exist_ok=True)
list_set = ["train", "dev", "test"]

new_csv_rows = []
for _set in list_set:
    csv_path = "../data/eating/{}.csv".format(_set)
    metadata_df = pd.read_csv(csv_path)
    for index, row in (pbar := (tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Splitting audio files", position=0))):
        filename = row["filename"]
        label = row["label"]
        input_path = row["path"]
        split_audio(input_path, output_folder, filename, label, N=N_seconds)

    new_metadata_df = pd.DataFrame(new_csv_rows)

    new_csv_path = "../data/eating/{}_chunked_{}secs.csv".format(_set, N_seconds)
    new_metadata_df.to_csv(new_csv_path, index=False)
