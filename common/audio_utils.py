import glob

import librosa
import soundfile as sf
import os

from tqdm import tqdm


def resample_audio_files(audio_dir, new_dir):
    audio_list = glob.glob(os.path.join(audio_dir, '*.wav'))
    # pbar = tqdm(audio_list)
    # loop through all files in directory
    for file_name in (pbar := tqdm(audio_list, total=len(audio_list), desc="Resampling audio files", position=0)):
        # set path to audio file
        audio_path = os.path.join(audio_dir, file_name)
        pbar.set_description(f"Resampling {file_name}")
        # load audio file and get its sampling rate
        audio, sr = librosa.load(audio_path, sr=None)
        os.makedirs(new_dir, exist_ok=True)

        # if sampling rate is not 16000, resample audio
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000

            # set new path for resampled audio file
            audio_name = os.path.splitext(os.path.basename(audio_path))[0]
            new_audio_path = os.path.join(new_dir, f"{audio_name}.wav")

            # save resampled audio to new path with same name
            sf.write(new_audio_path, audio, sr)
        else:
            new_audio_path = os.path.join(new_dir, f"{audio_name}.wav")
            sf.write(new_audio_path, audio, sr)
            print(f"{file_name}: sampling rate is already 16000, no resampling needed")

    print(f"Resampled audio saved to {new_dir}")



