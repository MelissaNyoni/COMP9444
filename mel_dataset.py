import os
import numpy as np
import librosa

input_dir = "/Users/oldsakura/Desktop/COMP9444/data/extracted/genres"
output_dir = "/Users/oldsakura/Desktop/COMP9444/mel"

genres = []
for folder in os.listdir(input_dir):
    if os.path.isdir(os.path.join(input_dir ,folder)) and not folder.startswith('.'):
        genres.append(folder)
genres.sort()


Failed_files = []
for genre in genres:
    print(f'\nProcessing {genre}')

    genre_path = os.path.join(input_dir, genre)

    output_genre_path = os.path.join(output_dir, genre)
    os.makedirs(output_genre_path, exist_ok = True)

    for wav_file in sorted(os.listdir(genre_path)):
        if wav_file.startswith('.'):
            continue
        wav_path = os.path.join(genre_path,wav_file)
        #print('Reading:',wav_path)
        try:
            y, sr = librosa.load(wav_path, sr = 22050, duration = 30)
            y = librosa.util.fix_length(y, size = 30 * sr)

            mel = librosa.feature.melspectrogram(y = y,sr = sr,n_mels = 128)
            mel_db = librosa.power_to_db(mel, ref = np.max)

            file_name = os.path.splitext(wav_file)[0]
            save_path = os.path.join(output_genre_path, file_name + '.npy')
            np.save(save_path, mel_db)

            print(f'Saved {wav_file}')
        except Exception as e:
            Failed_files.append(wav_file)
            print(f'Skip {wav_file}')
            continue
print(f"Failed_files{len(Failed_files)}")
print(Failed_files)