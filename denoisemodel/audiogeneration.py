import librosa
import soundfile as sf
import numpy as np

def convert_spectrograms_to_audio(epoch, spectrograms, max_val, min_val, hop_length, frame_size, sr):

    num_to_generate = min(10, spectrograms.shape[0])

    for i in range(num_to_generate):
        log_spectrogram = spectrograms[i, :, :, 0]  # Get rid of channel axis
        denormalised_log_spectrogram = log_spectrogram * (max_val - min_val) + min_val  # Denormalise
        # Add a 0 valued nyquist bin
        denormalised_log_spectrogram = np.vstack([denormalised_log_spectrogram,
                                                  np.zeros(denormalised_log_spectrogram.shape[1])])
        spec = librosa.db_to_amplitude(denormalised_log_spectrogram) # Convert back to linear from Db
        signal = librosa.griffinlim(spec,
                                    n_iter=32,
                                    hop_length=hop_length,  # Perform griffin Lim
                                    win_length=frame_size)
        audio = signal.astype(np.float32)  # Ensure correct data type
        sf.write(f'GENERATEDAUDIO/generated_AUDIO_{i}_epoch_{epoch}.wav', audio, sr)


def convert_spectrograms_to_audio2(epoch, spectrograms, max_val, min_val, hop_length, frame_size, sr):

    num_to_generate = min(10, spectrograms.shape[0])

    for i in range(num_to_generate):
        log_spectrogram = spectrograms[i, :, :, 0]  # Get rid of channel axis
        denormalised_log_spectrogram = log_spectrogram * (max_val - min_val) + min_val  # Denormalise
        # Add a 0 valued nyquist bin
        denormalised_log_spectrogram = np.vstack([denormalised_log_spectrogram,
                                                  np.zeros(denormalised_log_spectrogram.shape[1])])
        spec = librosa.db_to_amplitude(denormalised_log_spectrogram) # Convert back to linear from Db
        signal = librosa.griffinlim(spec,
                                    n_iter=32,
                                    hop_length=hop_length,  # Perform griffin Lim
                                    win_length=frame_size)
        audio = signal.astype(np.float32)  # Ensure correct data type
        sf.write(f'GENERATEDAUDIOTEST/generated_AUDIO_CLEAN_{i}_epoch_{epoch}.wav', audio, sr)

def convert_spectrograms_to_audio3(epoch, spectrograms, max_val, min_val, hop_length, frame_size, sr):

    num_to_generate = min(10, spectrograms.shape[0])

    for i in range(num_to_generate):
        log_spectrogram = spectrograms[i, :, :, 0]  # Get rid of channel axis
        denormalised_log_spectrogram = log_spectrogram * (max_val - min_val) + min_val  # Denormalise
        # Add a 0 valued nyquist bin
        denormalised_log_spectrogram = np.vstack([denormalised_log_spectrogram,
                                                  np.zeros(denormalised_log_spectrogram.shape[1])])
        spec = librosa.db_to_amplitude(denormalised_log_spectrogram) # Convert back to linear from Db
        signal = librosa.griffinlim(spec,
                                    n_iter=32,
                                    hop_length=hop_length,  # Perform griffin Lim
                                    win_length=frame_size)
        audio = signal.astype(np.float32)  # Ensure correct data type
        sf.write(f'GENERATEDAUDIOTEST/generated_AUDIO_NOISY_{i}_epoch_{epoch}.wav', audio, sr)













def convert_spectrograms_to_audio5(epoch, spectrograms, max_val, min_val, hop_length, sr):

    num_to_generate = min(10, spectrograms.shape[0])

    for i in range(num_to_generate):
        log_spectrogram = spectrograms[i, :, :, 0]
        denormalised_log_spectrogram = log_spectrogram * (max_val - min_val) + min_val
        spec = librosa.db_to_amplitude(denormalised_log_spectrogram)
        signal = librosa.istft(spec, hop_length=hop_length)
        audio = signal.astype(np.float32)  # Ensure correct data type
        sf.write(f'GENERATEDAUDIOTEST/generated_AUDIO_{i}_epoch_{epoch}.wav', audio, sr)
