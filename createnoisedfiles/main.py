import os
import numpy as np
import soundfile as sf
import librosa

# Parameters and Paths
input_folder = "/Users/isaac/Documents/DATASETS/DENOISE/CLEANSAMPLES"  # Path to the folder with original audio files
output_folder = "/Users/isaac/Documents/DATASETS/DENOISE/NOISYSAMPLES"  # Path to the folder where new files will be saved
noise_level = 0.009  # The magnitude of the white noise to be added

# Ensure the output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop through all files in the input folder
for file_name in os.listdir(input_folder):
    if file_name.endswith('.wav'):
        input_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, file_name)

        # Load the audio file
        y, sr = librosa.load(input_path, sr=None)

        # Generate white noise
        noise = np.random.normal(0, noise_level, len(y))

        # Add the white noise to the original audio
        y_noised = y + noise

        # Ensure the noised signal is in [-1, 1]
        y_noised = np.clip(y_noised, -1, 1)

        # Write the noised audio to the output path
        sf.write(output_path, y_noised, sr)

        print(f"Processed {file_name}")

print("Processing completed!")
