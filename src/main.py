import os
import numpy as np
from scipy.io import wavfile
import tensorflow as tf
import random
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from itertools import combinations, permutations
import soundfile as sf
import librosa
import pywt
from scipy.signal import windows
from scipy import optimize

from utils.Pipeline import (
    MP3DegradationPipeline
)
from utils.config import Config, RetrainConfig
from model import (
    WaveletUNet,
    gelu,
    DWTLayer,
    IDWTLayer,
    DownsamplingLayer,
    UpsamplingLayer,
    GatedSkipConnection,
)
from utils.data_preparation import process_audio_for_prediction_stereo, reconstruct_audio_from_clips
from pydub import AudioSegment

config = RetrainConfig()

def load_saved_model(model_dir, filename):
    """Load the model with multiple fallback options."""
    print(f"Attempting to load model from {model_dir}")

    custom_objects = {
        'WaveletUNet': WaveletUNet,
        'DWTLayer': DWTLayer,
        'IDWTLayer': IDWTLayer,
        'DownsamplingLayer': DownsamplingLayer,
        'UpsamplingLayer': UpsamplingLayer,
        'GatedSkipConnection': GatedSkipConnection,
        'gelu': gelu
    }

    loaded_model = tf.keras.models.load_model(os.path.join(model_dir, f"{filename}.keras"), custom_objects=custom_objects)
    loaded_model(tf.random.normal(shape=(config.BATCH_SIZE, config.SEGMENT_LENGTH, 1))) # dummy input to build model
    loaded_model.load_weights(os.path.join(model_dir, f"{filename}.weights.h5"))

    return loaded_model

def evaluate_model(model, test_generator, num_examples=5):
    """Evaluate the model and visualize separation results."""
    X_test, y_test = test_generator.__getitem__(0)
    indices = np.random.choice(X_test.shape[0], num_examples, replace=False)
    y_pred = model.predict(X_test[indices])
    sdrs = []
    
    for i in range(num_examples):
        example_sdrs = []
        for j in range(config.MAX_SOURCES):
            source_energy = np.sum(np.abs(y_test[indices[i], j]))
            
            if source_energy > 1e-6:
                target = y_test[indices[i], j, :, 0]
                estimate = y_pred[i, j, :, 0]
                
                target_energy = np.sum(target**2)
                error = target - estimate
                error_energy = np.sum(error**2)
                
                sdr = 10 * np.log10(target_energy / (error_energy + 1e-10))
                example_sdrs.append(sdr)
                print(f"Example {i+1}, Source {j+1}: SDR = {sdr:.2f} dB")
        
        if example_sdrs:
            avg_example_sdr = np.mean(example_sdrs)
            sdrs.append(avg_example_sdr)
            print(f"Example {i+1} Average SDR: {avg_example_sdr:.2f} dB")
    
    if sdrs:
        avg_sdr = np.mean(sdrs)
        print(f"Overall Average SDR: {avg_sdr:.2f} dB")
    
    plt.figure(figsize=(20, 4 * num_examples))
    
    for i in range(num_examples):
        plt.subplot(num_examples, config.MAX_SOURCES + 2, i * (config.MAX_SOURCES + 2) + 1)
        plt.plot(X_test[indices[i], :, 0])
        plt.title(f"Example {i+1} - Mixture")
        plt.ylim([-1, 1])
        
        for j in range(config.MAX_SOURCES):
            plt.subplot(num_examples, config.MAX_SOURCES + 2, i * (config.MAX_SOURCES + 2) + j + 2)
            plt.plot(y_test[indices[i], j, :, 0])
            plt.title(f"True Source {j+1}")
            plt.ylim([-1, 1])
            
            plt.subplot(num_examples, config.MAX_SOURCES + 2, i * (config.MAX_SOURCES + 2) + config.MAX_SOURCES + 2)
            plt.plot(y_pred[i, j, :, 0])
            plt.title(f"Pred Source {j+1}")
            plt.ylim([-1, 1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.CHECKPOINT_DIR, 'evaluation_results.png'))
    plt.show()
    
    return sdrs

def save_model(model, config, save_directory):
    print("Saving trained model...")
    model_save_path = os.path.join(config.CHECKPOINT_DIR, 'wavelet_unet_final.keras')
    model_save_path2 = os.path.join(save_directory, 'wavelet_unet_final.keras')
    
    model_json = model.to_json()
    with open(os.path.join(config.CHECKPOINT_DIR, 'model_architecture.json'), 'w') as json_file:
        json_file.write(model_json)
    
    with open(os.path.join(save_directory, 'model_architecture.json'), 'w') as json_file:
        json_file.write(model_json)
    
    model.save_weights(os.path.join(config.CHECKPOINT_DIR, 'model.weights.h5'))
    model.save_weights(os.path.join(save_directory, 'model.weights.h5'))
    
    try:
        saved_model_path = os.path.join(config.CHECKPOINT_DIR, 'wavelet_unet_savedmodel')
        tf.saved_model.save(model, saved_model_path)
        saved_model_path2 = os.path.join(save_directory, 'wavelet_unet_savedmodel')
        tf.saved_model.save(model, saved_model_path2)
        print(f"Model successfully saved to {saved_model_path}")
        print(f"Model successfully saved to {saved_model_path2}")
    except Exception as e:
        print(f"Error saving in SavedModel format: {e}")
        print("Falling back to H5 format only")
    
    try:
        model.save(model_save_path, save_format='h5')
        model.save(model_save_path2, save_format='h5')
        print(f"Model successfully saved to {model_save_path}")
        print(f"Model successfully saved to {model_save_path2}")
    except Exception as e:
        print(f"Error saving in H5 format: {e}")
        print("Model is saved as architecture + weights only")

def plot_model(history, config, save_directory):
    print("Plotting training history...")
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mse'])
    plt.plot(history.history['val_mse'])
    plt.title('Model MSE')
    plt.ylabel('MSE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.CHECKPOINT_DIR, 'training_history.png'))
    plt.savefig(os.path.join(save_directory, 'training_history.png'))
    
    print("Wavelet U-Net pipeline completed successfully!")



def separate_audio(model, audio_file, clip_duration_seconds=1.0, window_overlap_ratio=0.5):
    """Enhance audio using overlapping windows for better reconstruction.
    Stereo sources are supported: each channel is enhanced independently
    and recombined, preserving the stereo image of the original.
    
    Args:
        model: The trained enhancement model
        audio_file: Path to the input audio file
        clip_duration_seconds: Duration of each clip in seconds
        window_overlap_ratio: Overlap ratio between consecutive windows
        
    Returns:
        list: List of enhanced source arrays
    """
    clips, audio = process_audio_for_prediction_stereo(audio_file, clip_duration_seconds, window_overlap_ratio)
    
    is_stereo = len(audio.shape) > 1
    num_clips = clips.shape[0]
    num_channels = clips.shape[2] if is_stereo else 1
    
    separated_sources = np.zeros(clips.shape)
    
    for i, clip in enumerate(clips):
        for c in range(num_channels):
            clip_input = (clip[:, c] if is_stereo else clip).reshape(1, -1, 1)
            predictions = model.predict(clip_input, verbose=0)
            if is_stereo:
                separated_sources[i, :, c] = predictions[0, :, 0]
            else:
                separated_sources[i, :] = predictions[0, :, 0]
    
    reconstructed = reconstruct_audio_from_clips(separated_sources, clip_duration_seconds, window_overlap_ratio)
    reconstructed = reconstructed[:len(audio)]
    
    if is_stereo:
        reconstructed = reconstructed.reshape(len(audio), num_channels)
    
    return [reconstructed]


    
def generate_prediction(model_dir, model_filename, audio_dir, audio_filename, clip_duration_seconds=1.0, window_overlap_ratio=0.5, output_filename="output.wav", output_dir=None):
    audio_file = os.path.join(audio_dir, audio_filename)

    if output_dir is None:
        output_dir = os.path.join(audio_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    model = load_saved_model(model_dir, model_filename)
    separated_sources = separate_audio(model, audio_file, clip_duration_seconds=clip_duration_seconds, window_overlap_ratio=window_overlap_ratio)
    
    for i, source in enumerate(separated_sources):
        print(f"Writing output {output_dir}")
        sf.write(os.path.join(output_dir, output_filename), source, 44100)


if __name__ == "__main__":
    
    generate_prediction(model_dir="models", model_filename="model_13M", audio_dir="examples", audio_filename="test.mp3")
    