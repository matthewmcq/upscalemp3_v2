"""
Modified training pipeline for MP3 artifact removal.
Instead of source separation, this trains a model to restore audio quality
by removing MP3 compression artifacts.

The pipeline:
1. Original WAV (44.1kHz/16-bit) → Ground truth
2. WAV → MP3 → WAV → Degraded input with compression artifacts
3. Train model: Degraded input → Original quality
"""

import tensorflow as tf
import numpy as np
import io
import subprocess
import tempfile
import os
from pathlib import Path

class MP3DegradationPipeline:
    """
    Pipeline that simulates MP3 compression artifacts for training.
    """
    
    def __init__(self, mp3_bitrates=[64, 96, 128, 160, 192, 256]):
        """
        Initialize the MP3 degradation pipeline.
        
        Args:
            mp3_bitrates: List of bitrates to randomly choose from for MP3 encoding.
                         Lower bitrates = more artifacts = harder restoration task
                         Common bitrates:
                         - 64 kbps: Low quality, noticeable artifacts
                         - 96 kbps: Acceptable for speech
                         - 128 kbps: Decent quality, standard for many applications  
                         - 160 kbps: Good quality
                         - 192 kbps: High quality, fewer artifacts
                         - 256/320 kbps: Very high quality (might be too easy for training)
        """
        self.mp3_bitrates = mp3_bitrates
        self.sample_rate = 44100
        
    @tf.function
    def parse_tfrecord(self, example_proto):
        """Parse a TFRecord example containing 44.1kHz audio."""
        feature_description = {
            'audio_binary': tf.io.FixedLenFeature([], tf.string),
            'sample_rate': tf.io.FixedLenFeature([], tf.int64),
            'samples': tf.io.FixedLenFeature([], tf.int64),
            'source_file': tf.io.FixedLenFeature([], tf.string),
            'clip_index': tf.io.FixedLenFeature([], tf.int64),
            'dataset': tf.io.FixedLenFeature([], tf.string),
        }
        
        parsed_features = tf.io.parse_single_example(example_proto, feature_description)
        
        # Decode WAV to get original audio
        audio_tensor = tf.audio.decode_wav(
            parsed_features['audio_binary'],
            desired_channels=1,
            desired_samples=44100
        )
        
        original_audio = tf.reshape(audio_tensor.audio, [44100])
        
        # Normalize to [-1, 1] range
        max_val = tf.reduce_max(tf.abs(original_audio))
        original_audio = tf.cond(
            max_val > 0,
            lambda: original_audio / max_val,
            lambda: original_audio
        )
        
        return original_audio, parsed_features['audio_binary']
    
    def simulate_mp3_compression_py(self, audio_wav_bytes, bitrate):
        """
        Python function to simulate MP3 compression using ffmpeg.
        This will be wrapped with tf.py_function for use in the pipeline.
        
        Args:
            audio_wav_bytes: WAV file as bytes
            bitrate: MP3 bitrate in kbps
            
        Returns:
            Degraded audio as numpy array
        """
        try:
            # Create temporary files
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
                tmp_wav.write(audio_wav_bytes.numpy())
                tmp_wav_path = tmp_wav.name
            
            tmp_mp3_path = tmp_wav_path.replace('.wav', '.mp3')
            tmp_out_wav_path = tmp_wav_path.replace('.wav', '_out.wav')
            
            # Convert WAV → MP3 using ffmpeg (or lame)
            # Using ffmpeg (more commonly available):
            subprocess.run([
                'ffmpeg', '-i', tmp_wav_path,
                '-b:a', f'{bitrate}k',
                '-ar', '44100',
                '-ac', '1',
                '-y',  # Overwrite output
                tmp_mp3_path
            ], capture_output=True, check=True)
            
            # Convert MP3 → WAV
            subprocess.run([
                'ffmpeg', '-i', tmp_mp3_path,
                '-ar', '44100',
                '-ac', '1',
                '-y',
                tmp_out_wav_path
            ], capture_output=True, check=True)
            
            # Read the degraded WAV
            audio_tensor = tf.audio.decode_wav(tf.io.read_file(tmp_out_wav_path))
            degraded_audio = tf.reshape(audio_tensor.audio, [-1])
            
            # Clean up temp files
            os.unlink(tmp_wav_path)
            os.unlink(tmp_mp3_path)
            os.unlink(tmp_out_wav_path)
            
            # Ensure same length as original
            if len(degraded_audio) > 44100:
                degraded_audio = degraded_audio[:44100]
            elif len(degraded_audio) < 44100:
                degraded_audio = tf.pad(degraded_audio, [[0, 44100 - len(degraded_audio)]])
            
            return degraded_audio.numpy()
            
        except Exception as e:
            print(f"Error in MP3 compression: {e}")
            # Return original if compression fails
            audio_tensor = tf.audio.decode_wav(audio_wav_bytes.numpy())
            return tf.reshape(audio_tensor.audio, [44100]).numpy()
    
    def apply_mp3_degradation(self, original_audio, audio_wav_bytes):
        """
        Apply MP3 compression artifacts to audio.
        
        Args:
            original_audio: Original high-quality audio tensor
            audio_wav_bytes: Original audio as WAV bytes
            
        Returns:
            tuple: (degraded_input, original_target)
        """
        # Randomly select bitrate
        bitrate = np.random.choice(self.mp3_bitrates)
        
        # Apply MP3 compression using py_function
        degraded_audio = tf.py_function(
            func=self.simulate_mp3_compression_py,
            inp=[audio_wav_bytes, bitrate],
            Tout=tf.float32
        )
        
        # Reshape and normalize
        degraded_audio = tf.reshape(degraded_audio, [44100])
        max_val = tf.reduce_max(tf.abs(degraded_audio))
        degraded_audio = tf.cond(
            max_val > 0,
            lambda: degraded_audio / max_val,
            lambda: degraded_audio
        )
        
        return degraded_audio, original_audio
    
    def create_training_dataset(
        self, 
        tfrecord_files, 
        batch_size=32,
        shuffle_buffer=10000,
        cache=True
    ):
        """
        Create a training dataset with MP3 degradation.
        
        Args:
            tfrecord_files: List of TFRecord file paths
            batch_size: Batch size for training
            shuffle_buffer: Size of shuffle buffer
            cache: Whether to cache the dataset
            
        Returns:
            tf.data.Dataset that yields (degraded_input, original_target) pairs
        """
        # Create dataset from TFRecords
        dataset = tf.data.TFRecordDataset(
            tfrecord_files,
            compression_type='GZIP',
            num_parallel_reads=tf.data.AUTOTUNE
        )
        
        # Parse TFRecords
        dataset = dataset.map(
            self.parse_tfrecord,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Cache before degradation (cache the clean data)
        if cache:
            dataset = dataset.cache()
        
        # Apply MP3 degradation
        dataset = dataset.map(
            self.apply_mp3_degradation,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Shuffle and batch
        dataset = dataset.shuffle(shuffle_buffer)
        dataset = dataset.repeat(100).batch(batch_size)
        
        # Add channel dimension for conv layers
        dataset = dataset.map(
            lambda x, y: (
                tf.expand_dims(x, -1),  # (batch, 44100, 1)
                tf.expand_dims(y, -1)   # (batch, 44100, 1)
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Prefetch for performance
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset


class SimplerMP3DegradationPipeline:
    """
    Simpler alternative that uses spectral filtering to simulate MP3 artifacts
    instead of actual MP3 encoding (faster but less realistic).
    """
    
    def __init__(self, quality_levels=[0.3, 0.5, 0.7, 0.9]):
        """
        Args:
            quality_levels: List of quality factors (0-1, lower = more degradation)
        """
        self.quality_levels = quality_levels
    
    @tf.function
    def simulate_compression_artifacts(self, audio):
        """
        Simulate MP3-like artifacts using spectral manipulation.
        This is faster than actual MP3 encoding but less realistic.
        """
        # Randomly select quality level
        quality = tf.random.uniform([], 
                                   min(self.quality_levels), 
                                   max(self.quality_levels))
        
        # Apply FFT
        fft = tf.signal.rfft(audio)
        
        # Simulate frequency band limitation (MP3 cuts high frequencies)
        freq_cutoff = tf.cast(tf.cast(tf.shape(fft)[0], tf.float32) * quality, tf.int32)
        mask = tf.concat([
            tf.ones(freq_cutoff, dtype=tf.complex64),
            tf.zeros(tf.shape(fft)[0] - freq_cutoff, dtype=tf.complex64)
        ], axis=0)
        
        # Apply frequency mask
        fft_degraded = fft * mask
        
        # Add quantization noise (simulates lossy compression)
        noise_level = (1.0 - quality) * 0.01
        noise = tf.complex(
            tf.random.normal(tf.shape(fft), stddev=noise_level),
            tf.random.normal(tf.shape(fft), stddev=noise_level)
        )
        fft_degraded = fft_degraded + noise
        
        # Convert back to time domain
        degraded = tf.signal.irfft(fft_degraded)
        
        # Ensure same length
        degraded = degraded[:44100]
        if tf.shape(degraded)[0] < 44100:
            degraded = tf.pad(degraded, [[0, 44100 - tf.shape(degraded)[0]]])
        
        # Normalize
        degraded = degraded / (tf.reduce_max(tf.abs(degraded)) + 1e-8)
        
        return degraded
    
    def create_training_dataset(
        self,
        tfrecord_files,
        batch_size=32,
        shuffle_buffer=10000
    ):
        """Create training dataset with simulated artifacts."""
        
        feature_description = {
            'audio_binary': tf.io.FixedLenFeature([], tf.string),
            'sample_rate': tf.io.FixedLenFeature([], tf.int64),
            'samples': tf.io.FixedLenFeature([], tf.int64),
            'source_file': tf.io.FixedLenFeature([], tf.string),
            'clip_index': tf.io.FixedLenFeature([], tf.int64),
            'dataset': tf.io.FixedLenFeature([], tf.string),
        }
        
        def parse_and_degrade(example_proto):
            # Parse TFRecord
            parsed = tf.io.parse_single_example(example_proto, feature_description)
            
            # Decode audio
            audio_tensor = tf.audio.decode_wav(parsed['audio_binary'])
            original = tf.reshape(audio_tensor.audio, [44100])
            
            # Normalize
            original = original / (tf.reduce_max(tf.abs(original)) + 1e-8)
            
            # Create degraded version
            degraded = self.simulate_compression_artifacts(original)
            
            return degraded, original
        
        # Build dataset
        dataset = tf.data.TFRecordDataset(
            tfrecord_files,
            compression_type='GZIP',
            num_parallel_reads=tf.data.AUTOTUNE
        )
        
        dataset = dataset.map(
            parse_and_degrade,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        dataset = dataset.cache()
        dataset = dataset.shuffle(shuffle_buffer)
        dataset = dataset.repeat(100).batch(batch_size)
        
        # Add channel dimension
        dataset = dataset.map(
            lambda x, y: (
                tf.expand_dims(x, -1),
                tf.expand_dims(y, -1)
            )
        )
        
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset


def get_degradation_pipeline(method='simple', **kwargs):
    """
    Factory function to get the appropriate degradation pipeline.
    
    Args:
        method: 'mp3' for real MP3 encoding (realistic but slow)
                'simple' for spectral simulation (fast but less realistic)
        **kwargs: Additional arguments for the pipeline
        
    Returns:
        Pipeline instance
    """
    if method == 'mp3':
        # Check if ffmpeg is available
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
            print("Using real MP3 encoding pipeline (realistic but slower)")
            return MP3DegradationPipeline(**kwargs)
        except:
            print("ffmpeg not found, falling back to simulation pipeline")
            return SimplerMP3DegradationPipeline(**kwargs)
    else:
        print("Using simulated compression pipeline (faster but less realistic)")
        return SimplerMP3DegradationPipeline(**kwargs)


# Example usage for training
if __name__ == "__main__":
    print("MP3 Artifact Removal Pipeline")
    print("="*50)
    
    # Example configuration
    tfrecord_files = tf.io.gfile.glob("/content/data/tfrecords_44k/*.tfrecord")
    
    # Choose pipeline
    # Option 1: Real MP3 encoding (more realistic)
    pipeline = MP3DegradationPipeline(mp3_bitrates=[64, 96, 128, 160, 192, 256])
    
    # Option 2: Simulated artifacts (faster)
    # pipeline = SimplerMP3DegradationPipeline(quality_levels=[0.3, 0.5, 0.7, 0.9])
    
    # Create dataset
    train_dataset = pipeline.create_training_dataset(
        tfrecord_files[:450],  # 90% for training
        batch_size=16
    )
    
    val_dataset = pipeline.create_training_dataset(
        tfrecord_files[450:],  # 10% for validation
        batch_size=16
    )
    
    print(f"Training dataset ready")
    print(f"Input: MP3-degraded audio (44.1kHz)")
    print(f"Target: Original quality audio (44.1kHz)")
    print("\nModel will learn to remove compression artifacts")


# import json
# import os
# import random
# from pathlib import Path
# import numpy as np
# from scipy.io import wavfile
# from scipy.signal import windows
# import soundfile as sf
# import tensorflow as tf
# import pywt
# import glob
# import time
# import gc
# import multiprocessing as mp
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from tqdm.auto import tqdm

# class Waveform:
#     """Class to store waveform data with associated metadata"""
#     def __init__(self, waveform, source_id=None, source_ids=None):
#         self.waveform = waveform
#         self.source_id = source_id
#         self.source_ids = source_ids if source_ids else []
#         self.tensor_coeffs = None

# def makeWaveDictBatch(mixed_waveforms, source_waveforms, source_ids_list, mixed_ids):
#     """Create dictionary containing both mixed and source waveforms"""
#     wave_dict = {}
    
#     for i, (mixed_waveform, mix_id) in enumerate(zip(mixed_waveforms, mixed_ids)):
#         wave_dict[mix_id] = Waveform(mixed_waveform, source_id=mix_id, source_ids=source_ids_list[i])
    
#     for i, source_list in enumerate(source_ids_list):
#         for j, source_id in enumerate(source_list):
#             if source_id and source_id not in wave_dict:
#                 wave_dict[source_id] = Waveform(source_waveforms[i][j], source_id=source_id)
    
#     return wave_dict

# class AudioMixerGenerator:
#     def __init__(self, base_dir, clips_dir, num_speakers=4, batch_size=1000):
#         """
#         Initialize the AudioMixerGenerator for mixing pre-processed 1-second clips.
        
#         Args:
#             base_dir: Base directory containing the data
#             clips_dir: Directory containing the pre-processed 1-second audio clips
#             num_speakers: Number of speakers to mix together
#             batch_size: Number of files to process at once when selecting random files
#         """
#         self.base_dir = Path(base_dir)
#         self.clips_dir = Path(clips_dir)
#         self.input_dir = self.clips_dir
#         self.num_speakers = num_speakers
#         self.batch_size = batch_size
#         self.samples_per_clip = 16000
        
#     def load_clip(self, file_path):
#         """Load a pre-processed 1-second clip"""
#         try:
#             sample_rate, audio_array = wavfile.read(str(file_path))
#             if sample_rate != 16000:
#                 raise ValueError(f"Clip {file_path} must be 16kHz (found {sample_rate}Hz)")
            
#             if len(audio_array) != self.samples_per_clip:
#                 raise ValueError(f"Clip {file_path} must be exactly 1 second (found {len(audio_array)/16000:.2f}s)")
            
#             # Convert to float32 if needed
#             if audio_array.dtype != np.float32:
#                 audio_array = audio_array.astype(np.float32)
#                 if audio_array.max() > 1.0:
#                     audio_array = audio_array / 32768.0
            
#             return audio_array
            
#         except Exception as e:
#             print(f"Error processing {file_path}: {e}")
#             return None
    
#     def get_random_files(self):
#         """Get random files without loading entire directory"""
#         all_files = []
#         max_attempts = 10
#         attempts = 0
        
#         while len(all_files) < self.num_speakers and attempts < max_attempts:
#             subfolder = f"{random.randint(0, 499):03d}"
#             subfolder_path = self.clips_dir / subfolder
            
#             if not subfolder_path.exists():
#                 attempts += 1
#                 continue
            
#             files = [f for f in os.listdir(subfolder_path) if f.endswith('.wav')]
#             if not files:
#                 attempts += 1
#                 continue
            
#             needed = self.num_speakers - len(all_files)
#             batch_samples = random.sample(files, min(needed, len(files)))
#             all_files.extend(f"{subfolder}/{f}" for f in batch_samples)
        
#         if len(all_files) < self.num_speakers:
#             print(f"Warning: Could only find {len(all_files)} valid files")
        
#         return all_files
    
#     def generate_sample(self):
#         """Mix multiple 1-second clips together
        
#         Returns:
#             tuple: (mixed_audio, input_clips) where:
#                 - mixed_audio: tensor of shape (1, samples_per_clip) containing the mixed audio
#                 - input_clips: tensor of shape (num_speakers, samples_per_clip) containing the individual clips
#         """
#         selected_files = self.get_random_files()
#         mixed = np.zeros(self.samples_per_clip)
#         input_clips = []
#         source_ids = []
        
#         for rel_path in selected_files:
#             file_path = self.input_dir / rel_path
#             audio = self.load_clip(file_path)
            
#             if audio is not None:
#                 mixed += audio
#                 input_clips.append(audio)
#                 source_ids.append(rel_path)
        
#         if not input_clips:
#             return (
#                 tf.zeros((1, self.samples_per_clip), dtype=tf.float32),
#                 tf.zeros((self.num_speakers, self.samples_per_clip), dtype=tf.float32),
#                 []
#             )
        
#         max_val = np.max(np.abs(mixed))
#         if max_val > 0:
#             mixed = mixed / max_val
        
#         mixed_tensor = tf.convert_to_tensor([mixed], dtype=tf.float32)
        
#         while len(input_clips) < self.num_speakers:
#             input_clips.append(np.zeros(self.samples_per_clip))
#             source_ids.append('')
        
#         sources_tensor = tf.convert_to_tensor(input_clips, dtype=tf.float32)
        
#         return mixed_tensor, sources_tensor, source_ids

#     def batch_training_data(self, batch_size=32, wavelet_level=5):
#         """Batch the training data
        
#         params:
#         - batch_size: int, batch size
#         - wavelet_level: int, wavelet level
        
#         return:
#         - X: tf.Tensor, mixed waveforms
#         - Y: tf.Tensor, source waveforms
#         """
#         mixed_waveforms = []
#         all_source_waveforms = []
#         all_source_ids = []
        
#         for _ in range(batch_size):
#             mixed_tensor, sources_tensor, source_ids = self.generate_sample()
#             mixed_waveforms.append(mixed_tensor.numpy()[0])
#             all_source_waveforms.append(sources_tensor.numpy())
#             all_source_ids.append(source_ids)
        
#         mixed_ids = ["_".join(ids) for ids in all_source_ids]
#         wave_dict = makeWaveDictBatch(mixed_waveforms, all_source_waveforms, all_source_ids, mixed_ids)

#         X = []
#         Y = []

#         for mix_id in mixed_ids:
#             X.append(wave_dict[mix_id].waveform)
        
#         for source_ids in all_source_ids:
#             source_list = []
#             for source_id in source_ids:
#                 if source_id:
#                     source_list.append(wave_dict[source_id].waveform)
#                 else:
#                     source_list.append(np.zeros(self.samples_per_clip))
#             Y.append(source_list)

#         X = tf.convert_to_tensor(X)
#         Y = tf.convert_to_tensor(Y)
        
#         X = tf.expand_dims(X, axis=-1)
#         Y = tf.expand_dims(Y, axis=-1)

#         return X, Y

# def create_tf_dataset(base_dir, clips_dir, num_speakers, batch_size=32, wavelet_level=5):
#     """Create a TensorFlow dataset that generates audio samples on-the-fly"""
#     mixer = AudioMixerGenerator(
#         base_dir,
#         clips_dir,
#         num_speakers=num_speakers
#     )
    
#     def generator_fn():
#         while True:
#             X, Y = mixer.batch_training_data(batch_size, wavelet_level)
#             yield X, Y

#     dataset = tf.data.Dataset.from_generator(
#         generator_fn,
#         output_signature=(
#             tf.TensorSpec(shape=(batch_size, mixer.samples_per_clip, 1), dtype=tf.float32),
#             tf.TensorSpec(shape=(batch_size, num_speakers, mixer.samples_per_clip, 1), dtype=tf.float32)
#         )
#     )

#     return dataset.prefetch(tf.data.AUTOTUNE)

# def create_tf_dataset_from_tfrecords(tfrecord_files, min_sources=1, max_sources=3, batch_size=128, is_train=True):
#     """Create a TensorFlow dataset from TFRecord files containing audio data
#     Optimized for GPU execution on A100
    
#     Args:
#         tfrecord_files: List of TFRecord files
#         max_sources: Maximum number of audio sources to mix
#         batch_size: Batch size for training (increased for A100)
#         is_train: Whether to use training-specific logic
        
#     Returns:
#         tf.data.Dataset: Dataset that yields (mixed_audio, separated_audio) pairs
#         where separated_audio is padded with zeros if less than max_sources are used
#     """
    
#     samples_per_clip = 44100
#     options = tf.data.Options()
#     options.experimental_optimization.map_parallelization = True
#     options.experimental_optimization.map_fusion = True
#     options.experimental_optimization.parallel_batch = True
#     options.threading.max_intra_op_parallelism = 8
#     options.threading.private_threadpool_size = 32
#     options.experimental_deterministic = False
    
#     def _parse_tfrecord(example_proto):
#         """Parse one TFRecord example"""
#         feature_description = {
#             'audio_binary': tf.io.FixedLenFeature([], tf.string),
#             'path': tf.io.FixedLenFeature([], tf.string)
#         }
#         parsed_features = tf.io.parse_single_example(example_proto, feature_description)
#         audio_tensor = tf.audio.decode_wav(parsed_features['audio_binary'])
#         waveform = audio_tensor.audio
#         current_length = tf.shape(waveform)[0]
        
#         def pad_waveform():
#             padding = [[0, samples_per_clip - current_length], [0, 0]]
#             return tf.pad(waveform, padding)
        
#         def trim_waveform():
#             return waveform[:samples_per_clip]
        
#         waveform = tf.cond(
#             current_length < samples_per_clip,
#             pad_waveform,
#             trim_waveform
#         )
        
#         return tf.reshape(waveform, (samples_per_clip, 1))

#     @tf.function(jit_compile=True)
#     def _prepare_batch(waveforms):
#         """Prepare a batch of waveforms"""
#         batch_size = tf.shape(waveforms)[0]
        
#         mixed_audio = tf.zeros((batch_size, samples_per_clip, 1), dtype=tf.float32)
#         separated_audio = tf.zeros((batch_size, max_sources, samples_per_clip, 1), dtype=tf.float32)
        
#         num_sources_per_batch = tf.random.uniform(
#             shape=(batch_size,),
#             minval=min_sources,
#             maxval=max_sources + 1,
#             dtype=tf.int32
#         )
        
#         source_indices = tf.range(max_sources, dtype=tf.int32)
#         source_mask = tf.expand_dims(source_indices, 0) < tf.expand_dims(num_sources_per_batch, 1)
        
#         weights = tf.random.uniform((batch_size, max_sources), minval=0.5, maxval=1.5)
#         weights = weights * tf.cast(source_mask, tf.float32)
#         weights = weights / tf.reduce_sum(weights, axis=1, keepdims=True)
        
#         all_indices = tf.random.uniform((batch_size, max_sources), 0, batch_size, dtype=tf.int32)
        
#         for i in range(max_sources):
#             indices = all_indices[:, i]
#             indices = all_indices[:, i]
            
#             selected_waveforms = tf.gather(waveforms, indices)
#             batch_weights = tf.reshape(weights[:, i], (batch_size, 1, 1))
#             weighted_waveforms = selected_waveforms * batch_weights
#             source_mask_i = tf.reshape(source_mask[:, i], (batch_size, 1, 1))
#             mixed_audio += weighted_waveforms * tf.cast(source_mask_i, tf.float32)
#             update_indices = tf.stack([tf.range(batch_size), tf.fill((batch_size,), i)], axis=1)
#             separated_audio = tf.tensor_scatter_nd_update(
#                 separated_audio,
#                 update_indices,
#                 weighted_waveforms
#             )
        
#         return mixed_audio, separated_audio

#     dataset = tf.data.TFRecordDataset(
#         tfrecord_files, 
#         compression_type='GZIP', 
#         num_parallel_reads=tf.data.AUTOTUNE,
#         buffer_size=16 * 1024 * 1024
#     )
    
#     dataset = dataset.with_options(options)
#     dataset = dataset.map(_parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
#     dataset = dataset.cache()
#     if is_train:
#         dataset = dataset.shuffle(buffer_size=50000)
#         dataset = dataset.repeat()
#     dataset = dataset.batch(batch_size)
#     dataset = dataset.map(_prepare_batch, num_parallel_calls=tf.data.AUTOTUNE)
#     dataset = dataset.prefetch(tf.data.AUTOTUNE)
#     return dataset
    
# def generate_sample_from_clips(clip1, clip2):
#     """Mix multiple 1-second clips together (using this to generate one example right now can delete later)
    
#     Returns:
#         tuple: (mixed_audio, input_clips) where:
#             - mixed_audio: tensor of shape (1, samples_per_clip) containing the mixed audio
#             - input_clips: tensor of shape (num_speakers, samples_per_clip) containing the individual clips
#     """
#     mixed = np.zeros(44100)
#     input_clips = []
    
#     for clip in [clip1, clip2]:
#         if clip is not None:
#             mixed += clip
#             input_clips.append(clip)
    
#     if not input_clips:
#         return (
#             tf.zeros((1, 16000), dtype=tf.float32),
#             tf.zeros((3, 16000), dtype=tf.float32),
#             []
#         )
    
#     max_val = np.max(np.abs(mixed))
#     if max_val > 0:
#         mixed = mixed / max_val
    
#     mixed_tensor = tf.convert_to_tensor([mixed], dtype=tf.float32)
    
#     # while len(input_clips) < 3:
#     #     input_clips.append(np.zeros(16000))
    
#     # sources_tensor = tf.convert_to_tensor(input_clips, dtype=tf.float32)
    
#     return mixed_tensor#, sources_tensor