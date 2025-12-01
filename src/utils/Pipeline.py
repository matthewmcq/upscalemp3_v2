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
    
    def __init__(self, mp3_bitrates=[24, 32, 64, 96, 128]):
        """
        Initialize the MP3 degradation pipeline.
        Full = [24, 32, 64, 96, 128, 160, 192, 256]
        
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



# Example usage for training
if __name__ == "__main__":
    print("MP3 Artifact Removal Pipeline")
    print("="*50)
    
    # Example configuration
    tfrecord_files = tf.io.gfile.glob("/content/data/tfrecords_44k/*.tfrecord")
    
    pipeline = MP3DegradationPipeline(mp3_bitrates=[64, 96, 128, 160, 192, 256])
    
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
