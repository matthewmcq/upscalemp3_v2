import json
import os
import random
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from scipy.signal import windows
import soundfile as sf
import tensorflow as tf
import pywt
import glob
import time
import gc
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm
import wave
import librosa

class AudioProcessor:
    """
    Class to preprocess audio files to be used as input and training data for the model.
    
    Generally not intended for direct use. Use the @process_audio_files function instead.
    """
    def __init__(self, clip_duration_seconds=1.0, window_overlap_ratio=0.1):
        """Initialize audio processor with clip settings"""
        self.clip_duration_seconds = clip_duration_seconds
        self.window_overlap_ratio = window_overlap_ratio
        self.samples_per_clip = int(clip_duration_seconds * 44100)
        self.overlap_samples = int(self.samples_per_clip * window_overlap_ratio)
        self.window = windows.hann(self.samples_per_clip)
    
    def load_and_normalize_audio(self, file_path):
        """Load audio file and normalize it"""
        try:
            if str(file_path).lower().endswith('.wav'):
                sample_rate, audio_array = wavfile.read(str(file_path))
                if sample_rate != 44100:
                    print(f"Warning: Audio file {file_path} sample rate is {sample_rate}Hz (expected 44.1kHz)")
            elif str(file_path).lower().endswith('.mp3'):
                # Handle MP3 files with pydub
                print(f"Converting MP3 to WAV for {file_path}")
                from pydub import AudioSegment
                
                # Load MP3 and convert to WAV format in memory
                audio_segment = AudioSegment.from_mp3(str(file_path))
                audio_segment = audio_segment.set_frame_rate(44100)  # Resample to 44.1kHz
                audio_segment = audio_segment.set_channels(1)  # Convert to mono
                
                # Get the audio data as numpy array
                audio_array = np.array(audio_segment.get_array_of_samples()).astype(np.float32)
                print(f"Audio array shape: {audio_array.shape}, frame rate: {audio_segment.frame_rate}, channels: {audio_segment.channels}")
                # Normalize based on bit depth
                if audio_segment.sample_width == 2:  # 16-bit
                    audio_array = audio_array / 32768.0
                elif audio_segment.sample_width == 1:  # 8-bit
                    audio_array = audio_array / 128.0 - 1.0
                
                sample_rate = 44100
            else:
                # Handle other formats with soundfile
                audio_array, sample_rate = sf.read(str(file_path))
                
                # Resample if needed
                if sample_rate != 44100:
                    print(f"Resampling from {sample_rate}Hz to 44100Hz for {file_path}")
                    audio_array = librosa.resample(audio_array.T, orig_sr=sample_rate, target_sr=44100).T
                    sample_rate = 44100
            
            # Convert to float32 if needed
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)
                if audio_array.max() > 1.0:
                    audio_array = audio_array / 32768.0
            
            # Convert to mono if stereo
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)
            
            # Normalize to [-1, 1]
            max_val = np.max(np.abs(audio_array))
            if max_val > 0:
                audio_array = audio_array / max_val
            
            return audio_array
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None
    
    def split_into_clips(self, audio):
        """Split audio into non-overlapping 1-second clips with light Hann windowing"""
        window = windows.hann(self.samples_per_clip)
        window = 0.1 * window + 0.9
        num_clips = len(audio) // self.samples_per_clip
        clips = []
        
        for i in range(num_clips):
            start = i * self.samples_per_clip
            end = start + self.samples_per_clip
            clip = audio[start:end]
            clip = clip * window
            clips.append(clip)
        
        if len(audio) % self.samples_per_clip > 0:
            start = num_clips * self.samples_per_clip
            remaining = audio[start:]
            clip = np.pad(remaining, (0, self.samples_per_clip - len(remaining)))
            clip = clip * window
            clips.append(clip)
        
        return clips if clips else [np.zeros(self.samples_per_clip)]
    
    def segment_audio(self, input_file, output_dir):
        """Process a single audio file and save clips"""
        audio = self.load_and_normalize_audio(input_file)
        if audio is None:
            return
        
        clips = self.split_into_clips(audio)
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        base_name = Path(input_file).stem
        for i, clip in enumerate(clips):
            clip_name = f"{base_name}_clip_{i:03d}.wav"

            # split clips evenly across 500 subfolders
            hash_value = hash(clip_name) % 500
            subfolder = output_dir / f"{hash_value:03d}"
            subfolder.mkdir(exist_ok=True)
            output_path = subfolder / clip_name
            sf.write(str(output_path), clip, 44100, format='WAV')

    def _process_audio_file(self, audio_path):
        """Process a single audio file into a TF Example
        
        Args:
            audio_path: Path to audio file in GCS
            
        Returns:
            tuple: (audio_path, serialized_example) or (audio_path, None) if error
        """
        try:
            audio_binary = tf.io.read_file(audio_path)
            feature = {
                'audio_binary': tf.train.Feature(bytes_list=tf.train.BytesList(value=[audio_binary.numpy()])),
                'path': tf.train.Feature(bytes_list=tf.train.BytesList(value=[audio_path.encode()]))
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            return audio_path, example.SerializeToString()
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return audio_path, None

    def _process_files_parallel(self, audio_files, desc="Processing files", disable_tqdm=False):
        """Process audio files in parallel using a thread pool"""
        max_workers = max(1, mp.cpu_count() - 1)
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._process_audio_file, path) for path in audio_files]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc=desc, disable=disable_tqdm):
                results.append(future.result())
        
        return results

    def convert_dataset_to_tfrecords_in_batches(self, base_input_dir, base_output_dir, batch_size=50, pause_duration=5, force_recompress=True):
        """Convert audio files distributed across subfolders to TFRecords in batches using multiprocessing.
        Should only be used if you have a large number of audio files distributed across subfolders,
        most likely processed using @process_audio_files with save_as_tfrecords=False.
        
        Note: in theory this should work with any file system, but has only been tested in GCS.

        Args:
            base_input_dir: Input directory containing audio files distributed across subfolders
            base_output_dir: Output directory for TFRecords
            batch_size: Number of subdirectories to process in each batch
            pause_duration: Pause duration between batches in seconds
            force_recompress: If True, recompresses all audio files even if they already exist
        """
        
        if not tf.io.gfile.exists(base_output_dir):
            tf.io.gfile.makedirs(base_output_dir)
        
        all_subdirs = tf.io.gfile.glob(f"{base_input_dir}/*")
        all_subdirs = [d for d in all_subdirs if tf.io.gfile.isdir(d)]
        
        print(f"Found {len(all_subdirs)} subdirectories to process")
        
        if force_recompress:
            subdirs_to_process = all_subdirs
            print(f"Will process all {len(subdirs_to_process)} subdirectories to ensure compression")
        else:
            existing_tfrecords = tf.io.gfile.glob(f"{base_output_dir}/*.tfrecord")
            processed_subdirs = {os.path.basename(p).replace('.tfrecord', '') for p in existing_tfrecords}
            subdirs_to_process = [
                subdir for subdir in all_subdirs 
                if os.path.basename(subdir) not in processed_subdirs
            ]
            print(f"Skipping {len(all_subdirs) - len(subdirs_to_process)} already processed subdirectories")
            print(f"Will process {len(subdirs_to_process)} remaining subdirectories")
        
        for batch_idx in range(0, len(subdirs_to_process), batch_size):
            batch_start_time = time.time()
            
            batch_subdirs = subdirs_to_process[batch_idx:batch_idx + batch_size]
            print(f"\nProcessing batch {batch_idx//batch_size + 1}/{(len(subdirs_to_process)-1)//batch_size + 1}")
            
            for subdir in tqdm(batch_subdirs, desc="Processing subdirectories"):
                subdir_name = os.path.basename(subdir)
                output_path = f"{base_output_dir}/{subdir_name}.tfrecord"
                
                audio_files = []
                for ext in ['.wav', '.mp3', '.flac', '.ogg']:
                    audio_files.extend(tf.io.gfile.glob(f"{subdir}/*{ext}"))
                
                if not audio_files:
                    print(f"  - {subdir_name}: No audio files found")
                    continue
                
                print(f"  - {subdir_name}: Found {len(audio_files)} audio files")
                
                # compressed TFRecord
                options = tf.io.TFRecordOptions(
                    compression_type='GZIP',
                    compression_level=6
                )
                
                with tf.io.TFRecordWriter(output_path, options=options) as writer:
                    print(f"Processing {len(audio_files)} files for {subdir_name}...")
                    results = self._process_files_parallel(
                        audio_files, 
                        desc=f"Processing {subdir_name}"
                    )
                    
                    for _, serialized_example in tqdm(results, desc=f"Writing {subdir_name}"):
                        if serialized_example is not None:
                            writer.write(serialized_example)
            
            gc.collect()
            tf.keras.backend.clear_session()
            
            batch_duration = time.time() - batch_start_time
            print(f"Batch processed in {batch_duration:.2f} seconds")
            
            if batch_idx + batch_size < len(subdirs_to_process):
                print(f"Waiting {pause_duration} seconds before next batch...")
                time.sleep(pause_duration)

def process_audio_files(base_folder, output_folder, clip_duration_seconds=1.0, window_overlap_ratio=0.1, batch_size=100, save_as_tfrecords=False):
    """
    Process all audio files in a folder and save clips to either output folder with subfolder distribution for use as training data. 
    If save_as_tfrecords is True, saves output as TFRecords spread evenly across 500 TFRecords files.
    Use this function to process audio files with the AudioProcessor class rather than instantiating
    AudioProcessor directly.
    
    Args:
        base_folder: Input folder containing audio files
        output_folder: Output folder for processed clips
        clip_duration_seconds: Duration of each clip in seconds
        window_overlap_ratio: Overlap ratio between consecutive windows
        batch_size: Number of files to process in each batch
        save_as_tfrecords: If True, saves output as TFRecords instead of audio files
    """
    os.makedirs(output_folder, exist_ok=True)
    
    processor = AudioProcessor(
        clip_duration_seconds=clip_duration_seconds,
        window_overlap_ratio=window_overlap_ratio
    )
    
    audio_files = []
    for dirpath, dirnames, filenames in os.walk(base_folder):
        for filename in filenames:
            if filename.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a')):
                audio_files.append(os.path.join(dirpath, filename))
    
    print(f"Found {len(audio_files)} audio files to process")
    
    total_clips = 0
    successful_files = 0
    batch_start = 0
    
    clips_by_subdir = {} if save_as_tfrecords else None
    
    while batch_start < len(audio_files):
        batch_end = min(batch_start + batch_size, len(audio_files))
        batch = audio_files[batch_start:batch_end]
        
        print(f"\nProcessing batch {batch_start//batch_size + 1} ({batch_start}-{batch_end} of {len(audio_files)})")
        
        for audio_file in tqdm(batch, desc="Processing audio files"):
            try:
                if save_as_tfrecords:
                    audio = processor.load_and_normalize_audio(audio_file)
                    if audio is not None:
                        clips = processor.split_into_clips(audio)
                        total_clips += len(clips)
                        rel_path = os.path.relpath(audio_file, base_folder)
                        subdir_name = os.path.splitext(rel_path)[0]
                        
                        if subdir_name not in clips_by_subdir:
                            clips_by_subdir[subdir_name] = []
                        clips_by_subdir[subdir_name].extend(clips)
                        successful_files += 1
                else:
                    processor.segment_audio(audio_file, output_folder)
                    successful_files += 1
                    
                    audio = processor.load_and_normalize_audio(audio_file)
                    if audio is not None:
                        total_clips += len(processor.split_into_clips(audio))
                
            except Exception as e:
                print(f"Error processing {audio_file}: {str(e)}")
        
        batch_start = batch_end
    
    if save_as_tfrecords and clips_by_subdir:
        print("\nSaving TFRecord files...")
        options = tf.io.TFRecordOptions(
            compression_type='GZIP',
            compression_level=6
        )
        
        for subdir_name, clips in tqdm(clips_by_subdir.items(), desc="Saving TFRecords"):
            tfrecord_path = os.path.join(output_folder, f"{subdir_name}.tfrecord")
            os.makedirs(os.path.dirname(tfrecord_path), exist_ok=True)
            
            with tf.io.TFRecordWriter(tfrecord_path, options=options) as writer:
                for clip in clips:
                    clip_bytes = tf.audio.encode_wav(
                        tf.expand_dims(clip, 0),
                        sample_rate=processor.sample_rate
                    ).numpy()
                    
                    feature = {
                        'audio_binary': tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[clip_bytes])
                        ),
                        'sample_rate': tf.train.Feature(
                            int64_list=tf.train.Int64List(value=[processor.sample_rate])
                        ),
                        'duration': tf.train.Feature(
                            float_list=tf.train.FloatList(value=[clip_duration_seconds])
                        )
                    }
                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    writer.write(example.SerializeToString())
    
    print(f"Successfully processed {successful_files}/{len(audio_files)} files")
    if save_as_tfrecords:
        print(f"Generated {total_clips} clips saved as TFRecords in {output_folder}")
    else:
        print(f"Generated approximately {total_clips} clips distributed across subfolders in {output_folder}")

def process_audio_for_prediction(audio_file, clip_duration_seconds=1.0, window_overlap_ratio=0.25):
    """
    Process a single audio file into overlapping clips suitable for model prediction.
    The clips are designed to be reconstructed back into the full audio after prediction.
    
    Args:
        audio_file: Path to the input audio file
        clip_duration_seconds: Duration of each clip in seconds
        window_overlap_ratio: Overlap ratio between consecutive windows
        
    Returns:
        tuple: (clips, audio)
            - clips: numpy array of shape (num_clips, samples_per_clip) containing the audio clips
            - audio: numpy array of the original audio
    """
    processor = AudioProcessor(
        clip_duration_seconds=clip_duration_seconds,
        window_overlap_ratio=0.1
    )
    
    audio = processor.load_and_normalize_audio(audio_file)

    
    step_size = int(44100 * (1 - window_overlap_ratio))
    num_clips = max(1, (len(audio) - 44100) // step_size + 1)
    clips = split_into_clips_new(audio, 44100, step_size, num_clips)
    
    print(f"Processed {len(clips)} clips from audio file {audio_file}")
    
    return clips, audio
    
def split_into_clips_new(audio, samples_per_clip=44100, overlap_frames=4410, num_clips=None):
        """Split audio into non-overlapping 1-second clips with light Hann windowing"""
        window = windows.hann(samples_per_clip)
        window = 0.1 * window + 0.9
        # num_clips = len(audio) // samples_per_clip
        clips = []
        
        for i in range(num_clips):
            start = i * overlap_frames
            end = start + samples_per_clip
            clip = audio[start:end]
            clip = clip * window
            clips.append(clip)
        
        if len(audio) % samples_per_clip > 0:
            start = num_clips * samples_per_clip
            remaining = audio[start:]
            clip = np.pad(remaining, (0, samples_per_clip - len(remaining)))
            clip = clip * window
            clips.append(clip)
        
        return np.array(clips) if clips else np.zeros((1, samples_per_clip))

def reconstruct_audio_from_clips(clips, clip_duration_seconds=1.0, window_overlap_ratio=0.25):
    """
    Reconstruct a full audio stream from overlapping clips that were processed using process_audio_for_prediction.
    
    Args:
        clips: numpy array of shape (num_clips, samples_per_clip) containing the audio clips
        clip_duration_seconds: Duration of each clip in seconds (must match what was used in process_audio_for_prediction)
        window_overlap_ratio: Overlap ratio between consecutive windows (must match what was used in process_audio_for_prediction)
        
    Returns:
        numpy array: The reconstructed audio signal
    """
    if clips is None or len(clips) == 0:
        return None
    
    print(clips.shape)
    num_clips, samples_per_clip = clips.shape
    print(f"Reconstructing audio from {num_clips} clips of {samples_per_clip} samples each")
    step_size = int(samples_per_clip * (1 - window_overlap_ratio))
    
    total_length = (num_clips - 1) * step_size + samples_per_clip
    
    output = np.zeros(total_length)
    normalization = np.zeros(total_length)
    
    window = windows.hann(samples_per_clip)
    window = window_overlap_ratio * window + (1 - window_overlap_ratio)
    
    for i in range(num_clips):
        start = i * step_size
        end = start + samples_per_clip
        output[start:end] += clips[i] * window
        normalization[start:end] += window
    
    mask = normalization > 1e-10
    output[mask] /= normalization[mask]
    
    return output