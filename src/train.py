import os
import numpy as np
import tensorflow as tf
import random
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from itertools import combinations, permutations
import soundfile as sf
import librosa
import pywt
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
from main import save_model, plot_model

# config = Config()

def get_callbacks(save_directory, config):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(config.CHECKPOINT_DIR, 'wavelet_unet_{epoch:02d}_{val_loss:.4f}.keras'),
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            save_weights_only=False,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(save_directory, 'wavelet_unet_{epoch:02d}_{val_loss:.4f}.keras'),
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            save_weights_only=False,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(config.CHECKPOINT_DIR, 'wavelet_unet_{epoch:02d}_{val_loss:.4f}_weightsonly.weights.h5'),
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            save_weights_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(save_directory, 'wavelet_unet_{epoch:02d}_{val_loss:.4f}_weightsonly.weights.h5'),
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            save_weights_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(config.CHECKPOINT_DIR, 'logs'),
            histogram_freq=1,
            update_freq='epoch'
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(save_directory, 'logs'),
            histogram_freq=1,
            update_freq='epoch'
        )
    ]

def train_model(clips_dir=None, tfrecords_dir=None, save_directory=None, model=None, retrain=False):
    """Main function to run the audio source separation pipeline"""

    config = Config() 
    
    if retrain:
        config = RetrainConfig()
        
    tf.config.optimizer.set_jit(True)
    for device in tf.config.list_physical_devices('GPU'):
        try:
            tf.config.experimental.set_memory_growth(device, True)
            print(f"Memory growth enabled for {device}")
        except RuntimeError as e:
            print(f"Error setting memory growth: {e}")  
    
    print("Starting audio source separation pipeline...")
 
    if tfrecords_dir and not clips_dir:
        print("Creating TensorFlow dataset for training from TFRecords...")
        tfrecord_files = tf.io.gfile.glob(f"{tfrecords_dir}/*.tfrecord")
        if not tfrecord_files:
            raise ValueError(f"No TFRecord files found in {tfrecords_dir}")

        random.shuffle(tfrecord_files)
        num_val_records = int(len(tfrecord_files) * config.VAL_SPLIT)
        print(f"Num records: {len(tfrecord_files)}")
        print(f"Num val records: {num_val_records}")
        print(f"Num train records: {len(tfrecord_files) - num_val_records}")
        train_records = tfrecord_files[:-num_val_records]
        val_records = tfrecord_files[-num_val_records:]
        
    
        pipeline = MP3DegradationPipeline(mp3_bitrates=[64, 96, 128, 160, 192, 256])
        
        # Create dataset
        train_dataset = pipeline.create_training_dataset(
            train_records,  
            batch_size=config.BATCH_SIZE
        )
        
        val_dataset = pipeline.create_training_dataset(
            val_records,  
            batch_size=config.BATCH_SIZE
        )

        print(f"Training dataset ready")

    train_size = int(config.NUM_EXAMPLES * (1 - config.VAL_SPLIT))
    val_size = int(config.NUM_EXAMPLES * config.VAL_SPLIT)
    
    train_steps = int(train_size / config.BATCH_SIZE)
    val_steps = int(val_size / config.BATCH_SIZE)
    
    print(f"Training dataset created with {train_size} examples")
    print(f"Validation dataset created with {int(config.NUM_EXAMPLES * config.VAL_SPLIT)} examples")
    print("Creating Wavelet U-Net model...")
    
    if model is None:
        model = WaveletUNet(
            num_coeffs=config.NUM_COEFFS,
            wavelet_depth=config.WAVELET_DEPTH,
            batch_size=config.BATCH_SIZE,
            channels=config.CHANNELS,
            num_layers=config.NUM_LAYERS,
            num_init_filters=config.NUM_INIT_FILTERS,
            filter_size=config.FILTER_SIZE,
            merge_filter_size=config.MERGE_FILTER_SIZE,
            l1_reg=config.L1_REG,
            l2_reg=config.L2_REG,
            max_sources=config.MAX_SOURCES,
            wavelet_family=config.WAVELET_FAMILY
        )
    
    print("Compiling model...")
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
    loss_fn = tf.keras.losses.MeanSquaredError()
    dummy_input = tf.zeros((config.BATCH_SIZE, config.SEGMENT_LENGTH, 1))
    _ = model(dummy_input)
    
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=['mse'],
        jit_compile=True
    )
    
    model.summary()
    
    print("Setting up training callbacks...")
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    callbacks = get_callbacks(save_directory, config)

    print(f"Training model for {config.EPOCHS} epochs...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config.EPOCHS,
        callbacks=callbacks,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
    )
    
    save_model(model, config, save_directory)
    plot_model(history, config, save_directory)
    
    print("Wavelet U-Net training completed successfully!")
    return model, history

if __name__ == "__main__":
    train_model(tfrecords_dir="data", save_directory="checkpoints")
