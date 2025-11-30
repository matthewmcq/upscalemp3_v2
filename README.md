# upscalemp3_v2 - Wavelet-Based MP3 Audio Enhancement

A deep learning model for removing MP3 compression artifacts and restoring audio quality using a Wavelet U-Net architecture. This is a complete rewrite of the original upscalemp3, replacing the STFT-based approach with wavelets for significantly improved performance.

---

## [MODEL WEIGHTS - HUGGINGFACE](https://huggingface.co/spaces/matthewmcq/upscalemp3_v2/tree/main)

## Overview

upscalemp3_v2 uses a Wavelet U-Net architecture to learn the mapping between MP3-compressed audio and original high-fidelity audio. The model leverages the Discrete Wavelet Transform (DWT) to decompose audio signals into multi-resolution representations, enabling effective artifact removal while preserving audio quality.

### Why Wavelets?

The original upscalemp3 used Short-Time Fourier Transform (STFT) for frequency analysis, but this approach had limitations in capturing transient details and introduced artifacts at frame boundaries. The wavelet-based approach offers:

- **Better time-frequency localization**: Wavelets adapt their resolution based on frequency content
- **No blocking artifacts**: Unlike STFT's fixed window, wavelets handle transients smoothly
- **Multi-resolution analysis**: Natural decomposition into approximation and detail coefficients
- **Learnable reconstruction**: End-to-end training of both analysis and synthesis paths

### Key Features

- **Wavelet-based architecture**: Uses Daubechies wavelets (db4) for multi-resolution signal analysis
- **Residual learning**: Model learns to predict the enhanced audio residual
- **Gated skip connections**: Enhanced information flow between encoder and decoder paths
- **MP3 degradation simulation**: Training pipeline simulates various MP3 bitrates (64-256 kbps)
- **44.1kHz / 16-bit support**: Full CD-quality audio processing

---

## Architecture

The Wavelet U-Net consists of:

1. **Encoder path**: Series of downsampling blocks with DWT decomposition
2. **Bottleneck**: Dense feature processing at the lowest resolution
3. **Decoder path**: IDWT reconstruction with gated skip connections from encoder
4. **Output layer**: Residual prediction with tanh activation

```
Input (44100 samples)
    │
    ▼
┌─────────────────┐
│  Initial Conv   │
└────────┬────────┘
         │
    ┌────▼────┐
    │Encoder 1│──────────────────────────┐
    └────┬────┘                          │
         │ DWT                           │
    ┌────▼────┐                          │
    │Encoder 2│────────────────────┐     │
    └────┬────┘                    │     │
         │ DWT                     │     │
         ⋮ (10 layers)             │     │
         │                         │     │
    ┌────▼────┐                    │     │
    │Bottleneck│                   │     │
    └────┬────┘                    │     │
         │                         │     │
         │ IDWT                    │     │
    ┌────▼────┐    Gated Skip      │     │
    │Decoder 2│◄───────────────────┘     │
    └────┬────┘                          │
         │ IDWT                          │
    ┌────▼────┐    Gated Skip            │
    │Decoder 1│◄─────────────────────────┘
    └────┬────┘
         │
    ┌────▼────┐
    │  Output │
    └─────────┘
```

### Model Configuration

| Parameter | Value |
|-----------|-------|
| Sample Rate | 44,100 Hz |
| Segment Length | 1 second (44,100 samples) |
| Wavelet Family | Daubechies 4 (db4) |
| Wavelet Depth | 5 levels |
| Number of Layers | 10 |
| Initial Filters | 16 |
| Filter Size | 16 |
| Batch Size | 16 |
| Parameters | 12.6M |

---

## Training

### Training Data

The model was trained on a combination of two high-quality music datasets:

| Dataset | Description | Content |
|---------|-------------|---------|
| **MedleyDB 2.0** | Professional multitrack recordings | High-quality stereo mixtures from diverse genres |
| **MUSDB18-HQ** | Music separation benchmark dataset | Full stereo mixtures at 44.1kHz |

Combined, these datasets provide diverse musical content spanning multiple genres, instrumentation types, and recording conditions.

### Training Pipeline

The training pipeline simulates MP3 compression artifacts:

1. **Load original audio** (44.1kHz WAV) → Ground truth
2. **Encode to MP3** at random bitrate (64, 96, 128, 160, 192, or 256 kbps)
3. **Decode back to WAV** → Degraded input with compression artifacts
4. **Train model** to map degraded → original

This approach teaches the model to remove artifacts from various compression levels.

### Training Process

Training was conducted in **two phases** on Google Colab using an **NVIDIA A100 GPU**:

#### Phase 1: Initial Training
```python
# Using Config class
LEARNING_RATE = 1e-3
EPOCHS = 100
NUM_EXAMPLES = BATCH_SIZE * 3600
```

#### Phase 2: Fine-tuning (Retraining)
```python
# Using RetrainConfig class
LEARNING_RATE = 1e-4  # Lower LR for fine-tuning
EPOCHS = 100
NUM_EXAMPLES = BATCH_SIZE * 4800  # Full dataset
```

### Training Callbacks

- **ModelCheckpoint**: Saves best model based on validation loss
- **EarlyStopping**: Stops training after 10 epochs without improvement
- **ReduceLROnPlateau**: Reduces learning rate by 0.5x after 4 epochs plateau
- **TensorBoard**: Logging for visualization

---

## Installation

### Using pip

```bash
pip install -r requirements.txt
```

### Dependencies

- Python 3.11
- TensorFlow 2.19.0
- NumPy 2.0+
- SciPy 1.11.4+
- librosa 0.10.1
- protobuf 5.29.1
- PyWavelets 1.8
- pydub 0.25
- soundfile 0.12.1

**Note**: FFmpeg is required for MP3 encoding/decoding:
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg
```

---

## Usage

### Quick Start - Process Audio

```bash
python src/main.py
```

By default, this processes `examples/test.mp3` using the model in `models/model_13M`.

### Custom Audio Processing

```python
from main import generate_prediction

generate_prediction(
    model_dir="models",
    model_filename="model_13M",
    audio_dir="path/to/audio",
    audio_filename="your_file.mp3",
    clip_duration_seconds=1.0,
    window_overlap_ratio=0.1
)
```

### Training Your Own Model

#### 1. Prepare Training Data

Convert your audio files to TFRecords:

```python
from utils.data_preparation import process_audio_files

process_audio_files(
    base_folder="path/to/wav/files",
    output_folder="data/clips",
    clip_duration_seconds=1.0,
    save_as_tfrecords=True
)
```

Or use the Medley/MUSDB preprocessing script:

```python
from utils.mp3_to_wav_processing import create_tfrecords_from_datasets

create_tfrecords_from_datasets(
    medley_dir="path/to/MedleyDB",
    musdb_dir="path/to/MUSDB18",
    output_dir="./tfrecords_44k",
    num_shards=500
)
```

#### 2. Train the Model

```python
from train import train_model

model, history = train_model(
    tfrecords_dir="data/tfrecords",
    save_directory="checkpoints"
)
```

#### 3. Retrain/Fine-tune

Set `retrain=True` to use `RetrainConfig` with lower learning rate:

```python
model, history = train_model(
    tfrecords_dir="data/tfrecords",
    save_directory="checkpoints",
    model=pretrained_model,  # Optional: continue from checkpoint
    retrain=True
)
```

---

## Project Structure

```
upscalemp3_v2/
├── src/
│   ├── main.py              # Inference and prediction
│   ├── train.py             # Training script
│   ├── model.py             # Wavelet U-Net architecture
│   └── utils/
│       ├── config.py        # Training configurations
│       ├── Pipeline.py      # MP3 degradation simulation
│       ├── data_preparation.py    # Audio preprocessing
│       └── mp3_to_wav_processing.py  # Dataset processing
├── models/                  # Saved model weights
├── checkpoints/             # Training checkpoints
├── examples/                # Example audio files
├── requirements.txt         # pip dependencies
├── test_tf_records.py       # TFRecord verification utility
└── README.md
```

---

## Model Weights

Pre-trained weights are [available on huggingface](https://huggingface.co/spaces/matthewmcq/upscalemp3_v2/tree/main) and should be pasted in the `models/` directory:

- `model_13M.keras` - Full model (Keras format)
- `model_13M.weights.h5` - Weights only (H5 format)

---

## Technical Details

### Wavelet Transform Implementation

The model uses custom TensorFlow layers for DWT/IDWT:

- **DWTLayer**: Decomposes signal into approximation and detail coefficients
- **IDWTLayer**: Reconstructs signal from wavelet coefficients
- **Daubechies 4 (db4)**: 8-tap filter with good frequency localization

### Gated Skip Connections

Unlike standard U-Net skip connections, gated connections learn to weight the importance of encoder features:

```python
decoder_gated = decoder_features * sigmoid(gate(decoder_features))
encoder_gated = encoder_features * sigmoid(gate(encoder_features))
output = concat([decoder_gated, encoder_gated])
```

### Loss Function

Mean Squared Error (MSE) between predicted and original audio in the time domain.

---

## Evaluation

The model is evaluated using Signal-to-Distortion Ratio (SDR):

```
SDR = 10 * log10(||target||² / ||target - estimate||²)
```

Higher SDR indicates better reconstruction quality.

---

## Limitations

- **Processing time**: Real-time processing not yet optimized
- **Memory usage**: Long audio files are processed in 1-second chunks
- **Extreme compression**: Very low bitrates (<64 kbps) may have limited recovery, if this is desired feel free to retrain the model and pass any desired bitrates as a parameter when generating the pipeline
- **Non-music audio**: Trained primarily on music; speech/effects may vary

---

## Future Work

- [ ] Real-time inference optimization
- [ ] Extended receptive field for better temporal coherence
- [ ] Perceptual loss functions (spectral loss, multi-resolution STFT)
- [ ] Support for stereo audio
- [ ] Quantization for edge deployment

---

## Acknowledgments

- Training infrastructure provided by Google Colab (A100 GPU)
- MedleyDB 2.0 and MUSDB18-HQ datasets
- PyWavelets library for wavelet implementations
- This project was forked from [Parrotfish](https://github.com/rmeghji/parrotfish), a wavelet-based source separation model I made with [Rayhan Meghji](https://github.com/rmeghji)

---

## License

MIT License