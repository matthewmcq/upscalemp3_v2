class Config_small: # swapped to 32 filters US 16 DS
    """Configuration class for parrotfish""" 
    # Data settings
    DATA_DIR = "data/audio"
    CLIPS_DIR = "data/clips"
    SAMPLE_RATE = 44100
    SEGMENT_LENGTH = 44100  # 1 second at 16kHz
    
    # Model settings
    NUM_COEFFS = 44100  # 1 second at 16kHz
    WAVELET_DEPTH = 5
    BATCH_SIZE = 16 # 16-32
    CHANNELS = 1  # Mono audio
    NUM_LAYERS = 10 # 10-12
    NUM_INIT_FILTERS = 16 ## was 24
    FILTER_SIZE = 16 # was 16 should be 16
    MERGE_FILTER_SIZE = 16 # was 5 should be like 8
    L1_REG = 0*1e-6
    L2_REG = 0 *1e-6
    
    # Training settings
    LEARNING_RATE = 1e-3
    EPOCHS = 100
    VAL_SPLIT = 0.1
    CHECKPOINT_DIR = "checkpoints"
    MAX_WORKERS = 4
    CACHE_REFRESH_EVERY = 5
    
    # Mixture generation
    MIN_SOURCES = 1
    MAX_SOURCES = 1 # first curriculum learning
    NUM_EXAMPLES = BATCH_SIZE * 4000
    
    
    # Wavelet settings
    WAVELET_FAMILY = 'db4'  # Daubechies wavelet with 4 vanishing moments
    
class Config_flipped: # swapped to 32 filters US 16 DS
    """Configuration class for parrotfish""" 
    # Data settings
    DATA_DIR = "data/audio"
    CLIPS_DIR = "data/clips"
    SAMPLE_RATE = 16000
    SEGMENT_LENGTH = 16000  # 1 second at 16kHz
    
    # Model settings
    NUM_COEFFS = 16000  # 1 second at 16kHz
    WAVELET_DEPTH = 5
    BATCH_SIZE = 16 # 16-32
    CHANNELS = 1  # Mono audio
    NUM_LAYERS = 11 # 10-12
    NUM_INIT_FILTERS = 32 ## was 24
    FILTER_SIZE = 16 # was 16 should be 16
    MERGE_FILTER_SIZE = 32 # was 5 should be like 8
    L1_REG = 0*1e-6
    L2_REG = 0 *1e-6
    
    # Training settings
    LEARNING_RATE = 1e-4
    EPOCHS = 100
    VAL_SPLIT = 0.1
    CHECKPOINT_DIR = "checkpoints"
    MAX_WORKERS = 4
    CACHE_REFRESH_EVERY = 5
    
    # Mixture generation
    MIN_SOURCES = 2
    MAX_SOURCES = 2 # first curriculum learning
    NUM_EXAMPLES = BATCH_SIZE * 4000
    
    
    # Wavelet settings
    WAVELET_FAMILY = 'db4'  # Daubechies wavelet with 4 vanishing moments

class Config: # 32 filters DS 16 US
    """Configuration class for parrotfish""" 
    # Data settings
    DATA_DIR = "data/audio"
    CLIPS_DIR = "data/clips"
    SAMPLE_RATE = 16000
    SEGMENT_LENGTH = 16000  # 1 second at 16kHz
    
    # Model settings
    NUM_COEFFS = 16000  # 1 second at 16kHz
    WAVELET_DEPTH = 5
    BATCH_SIZE = 16 # 16-32
    CHANNELS = 1  # Mono audio
    NUM_LAYERS = 11 # 10-12
    NUM_INIT_FILTERS = 16 ## was 24
    FILTER_SIZE = 16 # was 16 should be 16
    MERGE_FILTER_SIZE = 16 # was 5 should be like 8
    L1_REG = 0*1e-6
    L2_REG = 0 *1e-6
    
    # Training settings
    LEARNING_RATE = 1e-3
    EPOCHS = 100
    VAL_SPLIT = 0.1
    CHECKPOINT_DIR = "checkpoints"
    MAX_WORKERS = 4
    CACHE_REFRESH_EVERY = 5
    
    # Mixture generation
    MIN_SOURCES = 2
    MAX_SOURCES = 3 # first curriculum learning
    NUM_EXAMPLES = BATCH_SIZE * 5000
    
    # Wavelet settings
    WAVELET_FAMILY = 'db4'  # Daubechies wavelet with 4 vanishing moments
    
    
class RetrainConfig: # 32 filters DS 16 US
    """Configuration class for parrotfish""" 
    # Data settings
    DATA_DIR = "data/audio"
    CLIPS_DIR = "data/clips"
    SAMPLE_RATE = 16000
    SEGMENT_LENGTH = 16000  # 1 second at 16kHz
    
    # Model settings
    NUM_COEFFS = 16000  # 1 second at 16kHz
    WAVELET_DEPTH = 5
    BATCH_SIZE = 32 # 16-32
    CHANNELS = 1  # Mono audio
    NUM_LAYERS = 11 # 10-12
    NUM_INIT_FILTERS = 16 ## was 24
    FILTER_SIZE = 16 # was 16 should be 16
    MERGE_FILTER_SIZE = 16 # was 5 should be like 8
    L1_REG = 0*1e-6
    L2_REG = 0 *1e-6
    
    # Training settings
    LEARNING_RATE = 1e-4
    EPOCHS = 100
    VAL_SPLIT = 0.1
    CHECKPOINT_DIR = "checkpoints"
    MAX_WORKERS = 4
    CACHE_REFRESH_EVERY = 5
    
    # Mixture generation
    MIN_SOURCES = 2
    MAX_SOURCES = 3 # first curriculum learning
    NUM_EXAMPLES = BATCH_SIZE * 5000
    
    # Wavelet settings
    WAVELET_FAMILY = 'db4'  # Daubechies wavelet with 4 vanishing moments
    
    
# class RetrainConfig:
#     """Configuration class for parrotfish (retraining)""" 
#     # Data settings
#     DATA_DIR = "data/audio"
#     CLIPS_DIR = "data/clips"
#     SAMPLE_RATE = 16000
#     SEGMENT_LENGTH = 16000  
    
#     # Model settings
#     NUM_COEFFS = 16000  
#     WAVELET_DEPTH = 5
#     BATCH_SIZE = 32 # INCREASE TO 32
#     CHANNELS = 1  
#     NUM_LAYERS = 11 
#     NUM_INIT_FILTERS = 32 
#     FILTER_SIZE = 32 
#     MERGE_FILTER_SIZE = 16 
#     L1_REG = 0*1e-6
#     L2_REG = 0 *1e-6
    
#     # Training settings
#     LEARNING_RATE = 1e-4 # DECREASE TO 1e-5
#     EPOCHS = 100
#     VAL_SPLIT = 0.1
#     CHECKPOINT_DIR = "checkpoints"
#     MAX_WORKERS = 4
#     CACHE_REFRESH_EVERY = 5
    
#     # Mixture generation
#     MIN_SOURCES = 2
#     MAX_SOURCES = 3 # first curriculum learning
#     NUM_EXAMPLES = BATCH_SIZE * 5000
    
#     # Wavelet settings
#     WAVELET_FAMILY = 'db4'  # Daubechies wavelet with 4 vanishing moments
    
class RetrainConfig_flipped:
    """Configuration class for parrotfish (retraining)""" 
    # Data settings
    DATA_DIR = "data/audio"
    CLIPS_DIR = "data/clips"
    SAMPLE_RATE = 16000
    SEGMENT_LENGTH = 16000  
    
    # Model settings
    NUM_COEFFS = 16000  
    WAVELET_DEPTH = 5
    BATCH_SIZE = 32 # INCREASE TO 32
    CHANNELS = 1  
    NUM_LAYERS = 11 
    NUM_INIT_FILTERS = 32 
    FILTER_SIZE = 16 
    MERGE_FILTER_SIZE = 32 
    L1_REG = 0*1e-6
    L2_REG = 0 *1e-6
    
    # Training settings
    LEARNING_RATE = 1e-5 # DECREASE TO 1e-5
    EPOCHS = 100
    VAL_SPLIT = 0.1
    CHECKPOINT_DIR = "checkpoints"
    MAX_WORKERS = 4
    CACHE_REFRESH_EVERY = 5
    
    # Mixture generation
    MIN_SOURCES = 2
    MAX_SOURCES = 2 # first curriculum learning
    NUM_EXAMPLES = BATCH_SIZE * 4000
    
    # Wavelet settings
    WAVELET_FAMILY = 'db4'  # Daubechies wavelet with 4 vanishing moments