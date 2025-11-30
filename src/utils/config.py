class Config: 
    """Configuration class for first training""" 
    # Data settings
    DATA_DIR = "data/audio"
    CLIPS_DIR = "data/clips"
    SAMPLE_RATE = 44100
    SEGMENT_LENGTH = 44100  # 1 second at 44.1kHz
    
    # Model settings
    NUM_COEFFS = 44100  # 1 second at 44.1kHz
    WAVELET_DEPTH = 5
    BATCH_SIZE = 16 # 16-32
    CHANNELS = 1  # Mono audio
    NUM_LAYERS = 10 # 10-12
    NUM_INIT_FILTERS = 16 
    FILTER_SIZE = 16 
    MERGE_FILTER_SIZE = 16 
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
    MIN_SOURCES = 1 # TOO LAZY TO CHANGE FOR UPSCALEMP3_V2 
                    # FOR THE LOVE OF GOD DONT CHANGE TO ANYTHING OTHER THAN 1
    MAX_SOURCES = 1 # first curriculum learning
    NUM_EXAMPLES = BATCH_SIZE * 3600
    
    
    # Wavelet settings
    WAVELET_FAMILY = 'db4'  # Daubechies wavelet with 4 vanishing moments
    


    
class RetrainConfig: # 32 filters DS 16 US
    """Configuration class for parrotfish""" 
    # Data settings
    DATA_DIR = "data/audio"
    CLIPS_DIR = "data/clips"
    SAMPLE_RATE = 44100
    SEGMENT_LENGTH = 44100  
    
    # Model settings
    NUM_COEFFS = 44100  
    WAVELET_DEPTH = 5
    BATCH_SIZE = 16 
    CHANNELS = 1  
    NUM_LAYERS = 10 
    NUM_INIT_FILTERS = 16 
    FILTER_SIZE = 16 
    MERGE_FILTER_SIZE = 16 
    L1_REG = 0*1e-6
    L2_REG = 0 *1e-6
    
    # Training settings
    LEARNING_RATE = 1e-4 #lower initial lr for retraining
    EPOCHS = 100
    VAL_SPLIT = 0.1
    CHECKPOINT_DIR = "checkpoints"
    MAX_WORKERS = 4
    CACHE_REFRESH_EVERY = 5
    
    # Mixture generation
    MIN_SOURCES = 1
    MAX_SOURCES = 1 
    NUM_EXAMPLES = BATCH_SIZE * 4800 # use all data for retraining
    
    
    # Wavelet settings
    WAVELET_FAMILY = 'db4'  # Daubechies wavelet with 4 vanishing moments
    
    