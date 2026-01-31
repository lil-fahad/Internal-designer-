"""
Configuration settings for the AI Interior Design Application
"""
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Base configuration"""
    
    # Application
    APP_NAME = "AI Interior Designer"
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 5000))
    
    # ML Model Configuration
    MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models', 'saved')
    CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), 'models', 'checkpoints')
    
    # Training Configuration
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', 32))
    EPOCHS = int(os.getenv('EPOCHS', 50))
    LEARNING_RATE = float(os.getenv('LEARNING_RATE', 0.0002))
    IMAGE_SIZE = (256, 256)
    
    # Dataset Configuration (Cloud URLs for streaming)
    FURNITURE_DATASET_URLS = [
        # Public furniture datasets (examples - can be replaced with actual public datasets)
        'https://storage.googleapis.com/tfds-data/dataset_info/furniture_detection/1.0.0/',
        'https://huggingface.co/datasets/furniture',
    ]
    
    # Cache settings (for streaming optimization, not full downloads)
    CACHE_DIR = os.path.join(os.path.dirname(__file__), 'data', 'cache')
    STREAM_BUFFER_SIZE = 1024 * 1024  # 1MB
    
    # API Configuration
    MAX_UPLOAD_SIZE = 16 * 1024 * 1024  # 16MB
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}
    
    @staticmethod
    def init_app():
        """Initialize application directories"""
        os.makedirs(Config.MODEL_DIR, exist_ok=True)
        os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
