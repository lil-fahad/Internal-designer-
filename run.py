#!/usr/bin/env python3
"""
Startup script for AI Interior Designer
Provides a simple entry point for running the application
"""
import os
import sys

def main():
    """Run the application"""
    print("=" * 60)
    print("AI Interior Designer - Starting Application")
    print("=" * 60)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"Python version: {sys.version}")
    
    # Check for required modules
    try:
        import tensorflow
        import flask
        from importlib.metadata import version
        print(f"TensorFlow version: {tensorflow.__version__}")
        print(f"Flask version: {version('flask')}")
    except ImportError as e:
        print(f"\nError: Missing required dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        sys.exit(1)
    
    # Set environment defaults
    os.environ.setdefault('DEBUG', 'False')
    os.environ.setdefault('HOST', '0.0.0.0')
    os.environ.setdefault('PORT', '5000')
    
    # Import and initialize app
    from config import Config
    Config.init_app()
    
    print(f"\nConfiguration:")
    print(f"  Debug: {Config.DEBUG}")
    print(f"  Host: {Config.HOST}")
    print(f"  Port: {Config.PORT}")
    print(f"  Batch Size: {Config.BATCH_SIZE}")
    print(f"  Image Size: {Config.IMAGE_SIZE}")
    
    print("\n" + "=" * 60)
    print("Starting Flask server...")
    print("=" * 60)
    
    # Run the app
    from app import app
    app.run(
        host=Config.HOST,
        port=Config.PORT,
        debug=Config.DEBUG
    )


if __name__ == '__main__':
    main()
