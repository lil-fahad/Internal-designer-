"""
Example: Train the AI Interior Design model with streaming data
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training_pipeline import TrainingPipeline
from config import Config


def main():
    """Train the model"""
    print("="*60)
    print("AI Interior Designer - Model Training")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Epochs: {Config.EPOCHS}")
    print(f"  Batch Size: {Config.BATCH_SIZE}")
    print(f"  Learning Rate: {Config.LEARNING_RATE}")
    print(f"  Image Size: {Config.IMAGE_SIZE}")
    print(f"  Streaming Mode: Enabled (no local downloads)")
    print("\n" + "="*60)
    
    # Initialize training pipeline
    print("\nInitializing training pipeline...")
    pipeline = TrainingPipeline()
    
    # Start training
    print("\nStarting training with streaming data...")
    print("Note: Data is streamed from cloud sources without local downloads\n")
    
    try:
        history = pipeline.train(epochs=Config.EPOCHS, save_freq=5)
        
        # Print summary
        print("\n" + "="*60)
        print("Training Completed!")
        print("="*60)
        print(f"\nFinal Metrics:")
        print(f"  Generator Loss: {history['gen_loss'][-1]:.4f}")
        print(f"  Discriminator Loss: {history['disc_loss'][-1]:.4f}")
        
        # Save final model
        print("\nSaving final model...")
        model_path = pipeline.save_model()
        print(f"Model saved to: {model_path}")
        
        print("\n✓ Training pipeline completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Saving current model state...")
        pipeline.save_model()
        print("Model saved!")
    
    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
