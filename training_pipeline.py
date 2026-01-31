"""
ML Training Pipeline for Interior Design Model
Implements industry-standard training practices with streaming data
"""
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, optimizers, callbacks
import numpy as np
import os
from datetime import datetime
from typing import Optional, Tuple

from config import Config
from data_streaming import FurnitureDatasetStreamer, FloorPlanDataStreamer


class InteriorDesignGenerator(keras.Model):
    """
    Advanced neural network for generating interior designs from floor plans.
    Based on conditional GAN architecture (Pix2Pix style).
    """
    
    def __init__(self, image_size: Tuple[int, int] = Config.IMAGE_SIZE):
        super(InteriorDesignGenerator, self).__init__()
        self.image_size = image_size
        
        # Encoder (floor plan -> features)
        self.encoder = self._build_encoder()
        
        # Decoder (features + furniture -> design)
        self.decoder = self._build_decoder()
        
    def _build_encoder(self) -> keras.Model:
        """Build encoder network for floor plan feature extraction"""
        inputs = layers.Input(shape=(*self.image_size, 3))
        
        # Downsampling path
        x = layers.Conv2D(64, 4, strides=2, padding='same')(inputs)
        x = layers.LeakyReLU(0.2)(x)
        
        x = layers.Conv2D(128, 4, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        
        x = layers.Conv2D(256, 4, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        
        x = layers.Conv2D(512, 4, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        
        # Bottleneck
        x = layers.Conv2D(512, 4, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        
        return keras.Model(inputs, x, name='encoder')
    
    def _build_decoder(self) -> keras.Model:
        """Build decoder network for design generation"""
        inputs = layers.Input(shape=(8, 8, 512))  # Bottleneck size
        
        # Upsampling path
        x = layers.Conv2DTranspose(512, 4, strides=2, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.ReLU()(x)
        
        x = layers.Conv2DTranspose(256, 4, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.ReLU()(x)
        
        x = layers.Conv2DTranspose(128, 4, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        x = layers.Conv2DTranspose(64, 4, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        x = layers.Conv2DTranspose(32, 4, strides=2, padding='same')(x)
        x = layers.ReLU()(x)
        
        # Output layer (RGB image)
        outputs = layers.Conv2D(3, 4, padding='same', activation='tanh')(x)
        
        return keras.Model(inputs, outputs, name='decoder')
    
    def call(self, inputs):
        """Forward pass"""
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded


class DesignDiscriminator(keras.Model):
    """
    Discriminator network to evaluate design quality.
    Used during training to ensure realistic outputs.
    """
    
    def __init__(self, image_size: Tuple[int, int] = Config.IMAGE_SIZE):
        super(DesignDiscriminator, self).__init__()
        self.image_size = image_size
        self.model = self._build_model()
    
    def _build_model(self) -> keras.Model:
        """Build discriminator network"""
        # Concatenate floor plan and design
        floor_plan = layers.Input(shape=(*self.image_size, 3))
        design = layers.Input(shape=(*self.image_size, 3))
        
        combined = layers.Concatenate()([floor_plan, design])
        
        # Convolutional layers
        x = layers.Conv2D(64, 4, strides=2, padding='same')(combined)
        x = layers.LeakyReLU(0.2)(x)
        
        x = layers.Conv2D(128, 4, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        
        x = layers.Conv2D(256, 4, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        
        # Patch output
        x = layers.Conv2D(512, 4, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        
        output = layers.Conv2D(1, 4, padding='same')(x)
        
        return keras.Model([floor_plan, design], output, name='discriminator')
    
    def call(self, inputs):
        """Forward pass"""
        return self.model(inputs)


class TrainingPipeline:
    """
    Industry-standard ML training pipeline with streaming data support.
    Implements best practices for model training and evaluation.
    """
    
    def __init__(self):
        self.generator = InteriorDesignGenerator()
        self.discriminator = DesignDiscriminator()
        self.data_streamer = FurnitureDatasetStreamer()
        
        # Optimizers
        self.gen_optimizer = optimizers.Adam(
            learning_rate=Config.LEARNING_RATE, beta_1=0.5
        )
        self.disc_optimizer = optimizers.Adam(
            learning_rate=Config.LEARNING_RATE, beta_1=0.5
        )
        
        # Loss functions
        self.bce_loss = keras.losses.BinaryCrossentropy(from_logits=True)
        self.mae_loss = keras.losses.MeanAbsoluteError()
        
        # Metrics
        self.gen_loss_metric = keras.metrics.Mean(name='gen_loss')
        self.disc_loss_metric = keras.metrics.Mean(name='disc_loss')
        
    def discriminator_loss(self, real_output, fake_output):
        """Calculate discriminator loss"""
        real_loss = self.bce_loss(
            tf.ones_like(real_output), real_output
        )
        fake_loss = self.bce_loss(
            tf.zeros_like(fake_output), fake_output
        )
        return real_loss + fake_loss
    
    def generator_loss(self, fake_output, generated_images, target_images):
        """Calculate generator loss (adversarial + L1)"""
        # Adversarial loss
        gan_loss = self.bce_loss(
            tf.ones_like(fake_output), fake_output
        )
        
        # L1 loss (pixel-wise similarity)
        l1_loss = self.mae_loss(target_images, generated_images)
        
        # Combined loss (weighted)
        total_loss = gan_loss + (100 * l1_loss)
        return total_loss
    
    @tf.function
    def train_step(self, floor_plan, target_design):
        """Single training step"""
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generate design
            generated_design = self.generator(floor_plan, training=True)
            
            # Discriminator outputs
            real_output = self.discriminator(
                [floor_plan, target_design], training=True
            )
            fake_output = self.discriminator(
                [floor_plan, generated_design], training=True
            )
            
            # Calculate losses
            gen_loss = self.generator_loss(
                fake_output, generated_design, target_design
            )
            disc_loss = self.discriminator_loss(real_output, fake_output)
        
        # Calculate gradients
        gen_gradients = gen_tape.gradient(
            gen_loss, self.generator.trainable_variables
        )
        disc_gradients = disc_tape.gradient(
            disc_loss, self.discriminator.trainable_variables
        )
        
        # Apply gradients
        self.gen_optimizer.apply_gradients(
            zip(gen_gradients, self.generator.trainable_variables)
        )
        self.disc_optimizer.apply_gradients(
            zip(disc_gradients, self.discriminator.trainable_variables)
        )
        
        # Update metrics
        self.gen_loss_metric.update_state(gen_loss)
        self.disc_loss_metric.update_state(disc_loss)
        
        return {'gen_loss': gen_loss, 'disc_loss': disc_loss}
    
    def train(self, epochs: int = Config.EPOCHS, 
              save_freq: int = 5) -> dict:
        """
        Train the model using streaming data pipeline.
        
        Args:
            epochs: Number of training epochs
            save_freq: Save checkpoint every N epochs
            
        Returns:
            dict: Training history
        """
        print(f"Starting training for {epochs} epochs...")
        print(f"Using streaming data pipeline (no local downloads)")
        
        # Get streaming dataset
        dataset = self.data_streamer.create_training_pipeline()
        
        # Training history
        history = {
            'gen_loss': [],
            'disc_loss': []
        }
        
        # Callbacks
        checkpoint_callback = callbacks.ModelCheckpoint(
            filepath=os.path.join(
                Config.CHECKPOINT_DIR, 
                'generator_epoch_{epoch:02d}.h5'
            ),
            save_freq='epoch',
            period=save_freq
        )
        
        # Training loop
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Reset metrics
            self.gen_loss_metric.reset_states()
            self.disc_loss_metric.reset_states()
            
            # Process batches from stream
            for batch_idx, (furniture_images, metadata) in enumerate(dataset.take(100)):
                # Get actual batch size from the furniture images
                actual_batch_size = tf.shape(furniture_images)[0]
                
                # Create synthetic floor plans for training
                # In production, use real floor plan dataset
                floor_plans = tf.random.normal(
                    shape=(actual_batch_size, *Config.IMAGE_SIZE, 3)
                )
                
                # Train step
                losses = self.train_step(floor_plans, furniture_images)
                
                if batch_idx % 10 == 0:
                    print(
                        f"Batch {batch_idx}: "
                        f"Gen Loss = {losses['gen_loss']:.4f}, "
                        f"Disc Loss = {losses['disc_loss']:.4f}"
                    )
            
            # Record epoch metrics
            epoch_gen_loss = self.gen_loss_metric.result()
            epoch_disc_loss = self.disc_loss_metric.result()
            
            history['gen_loss'].append(float(epoch_gen_loss))
            history['disc_loss'].append(float(epoch_disc_loss))
            
            print(
                f"Epoch {epoch + 1} - "
                f"Gen Loss: {epoch_gen_loss:.4f}, "
                f"Disc Loss: {epoch_disc_loss:.4f}"
            )
            
            # Save checkpoint
            if (epoch + 1) % save_freq == 0:
                self.save_model(epoch + 1)
        
        print("\nTraining completed!")
        return history
    
    def save_model(self, epoch: Optional[int] = None):
        """Save trained model"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        epoch_str = f"_epoch{epoch}" if epoch else ""
        
        model_path = os.path.join(
            Config.MODEL_DIR, 
            f'generator{epoch_str}_{timestamp}.h5'
        )
        
        self.generator.save_weights(model_path)
        print(f"Model saved to {model_path}")
        
        return model_path
    
    def load_model(self, model_path: str):
        """Load pre-trained model"""
        self.generator.load_weights(model_path)
        print(f"Model loaded from {model_path}")
