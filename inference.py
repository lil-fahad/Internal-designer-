"""
Interior Design Inference Module
Handles design generation from floor plans
"""
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from typing import Optional, Dict, Any

from config import Config
from training_pipeline import InteriorDesignGenerator
from data_streaming import FloorPlanDataStreamer


class InteriorDesigner:
    """
    Main inference class for generating interior designs from floor plans.
    """
    
    # Class-level constant for furniture suggestions - avoids recreating on every call
    FURNITURE_SUGGESTIONS = {
        'bedroom': [
            {'item': 'bed', 'size': 'queen', 'position': 'center'},
            {'item': 'nightstand', 'quantity': 2, 'position': 'bedside'},
            {'item': 'wardrobe', 'size': 'large', 'position': 'wall'},
            {'item': 'desk', 'size': 'small', 'position': 'corner'}
        ],
        'living_room': [
            {'item': 'sofa', 'size': 'large', 'position': 'center'},
            {'item': 'coffee_table', 'size': 'medium', 'position': 'front'},
            {'item': 'tv_stand', 'size': 'medium', 'position': 'wall'},
            {'item': 'armchair', 'quantity': 2, 'position': 'side'}
        ],
        'kitchen': [
            {'item': 'dining_table', 'size': 'medium', 'position': 'center'},
            {'item': 'chair', 'quantity': 4, 'position': 'table'},
            {'item': 'cabinet', 'size': 'large', 'position': 'wall'},
            {'item': 'counter', 'size': 'long', 'position': 'wall'}
        ],
        'office': [
            {'item': 'desk', 'size': 'large', 'position': 'center'},
            {'item': 'office_chair', 'quantity': 1, 'position': 'desk'},
            {'item': 'bookshelf', 'size': 'tall', 'position': 'wall'},
            {'item': 'filing_cabinet', 'size': 'medium', 'position': 'corner'}
        ]
    }
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the interior designer.
        
        Args:
            model_path: Path to pre-trained model weights
        """
        self.generator = InteriorDesignGenerator()
        self.floor_plan_streamer = FloorPlanDataStreamer()
        
        # Build model by calling it once
        dummy_input = tf.random.normal((1, *Config.IMAGE_SIZE, 3))
        _ = self.generator(dummy_input)
        
        # Load pre-trained weights if available
        if model_path:
            try:
                self.generator.load_weights(model_path)
                print(f"Loaded model from {model_path}")
            except Exception as e:
                print(f"Could not load model: {e}")
                print("Using untrained model (for demonstration)")
    
    def generate_design(
        self, 
        floor_plan_image: np.ndarray,
        style: str = 'modern',
        num_variations: int = 1
    ) -> list:
        """
        Generate interior design from floor plan.
        
        Args:
            floor_plan_image: Input floor plan as numpy array
            style: Design style (modern, classic, minimalist, rustic)
            num_variations: Number of design variations to generate
            
        Returns:
            list: Generated design images
        """
        # Preprocess floor plan
        preprocessed = self.floor_plan_streamer.preprocess_floor_plan(
            floor_plan_image
        )
        
        # Batch all variations together for more efficient inference
        # Tile the preprocessed input for num_variations
        floor_plan_batch = tf.tile(
            tf.expand_dims(preprocessed, 0), 
            [num_variations, 1, 1, 1]
        )
        
        # Generate all designs in a single forward pass
        generated = self.generator(floor_plan_batch, training=False)
        
        # Post-process all designs at once (denormalize from [-1, 1] to [0, 255])
        designs_array = (generated.numpy() + 1.0) * 127.5
        designs_array = np.clip(designs_array, 0, 255).astype(np.uint8)
        
        # Convert to list of individual designs
        designs = [designs_array[i] for i in range(num_variations)]
        
        return designs
    
    def generate_room_designs(
        self, 
        floor_plan_image: np.ndarray,
        style: str = 'modern'
    ) -> Dict[str, Any]:
        """
        Generate designs for individual rooms in a floor plan.
        
        Args:
            floor_plan_image: Input floor plan as numpy array
            style: Design style
            
        Returns:
            dict: Room designs and metadata
        """
        # Extract rooms from floor plan
        rooms = self.floor_plan_streamer.extract_rooms(floor_plan_image)
        
        if not rooms:
            return {
                'num_rooms': 0,
                'room_designs': [],
                'style': style
            }
        
        # Batch process all rooms together for efficiency
        room_images = []
        for room_data in rooms:
            room_img = room_data['image']
            # Resize to standard size
            room_resized = tf.image.resize(
                room_img, Config.IMAGE_SIZE
            ).numpy().astype(np.uint8)
            room_images.append(room_resized)
        
        # Stack all rooms into a batch
        room_batch = np.stack(room_images, axis=0)
        
        # Preprocess the batch (normalize only, images already resized)
        room_batch_tensor = tf.cast(room_batch, tf.float32)
        room_batch_tensor = (room_batch_tensor / 127.5) - 1.0
        
        # Generate designs for all rooms in a single forward pass
        generated_batch = self.generator(room_batch_tensor, training=False)
        
        # Post-process all designs
        designs_array = (generated_batch.numpy() + 1.0) * 127.5
        designs_array = np.clip(designs_array, 0, 255).astype(np.uint8)
        
        # Build room designs list
        room_designs = []
        for idx, room_data in enumerate(rooms):
            room_designs.append({
                'room_id': idx,
                'bbox': room_data['bbox'],
                'area': room_data['area'],
                'design': designs_array[idx],
                'style': style
            })
        
        return {
            'num_rooms': len(rooms),
            'room_designs': room_designs,
            'style': style
        }
    
    def suggest_furniture(
        self, 
        room_type: str, 
        room_dimensions: tuple,
        style: str = 'modern'
    ) -> list:
        """
        Suggest furniture items for a room based on type and dimensions.
        
        Args:
            room_type: Type of room (bedroom, living_room, kitchen, etc.)
            room_dimensions: (width, height, depth) in meters
            style: Design style
            
        Returns:
            list: Suggested furniture items
        """
        # Get base suggestions from class-level constant (avoids dict recreation)
        base_suggestions = self.FURNITURE_SUGGESTIONS.get(
            room_type, 
            self.FURNITURE_SUGGESTIONS['living_room']
        )
        
        # Adjust based on room size
        width, height, depth = room_dimensions
        area = width * depth
        
        # Filter suggestions based on room size
        filtered = []
        for item in base_suggestions:
            # Simple heuristic: larger rooms can fit more furniture
            include = False
            if area > 20:  # Large room
                include = True
            elif area > 10:  # Medium room
                include = item.get('size') != 'large'
            else:  # Small room
                include = item.get('size') in ['small', 'medium', None]
            
            if include:
                filtered.append(item)
        
        # Add style to all filtered items
        return [{**item, 'style': style} for item in filtered]
    
    def analyze_floor_plan(
        self, 
        floor_plan_image: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze floor plan and extract key information.
        
        Args:
            floor_plan_image: Input floor plan as numpy array
            
        Returns:
            dict: Analysis results
        """
        # Extract rooms
        rooms = self.floor_plan_streamer.extract_rooms(floor_plan_image)
        
        # Calculate total area (approximate)
        total_area = sum(room['area'] for room in rooms)
        
        # Classify rooms based on size and position
        room_classifications = []
        for idx, room in enumerate(rooms):
            # Simple classification based on area
            area = room['area']
            if area > 5000:
                room_type = 'living_room'
            elif area > 3000:
                room_type = 'bedroom'
            elif area > 2000:
                room_type = 'kitchen'
            else:
                room_type = 'bathroom'
            
            room_classifications.append({
                'room_id': idx,
                'type': room_type,
                'area': area,
                'bbox': room['bbox']
            })
        
        return {
            'num_rooms': len(rooms),
            'total_area': total_area,
            'rooms': room_classifications,
            'dimensions': floor_plan_image.shape[:2]
        }
