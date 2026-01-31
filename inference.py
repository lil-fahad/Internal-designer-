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
        
        # Add batch dimension
        floor_plan_batch = tf.expand_dims(preprocessed, 0)
        
        designs = []
        for _ in range(num_variations):
            # Generate design
            generated = self.generator(floor_plan_batch, training=False)
            
            # Post-process (denormalize from [-1, 1] to [0, 255])
            design = (generated[0].numpy() + 1.0) * 127.5
            design = np.clip(design, 0, 255).astype(np.uint8)
            
            designs.append(design)
        
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
        
        room_designs = []
        for idx, room_data in enumerate(rooms):
            room_img = room_data['image']
            
            # Resize to standard size
            room_resized = tf.image.resize(
                room_img, Config.IMAGE_SIZE
            ).numpy().astype(np.uint8)
            
            # Generate design for this room
            designs = self.generate_design(
                room_resized, 
                style=style, 
                num_variations=1
            )
            
            room_designs.append({
                'room_id': idx,
                'bbox': room_data['bbox'],
                'area': room_data['area'],
                'design': designs[0],
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
        furniture_suggestions = {
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
        
        # Get base suggestions
        suggestions = furniture_suggestions.get(
            room_type, 
            furniture_suggestions['living_room']
        )
        
        # Adjust based on room size
        width, height, depth = room_dimensions
        area = width * depth
        
        # Filter suggestions based on room size
        filtered = []
        for item in suggestions:
            # Simple heuristic: larger rooms can fit more furniture
            if area > 20:  # Large room
                filtered.append(item)
            elif area > 10:  # Medium room
                if item.get('size') != 'large':
                    filtered.append(item)
            else:  # Small room
                if item.get('size') in ['small', 'medium', None]:
                    filtered.append(item)
        
        # Add style information
        for item in filtered:
            item['style'] = style
        
        return filtered
    
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
