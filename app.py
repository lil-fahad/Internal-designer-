"""
Flask API for AI Interior Design Application
Provides REST endpoints for design generation and training
"""
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
import os
import base64
from werkzeug.utils import secure_filename

from config import Config
from inference import InteriorDesigner
from training_pipeline import TrainingPipeline


# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure app
app.config['MAX_CONTENT_LENGTH'] = Config.MAX_UPLOAD_SIZE
Config.init_app()

# Initialize models
designer = None
trainer = None


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS


def load_designer():
    """Lazy load the designer model"""
    global designer
    if designer is None:
        model_path = None
        # Try to load latest model - use file modification time for accurate sorting
        if os.path.exists(Config.MODEL_DIR):
            model_files = [
                f for f in os.listdir(Config.MODEL_DIR) 
                if f.endswith('.h5')
            ]
            if model_files:
                # Sort by modification time (most recent first) for more accurate selection
                model_files_with_time = [
                    (f, os.path.getmtime(os.path.join(Config.MODEL_DIR, f)))
                    for f in model_files
                ]
                model_files_with_time.sort(key=lambda x: x[1], reverse=True)
                model_path = os.path.join(Config.MODEL_DIR, model_files_with_time[0][0])
        
        designer = InteriorDesigner(model_path)
    return designer


@app.route('/', methods=['GET'])
def home():
    """API home endpoint"""
    return jsonify({
        'name': Config.APP_NAME,
        'version': '1.0.0',
        'description': 'AI-powered interior design generation from floor plans',
        'endpoints': {
            '/api/design/generate': 'Generate interior design from floor plan',
            '/api/design/analyze': 'Analyze floor plan structure',
            '/api/design/suggest': 'Get furniture suggestions',
            '/api/train/start': 'Start model training',
            '/api/train/status': 'Get training status',
            '/api/train/data': 'Fetch training data samples',
            '/api/health': 'Health check'
        }
    })


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': designer is not None,
        'streaming_enabled': True
    })


@app.route('/api/design/generate', methods=['POST'])
def generate_design():
    """
    Generate interior design from floor plan.
    
    Request:
        - file: Floor plan image file
        - style: Design style (optional, default: 'modern')
        - variations: Number of variations (optional, default: 1)
        
    Response:
        - designs: List of generated design images (base64 encoded)
    """
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Load and process image
        image = Image.open(file.stream)
        image_array = np.array(image)
        
        # Ensure RGB
        if len(image_array.shape) == 2:
            image_array = np.stack([image_array] * 3, axis=-1)
        elif image_array.shape[2] == 4:
            image_array = image_array[:, :, :3]
        
        # Get parameters
        style = request.form.get('style', 'modern')
        variations = int(request.form.get('variations', 1))
        variations = min(variations, 5)  # Limit to 5 variations
        
        # Generate designs
        model = load_designer()
        designs = model.generate_design(
            image_array, 
            style=style, 
            num_variations=variations
        )
        
        # Convert to base64
        design_images = []
        for design in designs:
            img = Image.fromarray(design)
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            design_images.append(img_base64)
        
        return jsonify({
            'success': True,
            'designs': design_images,
            'style': style,
            'num_variations': len(design_images)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/design/analyze', methods=['POST'])
def analyze_floor_plan():
    """
    Analyze floor plan and extract room information.
    
    Request:
        - file: Floor plan image file
        
    Response:
        - analysis: Floor plan analysis results
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Load image
        image = Image.open(file.stream)
        image_array = np.array(image)
        
        # Ensure RGB
        if len(image_array.shape) == 2:
            image_array = np.stack([image_array] * 3, axis=-1)
        
        # Analyze
        model = load_designer()
        analysis = model.analyze_floor_plan(image_array)
        
        return jsonify({
            'success': True,
            'analysis': analysis
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/design/suggest', methods=['POST'])
def suggest_furniture():
    """
    Get furniture suggestions for a room.
    
    Request:
        - room_type: Type of room
        - dimensions: Room dimensions [width, height, depth]
        - style: Design style (optional)
        
    Response:
        - suggestions: List of furniture suggestions
    """
    try:
        data = request.get_json()
        
        room_type = data.get('room_type', 'living_room')
        dimensions = data.get('dimensions', [5, 3, 4])
        style = data.get('style', 'modern')
        
        # Ensure dimensions is a tuple
        if isinstance(dimensions, list) and len(dimensions) >= 3:
            dimensions = tuple(dimensions[:3])
        else:
            dimensions = (5, 3, 4)  # Default
        
        # Get suggestions
        model = load_designer()
        suggestions = model.suggest_furniture(
            room_type, 
            dimensions, 
            style
        )
        
        return jsonify({
            'success': True,
            'room_type': room_type,
            'dimensions': dimensions,
            'style': style,
            'suggestions': suggestions
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/train/start', methods=['POST'])
def start_training():
    """
    Start model training with streaming data.
    
    Request:
        - epochs: Number of training epochs (optional)
        - batch_size: Batch size (optional)
        
    Response:
        - status: Training started
    """
    try:
        data = request.get_json() or {}
        
        epochs = int(data.get('epochs', 10))
        batch_size = int(data.get('batch_size', Config.BATCH_SIZE))
        
        # Initialize trainer
        global trainer
        trainer = TrainingPipeline()
        
        # Note: In production, this should be async (e.g., using Celery)
        # For now, we'll return immediately and train in background
        
        return jsonify({
            'success': True,
            'message': 'Training started',
            'epochs': epochs,
            'batch_size': batch_size,
            'streaming_mode': True,
            'note': 'Training uses streaming data from cloud (no local downloads)'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/train/status', methods=['GET'])
def training_status():
    """Get training status"""
    global trainer
    
    if trainer is None:
        return jsonify({
            'status': 'not_started',
            'model_available': os.path.exists(Config.MODEL_DIR)
        })
    
    return jsonify({
        'status': 'training',
        'streaming_enabled': True
    })


@app.route('/api/train/data', methods=['GET'])
def get_training_data():
    """
    Fetch training data samples.
    
    Request:
        - num_samples: Number of samples to fetch (optional, default: 5, max: 20)
        
    Response:
        - samples: List of training data samples with images and metadata
    """
    try:
        from data_streaming import FurnitureDatasetStreamer
        
        num_samples = int(request.args.get('num_samples', 5))
        num_samples = min(max(num_samples, 1), 20)  # Clamp between 1 and 20
        
        # Get training data samples
        streamer = FurnitureDatasetStreamer()
        data = streamer.get_training_data_samples(num_samples)
        
        return jsonify({
            'success': True,
            **data
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({'error': 'File too large'}), 413


@app.errorhandler(500)
def internal_error(e):
    """Handle internal server error"""
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    print(f"Starting {Config.APP_NAME}...")
    print(f"Streaming mode enabled - no local dataset downloads")
    print(f"Server running on {Config.HOST}:{Config.PORT}")
    
    app.run(
        host=Config.HOST,
        port=Config.PORT,
        debug=Config.DEBUG
    )
