# AI Interior Designer

A professional AI-powered interior design application that generates interior designs from building floor plans using advanced machine learning models with cloud-based streaming datasets.

## Features

- 🎨 **AI-Powered Design Generation**: Generate interior designs from floor plans using state-of-the-art neural networks
- ☁️ **Cloud Streaming**: Train models using cloud-hosted furniture datasets in streaming mode (no local downloads)
- 🏗️ **Industry-Standard ML Pipeline**: Professional training pipeline with best practices
- 🚀 **REST API**: Easy-to-use API endpoints for integration
- 📊 **Floor Plan Analysis**: Automatic room detection and classification
- 🪑 **Furniture Suggestions**: Smart furniture recommendations based on room type and dimensions
- 🎭 **Multiple Styles**: Support for modern, classic, minimalist, and rustic design styles

## Architecture

The application follows industry-standard ML architecture:

```
├── app.py                    # Flask REST API
├── config.py                 # Configuration management
├── data_streaming.py         # Cloud dataset streaming module
├── training_pipeline.py      # ML training pipeline
├── inference.py              # Design generation inference
└── requirements.txt          # Python dependencies
```

### ML Model Architecture

- **Generator**: Conditional GAN (Pix2Pix-style) for floor plan → design transformation
- **Discriminator**: PatchGAN discriminator for quality assessment
- **Streaming Pipeline**: TensorFlow data pipeline with prefetching and batching
- **Loss Functions**: Combined adversarial loss + L1 pixel loss

## Installation

### Prerequisites

- Python 3.8+
- pip
- Virtual environment (recommended)

### Quick Setup (Recommended)

```bash
git clone https://github.com/lil-fahad/Internal-designer-.git
cd Internal-designer-
./scripts/setup.sh
```

### Manual Setup

1. Clone the repository:
```bash
git clone https://github.com/lil-fahad/Internal-designer-.git
cd Internal-designer-
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create environment file (optional):
```bash
cp .env.example .env
# Edit .env with your settings
```

### Docker Setup

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build manually
docker build -t interior-designer .
docker run -p 5000:5000 interior-designer
```

## Usage

### Starting the API Server

**Development Mode:**
```bash
python run.py
# or
./scripts/start.sh dev
```

**Production Mode (Gunicorn):**
```bash
gunicorn wsgi:app --bind 0.0.0.0:5000 --workers 4
# or
./scripts/start.sh prod
```

**Docker Mode:**
```bash
docker-compose up
# or
./scripts/start.sh docker
```

The server will start on `http://localhost:5000`

### API Endpoints

#### 1. Generate Interior Design

```bash
POST /api/design/generate
```

**Request:**
- `file`: Floor plan image (PNG, JPG, JPEG, PDF)
- `style`: Design style (optional, default: 'modern')
  - Options: modern, classic, minimalist, rustic
- `variations`: Number of design variations (optional, default: 1, max: 5)

**Example using curl:**
```bash
curl -X POST http://localhost:5000/api/design/generate \
  -F "file=@floor_plan.png" \
  -F "style=modern" \
  -F "variations=3"
```

**Response:**
```json
{
  "success": true,
  "designs": ["base64_encoded_image_1", "base64_encoded_image_2", ...],
  "style": "modern",
  "num_variations": 3
}
```

#### 2. Analyze Floor Plan

```bash
POST /api/design/analyze
```

**Request:**
- `file`: Floor plan image

**Example:**
```bash
curl -X POST http://localhost:5000/api/design/analyze \
  -F "file=@floor_plan.png"
```

**Response:**
```json
{
  "success": true,
  "analysis": {
    "num_rooms": 4,
    "total_area": 15000,
    "rooms": [
      {
        "room_id": 0,
        "type": "living_room",
        "area": 5500,
        "bbox": [10, 20, 100, 150]
      },
      ...
    ],
    "dimensions": [1024, 768]
  }
}
```

#### 3. Get Furniture Suggestions

```bash
POST /api/design/suggest
```

**Request Body (JSON):**
```json
{
  "room_type": "bedroom",
  "dimensions": [5, 3, 4],
  "style": "modern"
}
```

**Response:**
```json
{
  "success": true,
  "room_type": "bedroom",
  "dimensions": [5, 3, 4],
  "style": "modern",
  "suggestions": [
    {
      "item": "bed",
      "size": "queen",
      "position": "center",
      "style": "modern"
    },
    ...
  ]
}
```

#### 4. Start Model Training

```bash
POST /api/train/start
```

**Request Body (JSON):**
```json
{
  "epochs": 50,
  "batch_size": 32
}
```

**Response:**
```json
{
  "success": true,
  "message": "Training started",
  "epochs": 50,
  "batch_size": 32,
  "streaming_mode": true,
  "note": "Training uses streaming data from cloud (no local downloads)"
}
```

#### 5. Health Check

```bash
GET /api/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "streaming_enabled": true
}
```

## Training Your Own Model

The application uses streaming data from cloud-hosted furniture datasets, eliminating the need for large local downloads.

### Quick Training

```python
from training_pipeline import TrainingPipeline

# Initialize pipeline
pipeline = TrainingPipeline()

# Train with streaming data (no downloads)
history = pipeline.train(epochs=50)

# Save model
pipeline.save_model()
```

### Streaming Data Pipeline

The data streaming module (`data_streaming.py`) implements:

- **TensorFlow Datasets Integration**: Streams from TFDS cloud storage
- **HTTP Streaming**: Streams from public URLs using chunked transfer
- **Memory Efficiency**: Processes data in batches without full downloads
- **Prefetching**: Overlaps data loading and model training
- **Augmentation**: Real-time data augmentation during training

### Custom Dataset URLs

You can configure custom furniture dataset URLs in `config.py`:

```python
FURNITURE_DATASET_URLS = [
    'https://your-cloud-storage.com/furniture-dataset',
    'https://public-dataset-repository.org/furniture',
]
```

## Design Generation Process

1. **Input**: Upload a floor plan image
2. **Preprocessing**: Floor plan is analyzed and normalized
3. **Room Detection**: Automatic room boundary detection
4. **Design Generation**: Neural network generates interior design
5. **Post-processing**: Image enhancement and style application
6. **Output**: Generated design image(s)

## ML Pipeline Best Practices

This application implements industry-standard practices:

✅ **Streaming Data**: No local dataset downloads, memory-efficient  
✅ **Modular Architecture**: Separation of concerns (data, training, inference)  
✅ **Checkpointing**: Automatic model checkpointing during training  
✅ **Metrics Tracking**: Loss and performance metrics  
✅ **Preprocessing Pipeline**: Standardized image preprocessing  
✅ **Batch Processing**: Efficient batch-based training  
✅ **GPU Support**: Automatic GPU detection and usage  
✅ **Error Handling**: Robust error handling throughout  

## Configuration

Edit `config.py` or use environment variables:

```bash
# Application
DEBUG=False
HOST=0.0.0.0
PORT=5000

# Training
BATCH_SIZE=32
EPOCHS=50
LEARNING_RATE=0.0002

# API
MAX_UPLOAD_SIZE=16777216  # 16MB
```

## API Integration Examples

### Python

```python
import requests

# Generate design
with open('floor_plan.png', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/api/design/generate',
        files={'file': f},
        data={'style': 'modern', 'variations': 3}
    )
    
result = response.json()
designs = result['designs']
```

### JavaScript

```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('style', 'modern');
formData.append('variations', 3);

fetch('http://localhost:5000/api/design/generate', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => {
    console.log('Generated designs:', data.designs);
});
```

### cURL

```bash
curl -X POST http://localhost:5000/api/design/generate \
  -F "file=@floor_plan.png" \
  -F "style=modern" \
  -F "variations=3" \
  -o response.json
```

## Project Structure

```
Internal-designer-/
├── app.py                    # Main Flask application
├── config.py                 # Configuration settings
├── data_streaming.py         # Cloud data streaming module
├── training_pipeline.py      # ML training pipeline
├── inference.py              # Inference and design generation
├── run.py                    # Application startup script
├── wsgi.py                   # WSGI entry point for production
├── requirements.txt          # Python dependencies
├── Dockerfile                # Docker image definition
├── docker-compose.yml        # Docker Compose configuration
├── Procfile                  # Heroku/Railway deployment
├── .env.example              # Environment variables template
├── .gitignore                # Git ignore rules
├── README.md                 # This file
├── API_DOCS.md               # API documentation
├── ARCHITECTURE.md           # System architecture
├── QUICKSTART.md             # Quick start guide
├── scripts/                  # Utility scripts
│   ├── setup.sh              # Setup script
│   ├── start.sh              # Start script
│   └── test.sh               # Test script
├── examples/                 # Example scripts
│   ├── train_model.py        # Training example
│   ├── generate_design.py    # Generation example
│   └── test_api.py           # API testing example
└── models/                   # Model storage (auto-created)
    ├── saved/                # Saved model weights
    └── checkpoints/          # Training checkpoints
```

## Security Considerations

- File upload size limits enforced
- File type validation
- Input sanitization
- No sensitive data in responses
- Environment-based configuration
- CORS protection

## Performance Optimization

- TensorFlow GPU acceleration (when available)
- Data prefetching and batching
- Streaming data pipeline
- Efficient image processing
- Model checkpointing

## Troubleshooting

### Issue: Model not found
**Solution**: Train a model first or the application will use an untrained model for demonstration.

### Issue: Out of memory during training
**Solution**: Reduce `BATCH_SIZE` in config.py or environment variables.

### Issue: Slow inference
**Solution**: Ensure GPU is available and TensorFlow can detect it. Check with:
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

## Future Enhancements

- [ ] Support for 3D floor plans
- [ ] Real-time collaboration features
- [ ] Export to CAD formats
- [ ] Virtual reality preview
- [ ] Cost estimation
- [ ] Material selection
- [ ] Lighting simulation
- [ ] Additional design styles

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - See LICENSE file for details

## Acknowledgments

- TensorFlow team for excellent ML framework
- Flask community for web framework
- Public furniture dataset contributors

## Contact

For questions or support, please open an issue on GitHub.

---

**Note**: This application uses synthetic data for demonstration. For production use, integrate with real cloud-hosted furniture datasets or train with your own data.