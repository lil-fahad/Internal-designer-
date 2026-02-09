# API Documentation

## Overview

The AI Interior Designer API provides REST endpoints for generating interior designs from floor plans using advanced machine learning models.

## Base URL

```
http://localhost:5000
```

## Authentication

Currently, the API does not require authentication. For production deployment, implement proper authentication mechanisms.

## Endpoints

### 1. Home

Get API information.

**Endpoint:** `GET /`

**Response:**
```json
{
  "name": "AI Interior Designer",
  "version": "1.0.0",
  "description": "AI-powered interior design generation from floor plans",
  "endpoints": {
    "/api/design/generate": "Generate interior design from floor plan",
    "/api/design/analyze": "Analyze floor plan structure",
    "/api/design/suggest": "Get furniture suggestions",
    "/api/train/start": "Start model training",
    "/api/train/status": "Get training status",
    "/api/health": "Health check"
  }
}
```

### 2. Health Check

Check API health status.

**Endpoint:** `GET /api/health`

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "streaming_enabled": true
}
```

### 3. Generate Interior Design

Generate interior design from a floor plan image.

**Endpoint:** `POST /api/design/generate`

**Request:**
- Content-Type: `multipart/form-data`
- Parameters:
  - `file` (required): Floor plan image file (PNG, JPG, JPEG, PDF)
  - `style` (optional): Design style - one of: `modern`, `classic`, `minimalist`, `rustic` (default: `modern`)
  - `variations` (optional): Number of design variations to generate (1-5, default: 1)

**Example:**
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
  "designs": [
    "iVBORw0KGgoAAAANSUhEUgAA...",  // base64 encoded image
    "iVBORw0KGgoAAAANSUhEUgAA...",
    "iVBORw0KGgoAAAANSUhEUgAA..."
  ],
  "style": "modern",
  "num_variations": 3
}
```

**Error Response:**
```json
{
  "error": "No file provided"
}
```

**Status Codes:**
- 200: Success
- 400: Bad request (missing file, invalid parameters)
- 413: File too large
- 500: Internal server error

### 4. Analyze Floor Plan

Analyze a floor plan to detect rooms and extract information.

**Endpoint:** `POST /api/design/analyze`

**Request:**
- Content-Type: `multipart/form-data`
- Parameters:
  - `file` (required): Floor plan image file

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
      {
        "room_id": 1,
        "type": "bedroom",
        "area": 4000,
        "bbox": [120, 20, 80, 120]
      }
    ],
    "dimensions": [1024, 768]
  }
}
```

### 5. Furniture Suggestions

Get furniture suggestions for a specific room.

**Endpoint:** `POST /api/design/suggest`

**Request:**
- Content-Type: `application/json`
- Body:
```json
{
  "room_type": "bedroom",
  "dimensions": [5, 3, 4],
  "style": "modern"
}
```

**Parameters:**
- `room_type`: Type of room (e.g., `bedroom`, `living_room`, `kitchen`, `office`)
- `dimensions`: Array of [width, height, depth] in meters
- `style`: Design style (optional, default: `modern`)

**Example:**
```bash
curl -X POST http://localhost:5000/api/design/suggest \
  -H "Content-Type: application/json" \
  -d '{
    "room_type": "bedroom",
    "dimensions": [5, 3, 4],
    "style": "modern"
  }'
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
    {
      "item": "nightstand",
      "quantity": 2,
      "position": "bedside",
      "style": "modern"
    },
    {
      "item": "wardrobe",
      "size": "large",
      "position": "wall",
      "style": "modern"
    }
  ]
}
```

### 6. Start Training

Start model training with streaming data.

**Endpoint:** `POST /api/train/start`

**Request:**
- Content-Type: `application/json`
- Body:
```json
{
  "epochs": 50,
  "batch_size": 32
}
```

**Parameters:**
- `epochs` (optional): Number of training epochs (default: from config)
- `batch_size` (optional): Batch size for training (default: from config)

**Example:**
```bash
curl -X POST http://localhost:5000/api/train/start \
  -H "Content-Type: application/json" \
  -d '{
    "epochs": 50,
    "batch_size": 32
  }'
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

### 7. Training Status

Get current training status.

**Endpoint:** `GET /api/train/status`

**Example:**
```bash
curl http://localhost:5000/api/train/status
```

**Response:**
```json
{
  "status": "training",
  "streaming_enabled": true
}
```

or

```json
{
  "status": "not_started",
  "model_available": false
}
```

### 8. Fetch Training Data

Fetch training data samples with metadata.

**Endpoint:** `GET /api/train/data`

**Parameters:**
- `num_samples` (optional): Number of samples to fetch (1-20, default: 5)

**Example:**
```bash
curl "http://localhost:5000/api/train/data?num_samples=3"
```

**Response:**
```json
{
  "success": true,
  "num_samples": 3,
  "image_size": [256, 256],
  "batch_size": 32,
  "streaming_mode": true,
  "samples": [
    {
      "id": 0,
      "image": "iVBORw0KGgoAAAANSUhEUgAA...",
      "label": "chair",
      "style": "modern",
      "dimensions": [45.2, 82.1, 50.3]
    },
    {
      "id": 1,
      "image": "iVBORw0KGgoAAAANSUhEUgAA...",
      "label": "table",
      "style": "classic",
      "dimensions": [120.5, 75.0, 80.2]
    }
  ]
}
```

## Error Handling

All endpoints return JSON responses. Error responses follow this format:

```json
{
  "error": "Description of the error"
}
```

Common status codes:
- `200`: Success
- `400`: Bad Request - Invalid parameters or missing required fields
- `413`: Payload Too Large - Uploaded file exceeds size limit
- `500`: Internal Server Error - Server-side error

## Rate Limiting

Currently, no rate limiting is implemented. For production deployment, implement appropriate rate limiting.

## File Size Limits

- Maximum upload size: 16MB (configurable in `config.py`)

## Supported File Formats

- Images: PNG, JPG, JPEG
- Documents: PDF (will be converted to image)

## CORS

CORS is enabled for all origins. For production, configure specific allowed origins.

## Examples

### Python

```python
import requests

# Generate design
with open('floor_plan.png', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/api/design/generate',
        files={'file': f},
        data={'style': 'modern', 'variations': 2}
    )
    result = response.json()
    print(f"Generated {len(result['designs'])} designs")
```

### JavaScript

```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('style', 'modern');

fetch('http://localhost:5000/api/design/generate', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => {
  console.log('Designs:', data.designs);
});
```

### cURL

```bash
# Generate design
curl -X POST http://localhost:5000/api/design/generate \
  -F "file=@floor_plan.png" \
  -F "style=modern" \
  -F "variations=2"

# Get furniture suggestions
curl -X POST http://localhost:5000/api/design/suggest \
  -H "Content-Type: application/json" \
  -d '{"room_type": "bedroom", "dimensions": [5, 3, 4]}'
```

## Streaming Data Pipeline

The training pipeline uses streaming data from cloud sources:

- **No Local Downloads**: Data is streamed directly from cloud storage
- **Memory Efficient**: Only processes batches in memory
- **TensorFlow Datasets**: Leverages TFDS for cloud streaming
- **HTTP Streaming**: Supports streaming from public URLs

## Model Information

- **Architecture**: Conditional GAN (Pix2Pix style)
- **Input**: Floor plan image (256x256 RGB)
- **Output**: Interior design image (256x256 RGB)
- **Training**: Adversarial loss + L1 pixel loss
- **Framework**: TensorFlow 2.15

## Security Considerations

For production deployment:

1. Implement authentication (JWT, OAuth, etc.)
2. Add rate limiting
3. Configure CORS for specific origins
4. Use HTTPS
5. Validate and sanitize all inputs
6. Implement request size limits
7. Add logging and monitoring
8. Use environment variables for sensitive config

## Support

For issues or questions, please open an issue on GitHub.
