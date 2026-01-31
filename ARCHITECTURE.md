# Architecture Documentation

## System Overview

The AI Interior Designer is a professional-grade machine learning application that generates interior designs from building floor plans. The system follows industry-standard practices with a modular, scalable architecture.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                         Client Layer                         │
│  (Web/Mobile App, CLI, API Consumers)                       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         │ HTTP/REST
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                      API Layer (Flask)                       │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│  │  Design    │  │  Analysis  │  │  Training  │            │
│  │ Generation │  │  Endpoints │  │  Control   │            │
│  └────────────┘  └────────────┘  └────────────┘            │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Business Logic Layer                      │
│  ┌──────────────────────┐  ┌─────────────────────────┐     │
│  │  Inference Engine    │  │  Training Pipeline      │     │
│  │  - Design Gen        │  │  - Generator Model      │     │
│  │  - Room Detection    │  │  - Discriminator Model  │     │
│  │  - Furniture Suggest │  │  - Loss Functions       │     │
│  └──────────────────────┘  └─────────────────────────┘     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                     Data Layer                               │
│  ┌────────────────────┐  ┌──────────────────────────┐      │
│  │  Streaming Module  │  │  Preprocessing Pipeline  │      │
│  │  - TFDS Streaming  │  │  - Image Normalization   │      │
│  │  - HTTP Streaming  │  │  - Augmentation          │      │
│  │  - Batch Pipeline  │  │  - Room Extraction       │      │
│  └────────────────────┘  └──────────────────────────┘      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   External Data Sources                      │
│  ┌────────────────┐  ┌──────────────────┐                  │
│  │  Cloud Storage │  │  Public Datasets │                  │
│  │  (TF Datasets) │  │  (HTTP URLs)     │                  │
│  └────────────────┘  └──────────────────┘                  │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. API Layer (`app.py`)

**Purpose**: RESTful API interface for all client interactions

**Endpoints**:
- `POST /api/design/generate` - Generate designs from floor plans
- `POST /api/design/analyze` - Analyze floor plan structure
- `POST /api/design/suggest` - Get furniture suggestions
- `POST /api/train/start` - Initiate model training
- `GET /api/train/status` - Check training progress
- `GET /api/health` - Health check

**Technologies**:
- Flask 3.0 for web framework
- Flask-CORS for cross-origin support
- Werkzeug for file handling

**Security Features**:
- File size limits (16MB)
- File type validation
- Input sanitization
- CORS protection

### 2. ML Models (`training_pipeline.py`)

#### Generator Model (InteriorDesignGenerator)

**Architecture**: Conditional GAN (Pix2Pix-based)

```
Input: Floor Plan (256x256x3)
    ↓
┌─────────────────────┐
│  Encoder (U-Net)    │
│  - Conv2D layers    │
│  - Downsampling     │
│  - Feature extract  │
└──────────┬──────────┘
           │
    Bottleneck (8x8x512)
           │
┌──────────▼──────────┐
│  Decoder            │
│  - Conv2DTranspose  │
│  - Upsampling       │
│  - Skip connections │
└──────────┬──────────┘
           ▼
Output: Interior Design (256x256x3)
```

**Parameters**: ~13.9M trainable parameters

**Key Features**:
- Skip connections for detail preservation
- Batch normalization for stable training
- Dropout for regularization
- Tanh activation for output normalization

#### Discriminator Model (DesignDiscriminator)

**Architecture**: PatchGAN discriminator

```
Inputs: Floor Plan + Design (concatenated)
    ↓
┌─────────────────────┐
│  Convolutional Net  │
│  - Conv2D layers    │
│  - LeakyReLU        │
│  - BatchNorm        │
└──────────┬──────────┘
           ▼
Output: Patch predictions (32x32x1)
```

**Parameters**: ~2.8M trainable parameters

**Purpose**: Evaluates design realism at patch level

### 3. Data Streaming (`data_streaming.py`)

**Key Innovation**: Zero local storage - all data streamed from cloud

#### Streaming Strategies

1. **TensorFlow Datasets Streaming**
   ```python
   tfds.load(dataset_name, download=False, try_gcs=True)
   ```
   - Streams from Google Cloud Storage
   - No local download required
   - Automatic prefetching

2. **HTTP Chunked Streaming**
   ```python
   requests.get(url, stream=True)
   for chunk in response.iter_content(chunk_size=1MB):
       process(chunk)
   ```
   - Memory-efficient
   - Processes data on-the-fly
   - Configurable buffer size

#### Data Pipeline

```
Cloud Dataset
    ↓
Stream (no download)
    ↓
Preprocess (normalize, resize, augment)
    ↓
Batch (32 samples)
    ↓
Prefetch (AUTOTUNE)
    ↓
Model Training
```

**Optimizations**:
- Parallel preprocessing (`num_parallel_calls=AUTOTUNE`)
- Prefetching for GPU/CPU overlap
- Shuffle buffer (1000 samples)
- Memory-efficient batching

### 4. Inference Engine (`inference.py`)

**Components**:

1. **Design Generation**
   - Loads trained generator
   - Preprocesses floor plan
   - Generates design variations
   - Post-processes output

2. **Floor Plan Analysis**
   - Room detection via contour finding
   - Area calculation
   - Room classification (heuristic)
   - Bounding box extraction

3. **Furniture Suggestions**
   - Room-type based recommendations
   - Size-aware filtering
   - Style customization
   - Position suggestions

### 5. Configuration (`config.py`)

Centralized configuration with environment variable support:

```python
Config
├── Application (host, port, debug)
├── Model (batch size, epochs, learning rate)
├── Data (dataset URLs, cache settings)
└── API (file limits, allowed types)
```

## Training Pipeline

### Training Flow

```
1. Initialize Pipeline
   ↓
2. Create Streaming Dataset
   ↓
3. For each epoch:
   │  For each batch (streamed):
   │    ├── Generate design
   │    ├── Discriminator evaluation
   │    ├── Calculate losses
   │    ├── Backpropagation
   │    └── Update weights
   ↓
4. Save checkpoints
   ↓
5. Save final model
```

### Loss Functions

**Generator Loss**:
```
L_gen = L_adversarial + λ * L_L1
      = BCE(D(G(x)), 1) + 100 * MAE(G(x), y)
```
- Adversarial loss: Fool discriminator
- L1 loss: Pixel-wise similarity (weighted 100x)

**Discriminator Loss**:
```
L_disc = BCE(D(x, y), 1) + BCE(D(x, G(x)), 0)
```
- Real loss: Classify real designs as real
- Fake loss: Classify generated designs as fake

### Training Strategy

1. **Alternating Updates**: Generator and discriminator trained alternately
2. **Checkpointing**: Save every 5 epochs
3. **Metrics Tracking**: Track gen_loss and disc_loss
4. **Early Stopping**: Manual (Ctrl+C with save)

## Data Flow

### Design Generation Request

```
1. Client uploads floor plan image
   ↓
2. API validates file (size, type)
   ↓
3. Load image and convert to array
   ↓
4. Preprocess:
   - Resize to 256x256
   - Normalize to [-1, 1]
   ↓
5. Generator inference
   ↓
6. Post-process:
   - Denormalize to [0, 255]
   - Convert to RGB image
   ↓
7. Encode to base64
   ↓
8. Return JSON response
```

### Training Data Flow

```
1. Start training request
   ↓
2. Initialize models and optimizers
   ↓
3. Create streaming dataset:
   - Connect to cloud source
   - Configure preprocessing
   - Set up batching/prefetching
   ↓
4. Stream batches:
   For each batch:
     - Fetch from cloud (no download)
     - Preprocess on-the-fly
     - Train step
     - Update metrics
   ↓
5. Save model checkpoint
   ↓
6. Return training status
```

## Scalability Considerations

### Horizontal Scaling

1. **API Server**: Multiple Flask instances behind load balancer
2. **Training**: Distributed training with TensorFlow Distributed
3. **Data**: Cloud storage handles concurrent access

### Vertical Scaling

1. **GPU Support**: Automatic GPU detection for acceleration
2. **Batch Size**: Configurable based on available memory
3. **Prefetching**: Automatic tuning with AUTOTUNE

### Performance Optimizations

1. **Lazy Loading**: Models loaded on first request
2. **Caching**: In-memory caching for frequent operations
3. **Streaming**: No disk I/O bottlenecks
4. **Prefetching**: Overlapped data loading and computation

## Security Architecture

### Input Validation

- File size limits (16MB max)
- File type whitelist (PNG, JPG, JPEG, PDF)
- Filename sanitization
- Content-type verification

### API Security

- CORS protection
- Error message sanitization
- No sensitive data in responses
- Environment-based secrets

### Code Security

- No SQL injection (no database)
- No command injection (validated inputs)
- No path traversal (secure file handling)
- Dependencies scanned for vulnerabilities

## Deployment Architecture

### Development

```
Local Machine
├── Python 3.8+
├── Virtual Environment
├── Flask Development Server
└── CPU/GPU for training
```

### Production (Recommended)

```
┌────────────────────┐
│   Load Balancer    │
└─────────┬──────────┘
          │
    ┌─────┴─────┬─────────┐
    ▼           ▼         ▼
┌─────────┐ ┌─────────┐ ┌─────────┐
│ Flask   │ │ Flask   │ │ Flask   │
│ Instance│ │ Instance│ │ Instance│
└─────────┘ └─────────┘ └─────────┘
    │           │         │
    └─────┬─────┴─────────┘
          ▼
┌────────────────────┐
│  Model Storage     │
│  (Cloud/Shared)    │
└────────────────────┘
          │
          ▼
┌────────────────────┐
│  Dataset Streaming │
│  (Cloud Sources)   │
└────────────────────┘
```

### Production Stack

- **Web Server**: Gunicorn
- **Reverse Proxy**: Nginx
- **Container**: Docker
- **Orchestration**: Kubernetes (optional)
- **Monitoring**: Prometheus + Grafana
- **Logging**: ELK Stack

## Technology Stack

### Backend
- **Python 3.8+**: Core language
- **Flask 3.0**: Web framework
- **TensorFlow 2.16+**: ML framework
- **NumPy**: Numerical computing
- **OpenCV**: Image processing
- **Pillow**: Image handling

### ML/AI
- **TensorFlow/Keras**: Model building
- **TensorFlow Datasets**: Data streaming
- **scikit-learn**: Utilities

### Infrastructure
- **Gunicorn**: WSGI server
- **Docker**: Containerization (optional)
- **Cloud Storage**: Dataset hosting

## Monitoring and Observability

### Metrics

1. **API Metrics**:
   - Request count
   - Response time
   - Error rate
   - Throughput

2. **Model Metrics**:
   - Generator loss
   - Discriminator loss
   - Training time
   - Inference time

3. **System Metrics**:
   - CPU/GPU usage
   - Memory usage
   - Network I/O
   - Storage I/O

### Logging

Structured logging with levels:
- DEBUG: Detailed diagnostic
- INFO: General information
- WARNING: Warning messages
- ERROR: Error events
- CRITICAL: Critical failures

## Future Enhancements

### Planned Features

1. **Advanced Models**:
   - 3D floor plan support
   - Style transfer
   - Custom furniture placement

2. **Data Pipeline**:
   - More dataset sources
   - Real-time augmentation
   - Dataset versioning

3. **API**:
   - WebSocket for real-time updates
   - Batch processing API
   - Async training with Celery

4. **UI**:
   - Web dashboard
   - Interactive editor
   - Design comparison tool

### Technical Debt

- Add comprehensive test suite
- Implement async training
- Add model versioning
- Implement A/B testing framework
- Add distributed training support

## Contributing

See main README.md for contribution guidelines.

## License

MIT License - See LICENSE file for details.
