# Quick Start Guide

Get up and running with the AI Interior Designer in 5 minutes!

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- 4GB RAM minimum (8GB recommended)
- Optional: NVIDIA GPU with CUDA support for faster training

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/lil-fahad/Internal-designer-.git
cd Internal-designer-
```

### Step 2: Create Virtual Environment

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all required packages including TensorFlow, Flask, and other dependencies.

## Quick Test

### Test 1: Verify Installation

```bash
python -c "import tensorflow as tf; import flask; print('✓ All dependencies installed')"
```

### Test 2: Run Design Generation Example

```bash
python examples/generate_design.py
```

This will:
- Create a sample floor plan
- Generate interior designs in 3 different styles
- Save the results as PNG files
- Show furniture suggestions

Expected output:
```
============================================================
AI Interior Designer - Design Generation
============================================================

Initializing AI Interior Designer...
Creating sample floor plan...
Sample floor plan saved to: sample_floor_plan.png

Analyzing floor plan...
  Number of rooms detected: 4
  Total area: 230400.0

Generating interior designs...
  Generating modern design...
    Saved: design_modern_v1.png
    Saved: design_modern_v2.png
...
✓ Design generation completed successfully!
```

### Test 3: Start the API Server

```bash
python app.py
```

Expected output:
```
Starting AI Interior Designer...
Streaming mode enabled - no local dataset downloads
Server running on 0.0.0.0:5000
 * Running on http://127.0.0.1:5000
```

The API is now running! Leave this terminal open.

### Test 4: Test the API

Open a **new terminal** and run:

```bash
python examples/test_api.py
```

This will test all API endpoints and show results.

## Basic Usage

### Generate a Design via API

1. Make sure the API server is running (`python app.py`)

2. Use curl to generate a design:

```bash
curl -X POST http://localhost:5000/api/design/generate \
  -F "file=@your_floor_plan.png" \
  -F "style=modern" \
  -F "variations=2"
```

3. Or use Python:

```python
import requests

with open('your_floor_plan.png', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/api/design/generate',
        files={'file': f},
        data={'style': 'modern', 'variations': 2}
    )
    
result = response.json()
print(f"Generated {len(result['designs'])} designs")
```

### Get Furniture Suggestions

```bash
curl -X POST http://localhost:5000/api/design/suggest \
  -H "Content-Type: application/json" \
  -d '{
    "room_type": "bedroom",
    "dimensions": [5, 3, 4],
    "style": "modern"
  }'
```

### Analyze a Floor Plan

```bash
curl -X POST http://localhost:5000/api/design/analyze \
  -F "file=@your_floor_plan.png"
```

## Train Your Own Model

### Quick Training (Demo)

```bash
python examples/train_model.py
```

This will train the model using streaming data from cloud sources (no downloads).

**Note**: For demonstration purposes, this uses synthetic data. In production, configure real furniture dataset URLs in `config.py`.

### Configure Training

Edit `config.py` or create `.env` file:

```bash
# Copy example environment file
cp .env.example .env

# Edit .env
nano .env
```

Set training parameters:
```
BATCH_SIZE=32
EPOCHS=50
LEARNING_RATE=0.0002
```

### Start Training via API

```bash
curl -X POST http://localhost:5000/api/train/start \
  -H "Content-Type: application/json" \
  -d '{
    "epochs": 10,
    "batch_size": 16
  }'
```

## Understanding the Output

### Design Generation

When you generate a design, you'll receive:
- One or more design images (base64 encoded)
- Applied style information
- Number of variations

### Floor Plan Analysis

When you analyze a floor plan, you'll receive:
- Number of rooms detected
- Room classifications (bedroom, living room, etc.)
- Bounding boxes for each room
- Estimated areas

### Furniture Suggestions

When you request suggestions, you'll receive:
- Furniture items appropriate for the room
- Sizes and positions
- Style-specific recommendations

## Next Steps

### 1. Try Different Styles

Available styles:
- `modern` - Clean, contemporary designs
- `classic` - Traditional, elegant styles
- `minimalist` - Simple, functional layouts
- `rustic` - Natural, warm aesthetics

### 2. Create Your Own Floor Plans

Tips for best results:
- Use clear, high-contrast images
- Draw walls in dark colors (black/dark gray)
- Use white/light background for rooms
- Keep dimensions between 500x500 and 2000x2000 pixels
- Use PNG or JPG format

### 3. Integrate with Your Application

See `API_DOCS.md` for complete API reference.

Example integration:
```python
from inference import InteriorDesigner
import numpy as np
from PIL import Image

# Initialize designer
designer = InteriorDesigner()

# Load your floor plan
floor_plan = np.array(Image.open('floor_plan.png'))

# Generate design
designs = designer.generate_design(
    floor_plan,
    style='modern',
    num_variations=3
)

# Save results
for i, design in enumerate(designs):
    Image.fromarray(design).save(f'design_{i}.png')
```

### 4. Train with Real Data

To use real furniture datasets:

1. Find public furniture datasets (e.g., from Kaggle, Hugging Face, etc.)
2. Update `config.py` with dataset URLs:
   ```python
   FURNITURE_DATASET_URLS = [
       'https://your-dataset-url.com/furniture',
   ]
   ```
3. Run training:
   ```bash
   python examples/train_model.py
   ```

## Common Issues

### Issue: "No module named 'tensorflow'"

**Solution**: Make sure you activated the virtual environment and installed dependencies:
```bash
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Issue: "Out of memory" during training

**Solution**: Reduce batch size in `.env` or `config.py`:
```
BATCH_SIZE=16  # or even smaller like 8
```

### Issue: API not responding

**Solution**: Check if server is running:
```bash
curl http://localhost:5000/api/health
```

Should return: `{"status": "healthy", ...}`

### Issue: Slow inference

**Solution**: 
- If you have GPU, ensure TensorFlow can detect it:
  ```python
  import tensorflow as tf
  print(tf.config.list_physical_devices('GPU'))
  ```
- If no GPU, inference will be slower but still works on CPU

## Performance Tips

### For Training
- Use GPU if available (10-100x faster)
- Increase batch size with more RAM/VRAM
- Use prefetching (already enabled)

### For Inference
- Keep model loaded (don't reinitialize for each request)
- Batch multiple requests if possible
- Use GPU for faster generation

### For API
- Use Gunicorn instead of Flask dev server in production:
  ```bash
  gunicorn -w 4 -b 0.0.0.0:5000 app:app
  ```
- Enable caching for frequently requested designs
- Use a reverse proxy (nginx) for static content

## Additional Resources

- **Full Documentation**: See `README.md`
- **API Reference**: See `API_DOCS.md`
- **Architecture Details**: See `ARCHITECTURE.md`
- **Examples**: Check `examples/` directory

## Getting Help

If you encounter issues:

1. Check the documentation
2. Look at example scripts
3. Open an issue on GitHub with:
   - Error message
   - Steps to reproduce
   - Your environment (OS, Python version)

## What's Next?

Now that you have the basics working:

1. ✅ Generate designs from your own floor plans
2. ✅ Experiment with different styles
3. ✅ Train with your own data
4. ✅ Build your own applications using the API
5. ✅ Contribute improvements to the project

Happy designing! 🎨
