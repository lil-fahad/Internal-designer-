#!/bin/bash
# Test script for AI Interior Designer
# This script runs all tests and validations

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Check for virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
fi

echo "============================================================"
echo "AI Interior Designer - Running Tests"
echo "============================================================"

echo ""
echo "1. Testing module imports..."
python3 -c "
from config import Config
from data_streaming import FurnitureDatasetStreamer, FloorPlanDataStreamer
from training_pipeline import TrainingPipeline, InteriorDesignGenerator
from inference import InteriorDesigner
from app import app
print('✓ All modules imported successfully')
"

echo ""
echo "2. Testing configuration..."
python3 -c "
from config import Config
Config.init_app()
print(f'  App Name: {Config.APP_NAME}')
print(f'  Host: {Config.HOST}')
print(f'  Port: {Config.PORT}')
print('✓ Configuration validated')
"

echo ""
echo "3. Running design generation test..."
cd "$PROJECT_DIR"
python3 examples/generate_design.py

echo ""
echo "4. Cleaning up test artifacts..."
rm -f sample_floor_plan.png design_*.png

echo ""
echo "============================================================"
echo "All tests passed successfully!"
echo "============================================================"
