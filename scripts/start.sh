#!/bin/bash
# Start script for AI Interior Designer
# This script starts the application in development or production mode

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Check for virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Parse command line arguments
MODE="${1:-dev}"

echo "============================================================"
echo "AI Interior Designer - Starting Application"
echo "============================================================"

case "$MODE" in
    dev|development)
        echo "Mode: Development"
        echo "Starting Flask development server..."
        echo ""
        export DEBUG=True
        python run.py
        ;;
    prod|production)
        echo "Mode: Production"
        echo "Starting Gunicorn server..."
        echo ""
        export DEBUG=False
        gunicorn wsgi:app --bind 0.0.0.0:${PORT:-5000} --workers 4
        ;;
    docker)
        echo "Mode: Docker"
        echo "Starting with Docker Compose..."
        echo ""
        docker-compose up --build
        ;;
    *)
        echo "Usage: $0 [dev|prod|docker]"
        echo ""
        echo "Options:"
        echo "  dev     - Start in development mode with Flask (default)"
        echo "  prod    - Start in production mode with Gunicorn"
        echo "  docker  - Start with Docker Compose"
        exit 1
        ;;
esac
