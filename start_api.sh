#!/bin/bash

# Finance Agent API Startup Script
# This script starts the Finance Agent API locally with proper environment setup

set -e

# Colors for output (only if terminal supports it)
if [ -t 1 ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    NC='\033[0m'
else
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    NC=''
fi

# Configuration
API_HOST="0.0.0.0"
API_PORT="8000"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$PROJECT_ROOT/venv"
PYTHON_PATH="$VENV_PATH/bin/python"
REQUIREMENTS_FILE="$PROJECT_ROOT/requirements.txt"

echo -e "${BLUE}🚀 Finance Agent API Startup Script${NC}"
echo -e "${BLUE}================================${NC}"
echo ""

# Function to print colored output
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠️${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ️${NC} $1"
}

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    print_warning "Virtual environment not found at $VENV_PATH"
    print_info "Creating virtual environment..."
    
    # Check if python3 is available
    if ! command -v python3 >/dev/null 2>&1; then
        print_error "Python3 is not installed or not in PATH"
        exit 1
    fi
    
    python3 -m venv "$VENV_PATH"
    print_status "Virtual environment created successfully"
fi

# Activate virtual environment
print_info "Activating virtual environment..."
. "$VENV_PATH/bin/activate"

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    print_error "Failed to activate virtual environment"
    exit 1
fi

print_status "Virtual environment activated: $VIRTUAL_ENV"

# Check if requirements are installed
print_info "Checking dependencies..."
if ! "$PYTHON_PATH" -c "import fastapi, uvicorn, openai" >/dev/null 2>&1; then
    print_warning "Some dependencies are missing. Installing requirements..."
    pip install -r "$REQUIREMENTS_FILE"
    print_status "Dependencies installed successfully"
else
    print_status "All dependencies are already installed"
fi

# Check for environment variables
print_info "Checking environment configuration..."

# Check for required environment variables
MISSING_ENV_VARS=""

if [ -z "${OPENAI_API_KEY:-}" ]; then
    MISSING_ENV_VARS="$MISSING_ENV_VARS OPENAI_API_KEY"
fi

if [ -z "${SUPABASE_URL:-}" ]; then
    MISSING_ENV_VARS="$MISSING_ENV_VARS SUPABASE_URL"
fi

if [ -z "${SUPABASE_KEY:-}" ]; then
    MISSING_ENV_VARS="$MISSING_ENV_VARS SUPABASE_KEY"
fi

# Load .env file if it exists
if [ -f "$PROJECT_ROOT/.env" ]; then
    print_info "Loading environment variables from .env file..."
    while IFS= read -r line || [ -n "$line" ]; do
        if [ -n "$line" ] && ! echo "$line" | grep -q '^[[:space:]]*#'; then
            line=$(echo "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
            if echo "$line" | grep -q '='; then
                var_name=$(echo "$line" | cut -d'=' -f1 | sed 's/[[:space:]]*$//')
                var_value=$(echo "$line" | cut -d'=' -f2- | sed 's/^[[:space:]]*//')
                export "$var_name=$var_value"
            fi
        fi
    done < "$PROJECT_ROOT/.env"
    print_status ".env file loaded"
fi

# Check again after loading .env
MISSING_ENV_VARS=""
if [ -z "${OPENAI_API_KEY:-}" ]; then
    MISSING_ENV_VARS="$MISSING_ENV_VARS OPENAI_API_KEY"
fi

if [ -z "${SUPABASE_URL:-}" ]; then
    MISSING_ENV_VARS="$MISSING_ENV_VARS SUPABASE_URL"
fi

if [ -z "${SUPABASE_KEY:-}" ]; then
    MISSING_ENV_VARS="$MISSING_ENV_VARS SUPABASE_KEY"
fi

# Remove leading space and check if any are missing
MISSING_ENV_VARS="${MISSING_ENV_VARS# }"

if [ -n "$MISSING_ENV_VARS" ]; then
    print_warning "Missing required environment variables:"
    for var in $MISSING_ENV_VARS; do
        echo "  - $var"
    done
    print_info "Please set these variables in your .env file or environment"
    print_info "You can create a .env file based on .env.example if available"
fi

# Check if port is available
print_info "Checking if port $API_PORT is available..."
if command -v lsof >/dev/null 2>&1 && lsof -Pi :$API_PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    print_warning "Port $API_PORT is already in use"
    print_info "Attempting to find an available port..."
    
    for port in 8001 8002 8003 8004 8005 8006 8007 8008 8009 8010; do
        if ! lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
            API_PORT=$port
            print_status "Using available port: $API_PORT"
            break
        fi
    done
else
    print_status "Port $API_PORT is available"
fi

# Set environment variables for MLflow
export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
export MLFLOW_EXPERIMENT_NAME="finance-agent"
export MLFLOW_ARTIFACT_LOCATION="./mlflow_artifacts"

print_info "MLflow configuration:"
echo "  - Tracking URI: $MLFLOW_TRACKING_URI"
echo "  - Experiment: $MLFLOW_EXPERIMENT_NAME"
echo "  - Artifacts: $MLFLOW_ARTIFACT_LOCATION"

# Create necessary directories
print_info "Creating necessary directories..."
mkdir -p "$PROJECT_ROOT/mlflow_artifacts"
mkdir -p "$PROJECT_ROOT/chroma_db"
print_status "Directories created"

# Start the API
echo ""
echo -e "${BLUE}🚀 Starting Finance Agent API...${NC}"
echo -e "${BLUE}================================${NC}"
print_info "API will be available at: http://$API_HOST:$API_PORT"
print_info "API Documentation: http://$API_HOST:$API_PORT/docs"
print_info "ReDoc Documentation: http://$API_HOST:$API_PORT/redoc"
echo ""

# Change to src directory and start the API
cd "$PROJECT_ROOT/src"

print_status "Starting API server..."
"$PYTHON_PATH" -m uvicorn api.main:app --host "$API_HOST" --port "$API_PORT" --reload
