#!/bin/bash

# Define the environment name
ENV_NAME="myenv"

# Specify Python version
PYTHON_VERSION="python3.9"

# Check if Python 3.9.20 is installed
if ! command -v $PYTHON_VERSION &> /dev/null; then
  echo "$PYTHON_VERSION is not installed. Please install Python 3.9.20."
  exit 1
fi

# Check if the virtual environment already exists
if [ ! -d "$ENV_NAME" ]; then
  # Create a virtual environment with Python 3.9 if it doesn't exist
  $PYTHON_VERSION -m venv $ENV_NAME
  echo "Virtual environment created at $ENV_NAME with Python 3.9.20"
fi

# Activate the virtual environment
source $ENV_NAME/bin/activate
echo "Virtual environment activated"

# Upgrade pip in the virtual environment
pip install --upgrade pip

# Install regular requirements
pip install -r requirements.txt

# Install flash-attn with --no-build-isolation
pip install flash-attn --no-build-isolation

echo "All packages installed successfully in the virtual environment $ENV_NAME"
