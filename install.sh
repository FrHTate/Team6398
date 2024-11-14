#!/bin/bash

# Define environment name and path
ENV_NAME="test_env"
ENV_PATH="$HOME/micromamba/envs/$ENV_NAME"

# Initialize micromamba for the current shell
eval "$(micromamba shell hook -s bash)"  # Replace 'bash' with your shell type if needed (e.g., zsh)

# Check if micromamba is available
if ! command -v micromamba &> /dev/null; then
    echo "Error: Micromamba is not available. Ensure it's installed and initialized."
    exit 1
fi

# Create the environment from the YAML file
echo "Creating Micromamba environment from environment.yml..."
micromamba create -f environment.yml -y

# Check if the environment was created successfully
if [ ! -d "$ENV_PATH" ]; then
    echo "Error: Environment was not created successfully at $ENV_PATH."
    exit 1
fi

# Activate the environment
echo "Activating the environment..."
micromamba activate "$ENV_NAME"

# Process requirements.txt for valid packages
if [ -f requirements.txt ]; then
    # Clean requirements.txt by removing lines with paths or build directories
    sed -i '/build_artifacts\|tmp\|feedstock_root/d' requirements.txt

    # Install packages using pip
    echo "Installing packages from requirements.txt..."
    pip install --no-cache-dir -r requirements.txt
else
    echo "Error: requirements.txt not found."
    exit 1
fi
echo "Installing flash-attn package..."
pip install flash-attn==2.6.3 --no-build-isolation

echo "Environment recreation completed successfully."
