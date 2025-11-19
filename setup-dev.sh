#!/bin/bash

# Script to setup conda environment based on Dockerfile configuration
# This replicates the conda setup from Dockerfile lines 68-72

set -e  # Exit on any error

echo "Setting up conda environment..."

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed. Please install conda first."
    echo "You can install miniconda from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Check if environment.yml exists
if [ ! -f "environment.yml" ]; then
    echo "Error: environment.yml not found in current directory."
    echo "Please run this script from the project root directory."
    exit 1
fi

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo "Error: requirements.txt not found in current directory."
    echo "Please run this script from the project root directory."
    exit 1
fi

echo "Creating or updating conda environment 'polyis' from environment.yml..."
# Create or update the 'polyis' environment from environment.yml
# This replicates Dockerfile line 68-69 but creates a new environment instead of updating base
if conda env list | grep -q "^polyis "; then
    echo "Environment 'polyis' already exists. Updating it..."
    conda env update -f environment.yml --name polyis --prune
else
    echo "Creating new environment 'polyis'..."
    conda env create -f environment.yml --name polyis
fi

echo "Cleaning conda cache..."
conda clean --all --yes

echo "Installing pip packages from requirements.txt in the 'polyis' environment..."
# Install pip packages in the polyis conda environment
# This replicates Dockerfile line 72: pip install --no-build-isolation -r /polyis/requirements.txt
conda run -n polyis pip install --no-build-isolation -r requirements.txt
