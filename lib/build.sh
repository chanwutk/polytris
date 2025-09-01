#!/bin/bash

# Build script for the Cython polyomino packing library

set -e

echo "Building Cython polyomino packing library..."

# Check if Cython is installed
if ! python -c "import Cython" 2>/dev/null; then
    echo "Installing Cython..."
    pip install Cython
fi

# Build the Cython extension
echo "Building Cython extension..."
python setup.py build_ext --inplace

echo "Build completed successfully!"
echo "You can now import the fast Cython implementation:"
echo "  from lib.pack_append import pack_append"
echo ""
echo "Run tests with: pytest tests/ -v"
