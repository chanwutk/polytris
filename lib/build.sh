#!/bin/bash

# Build script for the Cython polyomino packing library

set -e

# Parse command line arguments
RUN_TESTS=true
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-test|--skip-tests)
            RUN_TESTS=false
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--no-test|--skip-tests] [-h|--help]"
            echo ""
            echo "Options:"
            echo "  --no-test, --skip-tests    Skip running tests after build"
            echo "  -h, --help                 Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

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
echo "You can now import the fast Cython implementations:"
echo "  from lib.pack_append import pack_append"
echo "  from lib.group_tiles import group_tiles"

# Run tests by default unless disabled
if [ "$RUN_TESTS" = true ]; then
    echo ""
    echo "Running tests..."
    if command -v pytest >/dev/null 2>&1; then
        pytest tests/ -v
    else
        echo "pytest not found, skipping tests"
        echo "Install pytest and run: pytest tests/ -v"
    fi
else
    echo ""
    echo "Tests skipped (use pytest tests/ -v to run manually)"
fi
