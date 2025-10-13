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

# Clean up previous build artifacts
echo "Cleaning up previous build artifacts..."
find . -name "*.c" -type f -delete
find . -name "*.html" -type f -delete
find . -name "*.so" -type f -delete
echo "Cleanup completed."

# Build the Cython extension
echo "Building Cython extension..."
python setup.py build_ext --inplace

echo "Build completed successfully!"
echo "You can now import the fast Cython implementations:"
echo "  from lib.pack_append import pack_append"
echo "  from lib.group_tiles import group_tiles"
echo ""

# Run tests by default unless disabled
if [ "$RUN_TESTS" = true ]; then
    echo "Running tests..."
    pytest tests/ -v
else
    echo "Tests skipped (use pytest tests/ -v to run manually)"
fi
