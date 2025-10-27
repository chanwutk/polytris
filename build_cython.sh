#!/bin/bash

# Build script for the Cython polyomino packing library
# Run from the project root directory: /polyis

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
            echo ""
            echo "This script should be run from the project root: /polyis"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Clean up previous build artifacts in polyis/binpack
echo "Cleaning up previous build artifacts..."
find polyis/binpack -name "*.c" -type f -delete
find polyis/binpack -name "*.html" -type f -delete
find polyis/binpack -name "*.so" -type f -delete
echo "Cleanup completed."

# Build the Cython extensions
echo "Building Cython extensions..."
python setup_cython.py build_ext --inplace

echo "Build completed successfully!"
echo "You can now import the fast Cython implementations:"
echo "  from polyis.binpack.pack_append import pack_append"
echo "  from polyis.binpack.group_tiles import group_tiles"
echo ""

# Run tests by default unless disabled
if [ "$RUN_TESTS" = true ]; then
    echo "Running tests..."
    pytest tests/binpack/ -v
else
    echo "Tests skipped (use 'pytest tests/binpack/ -v' to run manually)"
fi

