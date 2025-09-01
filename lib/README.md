# Polyis Pack - High-Performance Implementation

## Building

```bash
# From the lib directory
chmod +x build.sh
./build.sh
```

Or manually:
```bash
pip install Cython
python setup.py build_ext --inplace
```

## Usage

```python
from lib.pack_append import pack_append

positions = pack_append(polyominoes, h, w, occupied_tiles)
if positions is not None:
    print(f"Successfully placed {len(positions)} polyominoes")
else:
    print("Packing failed")
```

## Directory Structure

- `pack_append.pyx`: Main Cython implementation with direct Python interface
- `setup.py`: Build configuration
- `build_cython.sh`: Build script
- `tests/`: Test suite using pytest
  - `test_pack_append.py`: Main pytest test suite
  - `test_pack_append_legacy.py`: Legacy test format
  - `conftest.py`: Pytest configuration and fixtures
- `integration_example.py`: Usage examples
