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

- `pack_append.pyx`: Cython implementation for polyomino packing
- `group_tiles.pyx`: Cython implementation for tile grouping
- `setup.py`: Build configuration
- `build.sh`: Build script
- `tests/`: Test suite using pytest
  - `test_pack_append.py`: Tests for pack_append implementation
  - `test_group_tiles.py`: Tests for group_tiles implementation
  - `test_pack_append_legacy.py`: Legacy test format
  - `conftest.py`: Pytest configuration and fixtures
