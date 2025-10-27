# Polyis Pack - High-Performance Cython Implementation

## Building

**Recommended:** Run the build script from the project root:

```bash
cd /polyis
./build_cython.sh
```

Or skip tests:
```bash
cd /polyis
./build_cython.sh --skip-tests
```

**Alternative (legacy):** Run from the `polyis/binpack` directory:

```bash
cd /polyis/polyis/binpack
./build.sh --skip-tests
```

**Manual build from root:**
```bash
cd /polyis
python setup_cython.py build_ext --inplace
```

## Usage

```python
from polyis.binpack.pack_append import pack_append
from polyis.binpack.group_tiles import group_tiles
from polyis.binpack.adapters import pack_append, group_tiles  # Alternative

positions = pack_append(polyominoes, h, w, occupied_tiles)
if positions is not None:
    print(f"Successfully placed {len(positions)} polyominoes")
else:
    print("Packing failed")
```

## Directory Structure

- `pack_append.pyx`: Cython implementation for polyomino packing
- `group_tiles.pyx`: Cython implementation for tile grouping
- `utilities.pyx`: Shared C data structures and utilities
- `adapters.pyx`: High-level Python API adapters
- `render.pyx`: Rendering utilities
- `setup.py`: Legacy build configuration (use root setup_cython.py instead)
- `build.sh`: Legacy build script (use root build_cython.sh instead)
- `*_original.py`: Reference Python implementations for testing

Tests are located in `/polyis/tests/binpack/`:
  - `test_pack_append.py`: Tests for pack_append implementation
  - `test_group_tiles.py`: Tests for group_tiles implementation
  - `test_performance_comparison.py`: Performance benchmarks
  - `test_pack_append_legacy.py`: Legacy test format
  - `conftest.py`: Pytest configuration and fixtures
  - `benchmark.py`: Performance benchmarking utilities
