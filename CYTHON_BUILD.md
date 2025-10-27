# Cython Build System

This document describes the build system for the high-performance Cython modules in the `polyis.binpack` package.

## Quick Start

From the project root (`/polyis`):

```bash
./build_cython.sh              # Build with tests
./build_cython.sh --skip-tests # Build without tests
```

## Files Overview

### Root Directory (`/polyis/`)

- **`build_cython.sh`** - Main build script (run from root)
- **`setup_cython.py`** - Setuptools configuration for Cython extensions

### Binpack Directory (`/polyis/polyis/binpack/`)

- **`build.sh`** - Legacy build script (can be run from within binpack/)
- **`setup.py`** - Legacy setup file (kept for backwards compatibility)

## Build Process

The build script performs the following steps:

1. **Clean** - Removes old `.c`, `.html`, and `.so` files
2. **Compile** - Runs Cython to generate C code and compile to `.so` extensions
3. **Copy** - Places compiled `.so` files in `polyis/binpack/`
4. **Test** - (Optional) Runs pytest on the test suite

## Output Files

After building, you should see:

```
polyis/binpack/
├── utilities.cpython-313-x86_64-linux-gnu.so
├── adapters.cpython-313-x86_64-linux-gnu.so
├── pack_append.cpython-313-x86_64-linux-gnu.so
├── group_tiles.cpython-313-x86_64-linux-gnu.so
└── render.cpython-313-x86_64-linux-gnu.so
```

## Import Paths

All modules use fully qualified package names:

```python
from polyis.binpack.pack_append import pack_append
from polyis.binpack.group_tiles import group_tiles
from polyis.binpack.adapters import pack_append, group_tiles  # High-level API
```

## Technical Details

### Why Fully Qualified Names?

Cython's `cimport` statements compile to C-level imports that reference the exact module name specified in `setup.py`. When using package hierarchies, you must use fully qualified names (e.g., `"polyis.binpack.utilities"`) instead of simple names (e.g., `"utilities"`).

### Module Dependencies

- `utilities.pyx` - Base data structures (no dependencies)
- `pack_append.pyx` - Depends on `utilities`
- `group_tiles.pyx` - Depends on `utilities`
- `adapters.pyx` - Depends on `utilities`, `pack_append`, `group_tiles`
- `render.pyx` - Depends on `utilities`

All inter-module imports use the format:
```python
from polyis.binpack.utilities cimport IntStack, Polyomino, ...
```

## Troubleshooting

### ModuleNotFoundError: No module named 'utilities'

This means the Cython modules were compiled with incorrect names. Rebuild from the root:

```bash
cd /polyis
./build_cython.sh --skip-tests
```

### Build fails with "No such file or directory"

Make sure you're running from the project root (`/polyis`), not from `polyis/binpack/`.

### Tests fail after rebuild

Clean everything and rebuild:

```bash
cd /polyis
find polyis/binpack -name "*.so" -delete
find polyis/binpack -name "*.c" -delete
./build_cython.sh
```

## Development Workflow

1. Edit `.pyx` files in `polyis/binpack/`
2. Run `./build_cython.sh` from root
3. Test your changes: `pytest tests/binpack/ -v`
4. Import and use: `from polyis.binpack.XXX import ...`
