#!/usr/local/bin/python

from setuptools import setup, Extension, Command
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize
import numpy
import os
import glob
import multiprocessing as mp

from mypyc.build import mypycify


ARGS = ["-O3", "-ffast-math", "-march=native", "-mtune=native", "-finline-functions"]
MACROS: list[tuple[str, str | None]] = [("NPY_NO_DEPRECATED_API", "NPY_2_3_API_VERSION")]


extensions = [
    Extension(
        "polyis.pack.cython.utilities",
        ["polyis/pack/cython/utilities.pyx"],
        include_dirs=[numpy.get_include()],
        define_macros=MACROS,
        extra_compile_args=ARGS,
    ),
    Extension(
        "polyis.pack.group_tiles",
        [
            "polyis/pack/group_tiles.pyx",
            "polyis/pack/c/utilities.c",
            "polyis/pack/c/group_tiles.c",
        ],
        include_dirs=["polyis/cbinpack", numpy.get_include()],
        define_macros=MACROS,
        extra_compile_args=ARGS + ["-std=c11"],
    ),
    Extension(
        "polyis.pack.adapters",
        [
            "polyis/pack/adapters.pyx",
            "polyis/pack/c/utilities.c",
        ],
        include_dirs=["polyis/cbinpack", numpy.get_include()],
        define_macros=MACROS,
        extra_compile_args=ARGS + ["-std=c11"],
    ),
    Extension(
        "polyis.pack.pack",
        [
            "polyis/pack/pack.pyx",
            "polyis/pack/c/utilities.c",
            "polyis/pack/c/pack.c",
        ],
        include_dirs=["polyis/cbinpack", numpy.get_include()],
        define_macros=MACROS,
        extra_compile_args=ARGS + ["-std=c11"],
    ),
    Extension(
        "polyis.pack.cython.adapters",
        ["polyis/pack/cython/adapters.pyx"],
        include_dirs=[numpy.get_include()],
        define_macros=MACROS,
        extra_compile_args=ARGS,
    ),
    Extension(
        "polyis.pack.cython.pack_append",
        ["polyis/pack/cython/pack_append.pyx"],
        include_dirs=[numpy.get_include()],
        define_macros=MACROS,
        extra_compile_args=ARGS,
    ),
    Extension(
        "polyis.pack.cython.group_tiles",
        ["polyis/pack/cython/group_tiles.pyx"],
        include_dirs=[numpy.get_include()],
        define_macros=MACROS,
        extra_compile_args=ARGS,
    ),
]


class CleanCommand(Command):
    """Custom clean command to remove build artifacts."""
    
    description = "Remove build artifacts (*.c, *.html, *.so files)"
    user_options = []
    
    # Patterns to clean - can be overridden in subclasses
    c_patterns = [
        'polyis/pack/cython/**/*.c',
        'polyis/pack/cython/*.c',
        'polyis/pack/*.c',
    ]
    so_patterns = [
        'polyis/pack/**/*.so',
        'polyis/pack/*.so',
    ]
    html_patterns = [
        'polyis/pack/**/*.html',
        'polyis/pack/*.html',
    ]
    
    def initialize_options(self):
        pass
    
    def finalize_options(self):
        pass
    
    def run(self):
        """Execute the clean command."""
        removed_count = 0
        for pattern in self.c_patterns + self.so_patterns + self.html_patterns:
            for filepath in glob.glob(pattern, recursive=True):
                try:
                    os.remove(filepath)
                    print(f"Removed: {filepath}")
                    removed_count += 1
                except OSError as e:
                    print(f"Error removing {filepath}: {e}")
        
        print(f"\nCleaned {removed_count} file(s)")


class CleanAnnotateCommand(CleanCommand):
    """Custom clean command to remove only annotation artifacts (*.c, *.html files)."""

    description = "Remove annotation artifacts (*.c, *.html files only)"
    so_patterns = []


class DocCommand(Command):
    """Custom command to generate documentation using Doxygen."""

    description = "Generate documentation for C code using Doxygen"
    user_options = [
        ('clean', 'c', 'Clean documentation before generating'),
        ('open', 'o', 'Open documentation in browser after generating'),
    ]

    def initialize_options(self):
        self.clean = False
        self.open = False

    def finalize_options(self):
        pass

    def run(self):
        """Execute the documentation generation command."""
        import subprocess
        import shutil

        # Path to C code directory
        c_dir = os.path.join('polyis', 'pack', 'c')
        doxyfile = os.path.join(c_dir, 'Doxyfile')
        docs_dir = os.path.join(c_dir, 'docs')
        theme_dir = os.path.join(c_dir, 'doxygen-awesome-css')

        # Check if Doxygen is installed
        if shutil.which('doxygen') is None:
            print("Error: Doxygen is not installed or not in PATH")
            print("Install with: sudo apt-get install doxygen graphviz")
            print("          or: conda install -c conda-forge doxygen graphviz")
            return

        # Check if theme is installed
        if not os.path.exists(theme_dir):
            print("Warning: doxygen-awesome-css theme not found")
            print(f"The theme should be at: {theme_dir}")
            print("\nTo install the theme:")
            print(f"  cd {c_dir}")
            print("  git clone https://github.com/jothepro/doxygen-awesome-css.git")
            print("\nContinuing without theme (will use default Doxygen style)...\n")

        # Clean documentation if requested
        if self.clean and os.path.exists(docs_dir):
            print(f"Cleaning existing documentation in {docs_dir}...")
            shutil.rmtree(docs_dir)
            print("Documentation cleaned.")

        # Check if Doxyfile exists
        if not os.path.exists(doxyfile):
            print(f"Error: Doxyfile not found at {doxyfile}")
            return

        # Generate documentation
        print(f"Generating documentation from {c_dir}...")
        print(f"Using Doxyfile: {doxyfile}")

        try:
            # Run doxygen from the C directory
            result = subprocess.run(
                ['doxygen', 'Doxyfile'],
                cwd=c_dir,
                capture_output=True,
                text=True
            )

            # Print output
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr)

            if result.returncode == 0:
                html_index = os.path.join(docs_dir, 'html', 'index.html')
                print(f"\nDocumentation generated successfully!")
                print(f"HTML documentation: {html_index}")

                # Open in browser if requested
                if self.open:
                    if os.path.exists(html_index):
                        print(f"Opening documentation in browser...")
                        if shutil.which('xdg-open'):  # Linux
                            subprocess.run(['xdg-open', html_index])
                        elif shutil.which('open'):  # macOS
                            subprocess.run(['open', html_index])
                        elif shutil.which('start'):  # Windows
                            subprocess.run(['start', html_index], shell=True)
                        else:
                            print(f"Could not detect browser opener. Open manually: {html_index}")
                    else:
                        print(f"Error: Generated documentation not found at {html_index}")
            else:
                print(f"\nError: Doxygen failed with return code {result.returncode}")

        except Exception as e:
            print(f"Error running Doxygen: {e}")


class BuildExt(build_ext):
    """Custom build_ext command that defaults to --inplace and cleans artifacts."""

    def initialize_options(self):
        super().initialize_options()
        # Set inplace to True by default
        self.inplace = True


ext_modules = cythonize(
    extensions,
    nthreads=mp.cpu_count() // 2,
    compiler_directives={
        "language_level": 3,
        # Performance optimizations
        "boundscheck": False,
        "wraparound": False,
        "initializedcheck": False,
        "nonecheck": False,
        "freethreading_compatible": True,
        "subinterpreters_compatible": 'own_gil',
        "overflowcheck": False,
        "overflowcheck.fold": False,
        "embedsignature": False,
        "cdivision": True,
        "cpow": True,
        "optimize.use_switch": True,
        "optimize.unpack_method_calls": True,
        "warn.undeclared": False,
        "warn.unreachable": False,
        "warn.maybe_uninitialized": False,
        "warn.unused": False,
        "warn.unused_arg": False,
        "warn.unused_result": False,
        "warn.multiple_declarators": False,
        # Additional performance settings
        "infer_types": True,
        "infer_types.verbose": False,
        "profile": False,
        "linetrace": False,
        "emit_code_comments": False,
        "annotation_typing": False,
        "c_string_type": "str",
        "c_string_encoding": "ascii",
        "type_version_tag": True,
        "unraisable_tracebacks": False,
        "iterable_coroutine": True,
        # "async_gil": True,
        # "freelist": 1000,
        "fast_gil": True,
        # "fast_math": True,
    },
    annotate=True
)

mypyc_extensions = mypycify([
    'polyis/tracker/kalman_filter.py',
    'polyis/tracker/sort.py',
])
# Add optimization flags and include directories to mypyc-compiled extensions
for ext in mypyc_extensions:
    ext.extra_compile_args = ARGS
    ext.define_macros = MACROS
    # Add numpy include directory if not already present
    if numpy.get_include() not in (ext.include_dirs or []):
        if ext.include_dirs is None:
            ext.include_dirs = []
        ext.include_dirs.append(numpy.get_include())
ext_modules.extend(mypyc_extensions)

setup(
    name="polyis",
    version="0.1.0",
    cmdclass={
        'clean': CleanCommand,
        'clean_annotate': CleanAnnotateCommand,
        'build_ext': BuildExt,
        'doc': DocCommand,
    },
    ext_modules=ext_modules,
    zip_safe=False,
)

