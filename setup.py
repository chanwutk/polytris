#!/usr/local/bin/python

from setuptools import setup, Extension, Command
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize
import numpy
import os
import glob
import multiprocessing as mp


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
            "polyis/pack/c/utilities_.c",
            "polyis/pack/c/group_tiles_.c",
        ],
        include_dirs=["polyis/cbinpack", numpy.get_include()],
        define_macros=MACROS,
        extra_compile_args=ARGS + ["-std=c11"],
    ),
    Extension(
        "polyis.pack.adapters",
        [
            "polyis/pack/adapters.pyx",
            "polyis/pack/c/utilities_.c",
        ],
        include_dirs=["polyis/cbinpack", numpy.get_include()],
        define_macros=MACROS,
        extra_compile_args=ARGS + ["-std=c11"],
    ),
    Extension(
        "polyis.pack.pack_ffd",
        [
            "polyis/pack/pack_ffd.pyx",
            "polyis/pack/c/utilities_.c",
            "polyis/pack/c/pack_ffd_.c",
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
        # 'polyis/pack/cython/**/*.so',
        # 'polyis/pack/cython/*.so',
        # 'polyis/pack/c/**/*.so',
        # 'polyis/pack/c/*.so',
        'polyis/pack/**/*.so',
        'polyis/pack/*.so',
    ]
    html_patterns = [
        # 'polyis/pack/cython/**/*.html',
        # 'polyis/pack/cython/*.html',
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


class BuildExt(build_ext):
    """Custom build_ext command that defaults to --inplace and cleans artifacts."""
    
    def initialize_options(self):
        super().initialize_options()
        # Set inplace to True by default
        self.inplace = True


setup(
    name="polyis",
    version="0.1.0",
    cmdclass={
        'clean': CleanCommand,
        'clean_annotate': CleanAnnotateCommand,
        'build_ext': BuildExt,
    },
    ext_modules=cythonize(
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
    ),
    zip_safe=False,
)

