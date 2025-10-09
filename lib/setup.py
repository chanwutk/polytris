from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy


extensions = [
    Extension(
        "pack_append",
        ["pack_append.pyx"],
        include_dirs=[numpy.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_2_3_API_VERSION")],
        extra_compile_args=["-O3", "-ffast-math", "-march=native", "-mtune=native"],
    ),
    Extension(
        "group_tiles",
        ["group_tiles.pyx"],
        include_dirs=[numpy.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_2_3_API_VERSION")],
        extra_compile_args=["-O3", "-ffast-math", "-march=native", "-mtune=native"],
    ),
    Extension(
        "render",
        ["render.pyx"],
        include_dirs=[numpy.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_2_3_API_VERSION")],
        extra_compile_args=["-O3", "-ffast-math", "-march=native", "-mtune=native"],
    )
]

setup(
    name="polyis_pack_cython",
    version="0.1.0",
    ext_modules=cythonize(
        extensions,
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
