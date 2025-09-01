from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "pack_append",
        ["pack_append.pyx"],
        include_dirs=[numpy.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        extra_compile_args=["-O3", "-ffast-math"],
    )
]

setup(
    name="polyis_pack_cython",
    version="0.1.0",
    ext_modules=cythonize(extensions, compiler_directives={"language_level": 3}),
    zip_safe=False,
)
