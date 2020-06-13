from distutils.extension import Extension
from distutils.core import setup
from Cython.Build import cythonize
import numpy

extensions = [
        Extension("_make_er", ["_make_er.pyx"],include_dirs=[numpy.get_include()]) # define_macros was not working here, so I put it in the pyx file
]

setup(
        ext_modules = cythonize(extensions)
)
