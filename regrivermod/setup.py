from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

import numpy as np

extNames = ['storage', 'utility', 'users', 'sdp', 'simulation']

def makeHtml(extName):
    extPath = extName + ".pyx", 
    cythonize(extPath, include_path = [np.get_include(), "..", "../include"], test="-a")

def makeExtension(extName):
    extPath = extName + ".pyx"
    return Extension(
        extName,
        [extPath],
        include_dirs = [np.get_include(), "..", "../include"],
        extra_compile_args = ["-ffast-math", "-march=native" ],
        libraries = ["m",]
        )

extensions = [makeExtension(name) for name in extNames]

setup(
  name = 'regrivermod',
  packages =["regrivermod"],
  cmdclass = {'build_ext': build_ext}, 
  ext_modules = extensions, 
) 

