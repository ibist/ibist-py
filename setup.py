"""Compile the Python binding to the bundled, shared Wang C++ core."""

import os

from setuptools import Extension, setup


setup(ext_modules=[Extension(
    "ibist._wang_native", sources=["src/ibist/_native/wang_module.cpp"],
    depends=["src/ibist/_native/wang_core.h"], language="c++",
    extra_compile_args=(["/std:c++14", "/EHsc"] if os.name == "nt" else
                        ["-std=c++11"])
)])
