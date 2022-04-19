#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 19:15:39 2021

@author: cimat-mty
"""
from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name='utils',
    ext_modules=cythonize("_utils.pyx"),
    include_dirs=[numpy.get_include()],
    zip_safe=False,
)