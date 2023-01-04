# Python API

This folder provides a python API for the most salient features of OpenImpala. The aim is to provide simple access to the high-performance
library for a wide variety of workflows without the need to compile individual programs. 

## Compilation

The python API consists of a shared library called `pyoimp.so`. This shared library is statically linking dependencies in order to enable simple distribution. However, static dependencies need to be compiled as **position independent code** in order to be inclued in a shared library. 

The two main static dependencies are *AMReX* and *HYPRE*, both are not compiled as position independent code by default. The *AMReX* build system provides the `AMREX_PIC` configuration parameter, which needs to be set to `ON`. To compile *HYPRE*, one can set `HYPRE_WITH_EXTRA_CFLAGS=-fPIC` and `HYPRE_WITH_EXTRA_CXXFLAGS=-fPIC` to achieve a static library with position independent code.

## Installation

The `pyoimp.so` shared library must reside in a path defined in `$PYTHONPATH`. We will provide a `setup.py` file at a later point in time.

## Usage

TODO: Give a number of usage examples
