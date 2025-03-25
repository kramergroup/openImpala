# OpenImpala

<img src="https://user-images.githubusercontent.com/37665786/93309082-002ca800-f7fb-11ea-9ce7-d57b3e80c6ec.jpg" width="800" />

[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![DOI](https://zenodo.org/badge/DOI/10.1016/j.softx.2021.100729.svg)](https://doi.org/10.1016/j.softx.2021.100729)
OpenImpala is an open-source, image-based, parallelizable solver for calculating effective transport properties (like diffusivity, conductivity) directly from 3D microstructural data (e.g., CT, FIB-SEM). It uses a finite difference/volume approach on the voxel grid via the [AMReX library](https://github.com/AMReX-Codes/amrex), making it suitable for complex geometries where analytical models may be inaccurate and for large datasets requiring parallel computation.

---

## Table of Contents

* [Features](#features)
* [Installation](#installation)
    * [Dependencies](#dependencies)
    * [Building from Source](#building-from-source)
* [Quick Start / Basic Usage](#quick-start--basic-usage)
    * [Input Data](#input-data)
    * [Running a Calculation](#running-a-calculation)
    * [Example `inputs` File](#example-inputs-file)
    * [Output](#output)
* [Applications & Related Publications](#applications--related-publications)
* [Contributing](#contributing)
* [Citation](#citation)
* [License](#license)

---

## Features

* Calculates effective transport properties (e.g., effective diffusivity, electrical/thermal conductivity) based on steady-state physics.
* Operates directly on segmented 3D image stacks (TIFF, HDF5, DAT support via internal readers).
* Massively parallel execution using MPI via the AMReX framework.
* Finite difference / finite volume method on the voxel grid.
* Suitable for complex, heterogeneous microstructures.
* Can be used for Representative Elementary Volume (REV) analysis (see related publications).
* Output usable for parameterizing continuum models (e.g., PyBamm, DandeLiion).

---

## Installation

### Dependencies

OpenImpala relies on several external libraries. While a containerised environment might be available (see original docs/fork if applicable), building natively requires:

* A modern C++ compiler supporting C++17 (e.g., GCC >= 7, Clang >= 6).
* A Fortran compiler compatible with your C++ compiler and MPI library (e.g., gfortran).
* CMake (version 3.10 or later recommended).
* An MPI library implementation (e.g., OpenMPI, MPICH, Intel MPI).
* **AMReX Library:** Core dependency ([https://github.com/AMReX-Codes/amrex](https://github.com/AMReX-Codes/amrex)). Ensure `AMREX_HOME` is set or AMReX is findable by CMake.
* **HYPRE Library:** Required for the default linear solver ([https://github.com/hypre-space/hypre](https://github.com/hypre-space/hypre)). Ensure `HYPRE_HOME` is set or Hypre is findable.
* **LibTIFF Library:** Required for reading TIFF input files (development package needed, e.g., `libtiff-dev`, `libtiff-devel`).
* **HDF5 Library:** Required for reading HDF5 input datasets (C and C++ bindings, development package needed). Needed if building with HDF5 support.
* *(Optional)* **Boost Filesystem:** May be required depending on internal path handling (if `std::filesystem` from C++17 is not fully used).
* *(Optional)* Other libraries depending on enabled features.

### Building from Source

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/kramergroup/openImpala.git](https://github.com/kramergroup/openImpala.git) # <-- Verify correct URL
    cd openImpala
    ```

2.  **Configure using CMake:** Create a build directory.
    ```bash
    mkdir build && cd build
    # Basic configuration (installs to ../install relative to source):
    cmake .. -DCMAKE_INSTALL_PREFIX=../install -DCMAKE_BUILD_TYPE=Release
    # Or add paths to dependencies if not found automatically:
    # cmake .. -DCMAKE_INSTALL_PREFIX=../install -DCMAKE_BUILD_TYPE=Release \
    #       -DAMReX_DIR=/path/to/amrex/lib/cmake/AMReX \
    #       -DHYPRE_DIR=/path/to/hypre/lib/cmake/hypre \
    #       -DHDF5_DIR=/path/to/hdf5/share/cmake \
    #       -DLibTIFF_DIR=/path/to/libtiff/lib/cmake/tiff-X.Y
    ```
    * Use `ccmake ..` or `cmake-gui ..` for interactive configuration.
    * Set `CMAKE_CXX_COMPILER` and `CMAKE_Fortran_COMPILER` environment variables or CMake variables if needed (e.g., to `mpicxx`, `mpif90`).

3.  **Compile the code:**
    ```bash
    make -j4 # Adjust '-j' number based on your system's cores
    ```

4.  **Install (Optional):**
    ```bash
    make install
    ```
    The executable (e.g., `Diffusion`) will be in `build/apps/` or `../install/bin` if installed. Ensure the location is in your `PATH`.

---

## Quick Start / Basic Usage

Here's a minimal example of calculating the effective diffusivity of a porous material from a segmented image using the `Diffusion` application.

### Input Data

* Requires a segmented 3D image stack (e.g., TIFF, DAT, HDF5). Ensure the correct reader class was compiled.
* The image should represent different material phases with distinct integer voxel values.
* For the `Diffusion` app using the default setup, typically:
    * Voxel value representing the non-conducting/inactive phase (e.g., `0`).
    * Voxel value representing the conducting/active phase (e.g., `1`). Use the `phase_id` parameter in the `inputs` file to specify which phase to analyze.
* If using RAW/DAT formats, dimensions and data type metadata must be provided in the `inputs` file.

### Running a Calculation

The `Diffusion` application is configured using a text input file (commonly named `inputs`).

```bash
# Example: Calculate effective diffusivity for phase '1' in a binary TIFF image
# Run using 8 parallel processes

# Ensure Diffusion executable is in your PATH or use relative/absolute path
# Assumes 'inputs' file is in the current directory
mpirun -np 8 Diffusion inputs
