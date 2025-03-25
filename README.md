# OpenImpala

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.1016/j.softx.2021.100729.svg)](https://doi.org/10.1016/j.softx.2021.100729)
OpenImpala is an open-source, image-based, parallelizable solver for calculating effective transport properties (like diffusivity, conductivity) directly from 3D microstructural data (e.g., CT, FIB-SEM). It uses a finite difference/volume approach on the voxel grid, making it suitable for complex geometries where analytical models may be inaccurate and for large datasets requiring parallel computation.

---

## Table of Contents

* [Features](#features)
* [Installation](#installation)
    * [Dependencies](#dependencies)
    * [Building from Source](#building-from-source)
* [Quick Start / Basic Usage](#quick-start--basic-usage)
    * [Input Data](#input-data)
    * [Running a Calculation](#running-a-calculation)
    * [Output](#output)
* [Applications & Related Publications](#applications--related-publications)
* [Contributing](#contributing)
* [Citation](#citation)
* [License](#license)

---

## Features

* Calculates effective transport properties (e.g., effective diffusivity, electrical/thermal conductivity) based on steady-state physics.
* Operates directly on segmented 3D image stacks (e.g., TIFF, DAT, HDF5 - requires appropriate reader).
* Massively parallel execution using MPI for analyzing large datasets efficiently.
* Uses a finite difference / finite volume method directly on the voxel grid (via AMReX).
* Suitable for complex, heterogeneous microstructures.
* Can be used for Representative Elementary Volume (REV) analysis.

---

## Installation

### Dependencies

OpenImpala requires the following dependencies:

* A modern C++ compiler supporting C++17 (e.g., GCC >= 7, Clang >= 6).
* A Fortran compiler compatible with your C++ compiler and MPI library (e.g., gfortran).
* CMake (version 3.10 or later recommended).
* An MPI library implementation (e.g., OpenMPI, MPICH, Intel MPI).
* **AMReX Library:** Core dependency ([https://github.com/AMReX-Codes/amrex](https://github.com/AMReX-Codes/amrex)). Ensure `AMREX_HOME` is set for native builds or AMReX is findable by CMake.
* **HYPRE Library:** Required for the default linear solver ([https://github.com/hypre-space/hypre](https://github.com/hypre-space/hypre)). Ensure `HYPRE_HOME` is set or Hypre is findable.
* **LibTIFF Library:** Required for reading TIFF input files (development package needed, e.g., `libtiff-dev`, `libtiff-devel`).
* **HDF5 Library:** Required for reading HDF5 input datasets (C and C++ bindings, development package needed). Needed if `ENABLE_HDF5=ON` in CMake.
* *(Optional)* **Boost Filesystem:** May be required depending on internal path handling (if `std::filesystem` is not fully used).
* *(Optional)* Other libraries depending on enabled features.

### Building from Source

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/kramergroup/openImpala.git](https://github.com/kramergroup/openImpala.git) # <-- VERIFY/REPLACE with correct URL
    cd openImpala
    ```

2.  **Configure using CMake:** Create a build directory.
    ```bash
    mkdir build && cd build
    # Basic configuration:
    cmake .. -DCMAKE_INSTALL_PREFIX=../install -DCMAKE_BUILD_TYPE=Release # Example install prefix & type
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
    The executable (e.g., `Diffusion`) will be in `build/apps/` or `../install/bin` if installed.

---

## Quick Start / Basic Usage

Here's a minimal example of calculating the effective diffusivity of a porous material from a segmented image using the `Diffusion` application.

### Input Data

* Requires a segmented 3D image stack (e.g., TIFF, DAT, HDF5).
* The image should represent different material phases with distinct integer voxel values.
* For the `Diffusion` app using the default setup, typically:
    * Voxel value representing the non-conducting/inactive phase (e.g., `0`).
    * Voxel value representing the conducting/active phase (e.g., `1`). Use the `phase_id` parameter in the `inputs` file to specify which phase to analyze.
* Ensure the file format, dimensions, and data type match the reader being used (e.g., provide metadata for `.raw`/`.dat` files via the `inputs` file).

### Running a Calculation

The Diffusion application is configured using a text input file (see below or `apps/diffusion/inputs` example).

```bash
# Example: Calculate effective diffusivity for phase '1' in a binary TIFF image
# Run using 8 parallel processes
# Ensure Diffusion executable is in your PATH or use relative/absolute path
mpirun -np 8 Diffusion inputs
```

Create an `inputs` file based on the example below or in `apps/diffusion/`.
Modify `filename`, `data_path`, `phase_id`, `direction`, `results_dir` etc. in your inputs file as needed.

### Example `inputs` File
```bash
# Input file for Diffusion application

filename = "data/SampleData_2Phase.tif" # Input file relative to execution or data_path
data_path = "."                         # Path prefix for filename

phase_id = 1                            # Phase to analyze
threshold_value = 1.0                   # Threshold used by reader (adjust if needed)

direction = "All"                       # Compute for X, Y, Z ("X", "Y", "Z", "All")

solver_type = "FlexGMRES"               # HYPRE solver (GMRES, FlexGMRES, Jacobi, PCG)
hypre_eps = 1e-9
hypre_maxiter = 200

box_size = 32                           # AMReX max grid size

results_dir = "./Diffusion_Results/"    # Directory for output text/plot files
output_filename = "results.txt"         # Name for summary output file
write_plotfile = 0                      # Set to 1 to save AMReX plot files

verbose = 1                             # Output level
```
## Output

* **Console**: Prints progress information, calculated Volume Fraction, calculated Tortuosity for each direction, and total runtime.
* **Results File**: Writes key input parameters and final calculated results (VF, Tortuosity) to the file specified by `output_filename` within the `results_dir`. Example (`results.txt`):

```bash
# Diffusion Calculation Results
# Input File: data/SampleData_2Phase.tif
# Analysis Phase ID: 1
# Threshold Value: 1.0
# Solver: FlexGMRES
# ... other parameters ...
# -----------------------------
VolumeFraction: 0.512345678
Tortuosity_X: 1.987654321
Tortuosity_Y: 1.998765432
Tortuosity_Z: 1.976543210
```
* **Plotfiles (Optional)**: If `write_plotfile = 1`, AMReX plotfiles (e.g., potential field) may be generated in `results_dir`. These can be viewed with ParaView, VisIt, yt, etc.

---

# Applications & Related Publications

OpenImpala has been used in various research projects. Here are some examples:
* **Optimising Aqueous Al-Ion Battery Electrodes**: OpenImpala was used in a high-throughput study to analyse the effect of compression on carbon felt electrodes using synchrotron CT data (99 tomograms). It calculated porosity, tortuosity, volume-specific area, ionic diffusivity, and electrical conduction to help predict the optimal compression ratio.
  * Le Houx, J., Melzack, N., James, A., Dehyle, H., Aslani, N., Pimblott, M., Ahmed, S., & Wills, R. G. A. (2024). The Aqueous Aluminium-Ion Battery: Optimising the Electrode Compression Ratio through Image-Based Modelling. ECS Meeting Abstracts, MA2024-01, 2579. https://doi.org/10.1149/MA2024-01462579mtgabs
* **Modelling Soluble Lead Flow Battery Electrodes**: OpenImpala was used to calculate porosity and effective tortuosity from CT scans of Reticulated Vitreous Carbon (RVC) electrodes, including virtual deposit growth via dilation. These properties were then used as inputs for a larger COMSOL simulation to study battery performance.
  * Fraser, E. J., Le Houx, J. P., Arenas, L. F., Ranga Dinesh, K. K. J., & Wills, R. G. A. (2022). The soluble lead flow battery: Image-based modelling of porous carbon electrodes. Journal of Energy Storage, 52, 104791. https://doi.org/10.1016/j.est.2022.104791
* **Investigating Tomography Resolution Effects**: OpenImpala was employed to study how the spatial resolution of CT scans affects the calculated porosity and tortuosity of Lithium Iron Phosphate (LFP) electrodes, comparing results against the Bruggeman correlation.
  * Le Houx, J., Osenberg, M., Neumann, M., Binder, J. R., Schmidt, V., Manke, I., Carraro, T., & Kramer, D. (2020). Effect of Tomography Resolution on Calculation of Microstructural Properties for Lithium Ion Porous Electrodes. ECS Transactions, 97(7), 255. https://doi.org/10.1149/09707.0255ecst   
* **Statistical Effective Diffusivity**: Enhancements and validation of OpenImpala for calculating effective diffusivity using homogenization theory, suitable for on-site synchrotron workflows.
  * Le Houx, J., Ruiz, S., McKay Fletcher, D., Ahmed, S., & Roose, T. (2023). Statistical Effective Diffusivity Estimation in Porous Media Using an Integrated On-site Imaging Workflow for Synchrotron Users. Transport in Porous Media, 150, 71–88. https://doi.org/10.1007/s11242-023-01993-7   
 
---

# Contributing

Contributions are welcome! Please refer to the GitHub repository's issue tracker for bug reports and feature requests. If you'd like to contribute code, please fork the repository, create a feature branch, and submit a pull request for review. Adherence to the Code of Conduct is expected.
---

# Citation

If you use OpenImpala in your research, please cite the relevant publication(s):

Primary Software Paper:

```bash
@article{LeHoux2021OpenImpala,
  title = {{{OpenImpala}}: {{OPEN}} source {{IMage}} based {{PArallisable}} {{Linear}} {{Algebra}} solver},
  author = {Le Houx, James and Kramer, Denis},
  year = {2021},
  journal = {SoftwareX},
  volume = {15},
  pages = {100729},
  doi = {10.1016/j.softx.2021.100729},
  issn = {2352-7110}
}
```
Le Houx, J., & Kramer, D. (2021). OpenImpala: OPEN source IMage based PArallisable Linear Algebra solver. SoftwareX, 15, 100729. https://doi.org/10.1016/j.softx.2021.100729

(Optional) For Effective Diffusivity Calculation Method:

```bash
@article{le2023statistical,
  title={Statistical Effective Diffusivity Estimation in Porous Media Using an Integrated On-site Imaging Workflow for Synchrotron Users},
  author={Le Houx, James and Ruiz, Siul and McKay Fletcher, Daniel and Ahmed, Sharif and Roose, Tiina},
  journal={Transport in Porous Media},
  volume={150},
  number={1},
  pages={71--88},
  year={2023},
  publisher={Springer},
  doi={10.1007/s11242-023-01993-7}
}
```
Le Houx, J., Ruiz, S., McKay Fletcher, D., Ahmed, S., & Roose, T. (2023). Statistical Effective Diffusivity Estimation in Porous Media Using an Integrated On-site Imaging Workflow for Synchrotron Users. Transport in Porous Media, 150, 71–88. https://doi.org/10.1007/s11242-023-01993-7   

