# OpenImpala

<img src="https://user-images.githubusercontent.com/37665786/93309082-002ca800-f7fb-11ea-9ce7-d57b3e80c6ec.jpg" width="800" />

[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![DOI](https://img.shields.io/badge/DOI-10.1016/j.softx.2021.100729-blue)](https://doi.org/10.1016/j.softx.2021.100729)

A common challenge from image-based modelling is the size of 3D tomography datasets, which can be of the order of several billion voxels. OpenImpala is a data-driven, fully parallelisable, image-based modelling framework designed specifically for these high computational cost problems, built upon the [AMReX library](https://github.com/AMReX-Codes/amrex).

3D datasets are used as the computational domain within a finite-differences-based model to directly solve physical equations (like steady-state diffusion/conduction) on the image dataset, removing the need for additional meshing.

OpenImpala then calculates the equivalent homogenised transport coefficients (e.g., effective diffusivity, tortuosity, conductivity) for the given microstructure. These coefficients can be written into parameterised files for direct compatibility with popular continuum battery models like [PyBamm](https://github.com/pybamm-team/PyBaMM) and [DandeLiion](https://github.com/tinosulzer/DandeLiion), facilitating the link between different computational battery modelling scales.

OpenImpala has been shown to scale well with an increasing number of computational cores on distributed memory architectures using MPI, making it applicable to large datasets typical of modern tomography.

---

## Citation

If you've used OpenImpala in the preparation of a publication, please consider citing this publication: https://doi.org/10.1016/j.softx.2021.100729 

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
* Massively parallel using MPI via the AMReX framework.
* Finite difference / finite volume method on the voxel grid.
* Output usable for parameterizing continuum models (e.g., PyBamm, DandeLiion).
* Includes tools for Volume Fraction calculation and basic image reading tests.

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
    git clone [https://github.com/kramergroup/openImpala.git](https://github.com/kramergroup/openImpala.git) # <-- VERIFY/REPLACE with correct URL
    cd openImpala
    ```

2.  **Configure using CMake:** Create a build directory.
    ```bash
    mkdir build && cd build
    # Basic configuration (installs to ../install relative to source):
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
```

* Create an inputs file based on the example below or from `apps/diffusion/inputs` in the `source` tree.
* Modify parameters within your `inputs` file as needed for your specific case.

### Example `inputs` File

```inputs
# Example input file for Diffusion application

# --- Input File ---
filename = "data/SampleData_2Phase.tif" # Input file relative to execution or data_path
data_path = "."                         # Path prefix for filename
# Required for HDF5:
# hdf5_dataset = "exchange/data"
# Required for RAW/DAT:
# raw_width = 100; raw_height = 100; raw_depth = 100; raw_datatype = "UINT8"
# Required for TIFF sequence:
# tiff_stack_size = 100 # If filename is base pattern

# --- Analysis ---
phase_id = 1                            # Phase to analyze
threshold_value = 1.0                   # Threshold used by image readers (if applicable)
direction = "All"                       # Compute for X, Y, Z ("X", "Y", "Z", "All", or "X Z" etc.)

# --- Solver ---
solver_type = "FlexGMRES"               # HYPRE solver (GMRES, FlexGMRES, Jacobi, PCG)
hypre_eps = 1e-9                        # Solver relative tolerance
hypre_maxiter = 200                     # Solver max iterations

# --- Grid ---
box_size = 32                           # AMReX max grid size

# --- Output ---
results_dir = "./Diffusion_Results/"    # Directory for output text/plot files ('~/' supported)
output_filename = "results.txt"         # Name for summary output file
write_plotfile = 0                      # Set to 1 to save AMReX plot files from solver

# --- Control ---
verbose = 1                             # Output level (0=minimal)
```
### Output

The `Diffusion` application produces results primarily through console messages and output files:

* **Console:**
    * Prints progress information during setup and solver execution (level controlled by the `verbose` input parameter).
    * Reports final calculated Volume Fraction (`VF`) for the specified `phase_id`.
    * Reports final calculated Tortuosity (`Tau`) for each computed direction.
    * May print solver diagnostics like number of iterations and final residual, especially if `verbose > 0`.
    * Reports total wall-clock runtime for the simulation.

* **Results File:**
    * A summary text file is written to the location specified by `results_dir` and `output_filename` in the `inputs` file.
    * This file typically includes a record of the key input parameters used for the run (e.g., filename, phase, direction, solver settings) followed by the final calculated results (Volume Fraction, Tortuosity components).
    * **Example** (`results.txt`, content may vary slightly):
        ```
        # Diffusion Calculation Results
        # Input File: data/SampleData_2Phase.tif
        # Analysis Phase ID: 1
        # Threshold Value: 1.0
        # Solver: FlexGMRES
        # ... other key parameters ...
        # -----------------------------
        # Note: Property values are typically non-dimensionalized or assume unit intrinsic properties.
        # Check units/scaling based on solver details.
        VolumeFraction: 0.512345678
        Tortuosity_X: 1.987654321
        Tortuosity_Y: 1.998765432
        Tortuosity_Z: 1.976543210
        ```

* **Plotfiles (Optional):**
    * If `write_plotfile = 1` in the `inputs` file, AMReX plotfiles may be generated in the `results_dir`.
    * These typically contain the final computed field (e.g., the potential field, named perhaps `potential_X`, `potential_Y`, or `potential_Z` depending on the run) and often include the thresholded phase map (`phase_threshold`) used for the calculation.
    * Plotfile names might include direction identifiers or timestamps.
    * View these files using standard visualization tools like ParaView, VisIt, yt (see [AMReX Visualization Docs](https://amrex-codes.github.io/amrex/docs_html/Visualization.html)).
 
---

## Visualisation

OpenImpala is built on the AMReX software framework. Output plotfiles (generated when `write_plotfile = 1`) can be visualised using several open-source visualisation packages, e.g. [ParaView](https://www.paraview.org/), [VisIt](https://visit.llnl.gov/), [yt](https://yt-project.org/) or AMRVis.

* **AMReX Documentation:** For further information on native AMReX plotfile formats and viewing options, see the [AMReX Visualization Docs](https://amrex-codes.github.io/amrex/docs_html/Visualization.html).
* **Jupyter Notebooks:** Alternatively, you can use Jupyter notebooks for analysis and visualisation. A guide demonstrating how to load and plot data from OpenImpala output is available here: [https://github.com/jameslehoux/openimpala-jupyter](https://github.com/jameslehoux/openimpala-jupyter).

As an example, the image below shows a calculated concentration gradient for steady-state diffusive flow (solved in the x-direction) within a 499^3 voxel Lithium Iron Phosphate (LFP) electrode microstructure (Source: [1]):

<img src="https://user-images.githubusercontent.com/37665786/93310161-577f4800-f7fc-11ea-8b8c-3cae084f18a5.png" width="800" alt="3D visualisation showing a concentration gradient across a porous microstructure, ranging from blue (low concentration) on one side to red (high concentration) on the other." />

[1]: Le Houx, J., Osenberg, M., Neumann, M., Binder, J.R., Schmidt, V., Manke, I., Carraro, T. and Kramer, D., 2020. Effect of Tomography Resolution on Calculation of Microstructural Properties for Lithium Ion Porous Electrodes. *ECS Transactions*, 97(7), p.255.

## Applications & Related Publications

OpenImpala is actively used in research. If you use OpenImpala in work leading to a publication, please cite the core software paper(s) (see Citation section) and consider letting the developers know or submitting a pull request to add your work here!

Below are some examples of publications using or discussing OpenImpala:

* Le Houx, J., Melzack, N., James, A., Dehyle, H., Aslani, N., Pimblott, M., Ahmed, S., & Wills, R. G. A. (2024). **The Aqueous Aluminium-Ion Battery: Optimising the Electrode Compression Ratio through Image-Based Modelling.** *ECS Meeting Abstracts*, *MA2024-01*, 2579. [https://doi.org/10.1149/MA2024-01462579mtgabs](https://doi.org/10.1149/MA2024-01462579mtgabs) *(Application: Aqueous Al-ion battery electrode compression analysis)*

* Le Houx, J., Ruiz, S., McKay Fletcher, D., Ahmed, S., & Roose, T. (2023). **Statistical Effective Diffusivity Estimation in Porous Media Using an Integrated On-site Imaging Workflow for Synchrotron Users.** *Transport in Porous Media*, *150*, 71–88. [https://doi.org/10.1007/s11242-023-01993-7](https://doi.org/10.1007/s11242-023-01993-7) *(Methodology: Effective diffusivity calculation & validation)*

* Fraser, E. J., Le Houx, J. P., Arenas, L. F., Ranga Dinesh, K. K. J., & Wills, R. G. A. (2022). **The soluble lead flow battery: Image-based modelling of porous carbon electrodes.** *Journal of Energy Storage*, *52*, 104791. [https://doi.org/10.1016/j.est.2022.104791](https://doi.org/10.1016/j.est.2022.104791) *(Application: Soluble lead flow battery RVC electrodes)*

* Le Houx, J., & Kramer, D. (2021). **OpenImpala: OPEN source IMage based PArallisable Linear Algebra solver.** *SoftwareX*, *15*, 100729. [https://doi.org/10.1016/j.softx.2021.100729](https://doi.org/10.1016/j.softx.2021.100729) *(Core Software Paper)*

* Le Houx, J., Osenberg, M., Neumann, M., Binder, J. R., Schmidt, V., Manke, I., Carraro, T., & Kramer, D. (2020). **Effect of Tomography Resolution on Calculation of Microstructural Properties for Lithium Ion Porous Electrodes.** *ECS Transactions*, *97*(7), 255. [https://doi.org/10.1149/09707.0255ecst](https://doi.org/10.1149/09707.0255ecst) *(Application: LFP electrode resolution effects)*

## Contributing

Contributions to OpenImpala are welcome! Whether it's reporting bugs, suggesting features, improving documentation, or submitting code, your input is valuable.

* **Bug Reports & Feature Requests:** Please use the [GitHub Issues tracker](https://github.com/kramergroup/openImpala/issues) to report problems or propose new features. Provide as much detail as possible, including steps to reproduce for bugs.
* **Code Contributions:**
    * If you plan to make significant changes, please open an issue first to discuss your ideas.
    * For code contributions (bug fixes, enhancements, new features, tests), please follow this general workflow:
        1.  Fork the repository ([https://github.com/kramergroup/openImpala](https://github.com/kramergroup/openImpala)).
        2.  Create a new branch for your feature or fix (`git checkout -b feature/my-new-feature`).
        3.  Make your changes and commit them with clear messages.
        4.  Push your branch to your fork (`git push origin feature/my-new-feature`).
        5.  Submit a [Pull Request](https://github.com/kramergroup/openImpala/pulls) to the main repository.
* **Documentation:** Improvements to the README, code comments, or other documentation are always appreciated. You can submit these via Pull Requests.

## Citation

If you use OpenImpala in your research or publications, we kindly ask that you cite the relevant paper(s).

1.  **General Use & Software Framework:** Please cite this paper when using OpenImpala for simulations or analysis based on its core functionality.
    ```bibtex
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
    Le Houx, J., & Kramer, D. (2021). OpenImpala: OPEN source IMage based PArallisable Linear Algebra solver. *SoftwareX*, *15*, 100729. [![DOI](https://img.shields.io/badge/DOI-10.1016/j.softx.2021.100729-blue)](https://doi.org/10.1016/j.softx.2021.100729)

2.  **Specific Methods (e.g., Effective Diffusivity via Homogenization):** Please consider citing this paper *in addition* to the primary software paper if you are using or comparing against the specific homogenization methods or on-site workflow described therein.
    ```bibtex
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
    Le Houx, J., Ruiz, S., McKay Fletcher, D., Ahmed, S., & Roose, T. (2023). Statistical Effective Diffusivity Estimation in Porous Media Using an Integrated On-site Imaging Workflow for Synchrotron Users. *Transport in Porous Media*, *150*, 71–88. [![DOI](https://img.shields.io/badge/DOI-10.1007/s11242--023--01993--7-blue)](https://doi.org/10.1007/s11242-023-01993-7)

## License

OpenImpala Copyright (c) 2020-2025, University of Southampton and contributors.
All rights reserved.

The software is licensed under the **BSD 3-Clause "New" or "Revised" License**. The full license text can be found in the [LICENSE](LICENSE) file.

## Acknowledgements

This work was financially supported by the EPSRC Centre for Doctoral Training (CDT) in Energy Storage and its Applications [grant ref: EP/R021295/1], the Ada Lovelace Centre (ALC) STFC project, CANVAS-NXtomo, ContAiNerised Voxel-bAsed Simulation of Neutron and X-ray Tomography data, and part funded by the EPSRC prosperity partnership with Imperial College, INFUSE, Interface with the Future - Underpinning Science to Support the Energy transition EP/V038044/1.

The authors acknowledge the use of the IRIDIS High Performance Computing Facility, and associated support services at the University of Southampton, in the completion of this work.

We thank the developers of [AMReX](https://github.com/AMReX-Codes/amrex), [HYPRE](https://github.com/hypre-space/hypre), [libtiff](http://www.libtiff.org/), and [HDF5](https://www.hdfgroup.org/solutions/hdf5/) upon which OpenImpala relies.

---

## Contact & Support

For questions, bug reports, or feature requests, please use the [GitHub Issues tracker](https://github.com/kramergroup/openImpala/issues) for this repository.
