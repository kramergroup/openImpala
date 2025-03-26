# OpenImpala

<img src="https://user-images.githubusercontent.com/37665786/93309082-002ca800-f7fb-11ea-9ce7-d57b3e80c6ec.jpg" width="800" />

[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![DOI](https://img.shields.io/badge/DOI-10.1016/j.softx.2021.100729-blue)](https://doi.org/10.1016/j.softx.2021.100729)
[![GitHub release (latest by date)](https://img.shields.io/github/v/release/kramergroup/openImpala)](https://github.com/kramergroup/openImpala/releases/latest)
[![GitHub contributors](https://img.shields.io/github/contributors/kramergroup/openImpala)](https://github.com/kramergroup/openImpala/graphs/contributors)
[![Pipeline Status](https://gitlab.com/JleHoux/openImpala/badges/master/pipeline.svg)](https://gitlab.com/JleHoux/openImpala/commits/master)
OpenImpala is a high-performance computing framework for image-based modelling, built upon the [AMReX library](https://github.com/AMReX-Codes/amrex) for massive parallelism using MPI. It tackles the challenge posed by large 3D imaging datasets (often billions of voxels) common in materials science and tomography.

OpenImpala directly solves physical equations, such as steady-state diffusion or conduction problems, on the voxel grid of the input image using finite differences. This approach bypasses the need for explicit mesh generation, working directly with the acquired image data. From the simulation results, it calculates effective homogenised transport properties (e.g., diffusivity, conductivity, tortuosity) characteristic of the microstructure.

These calculated coefficients can directly parameterize continuum-scale models, notably battery simulators like [PyBamm](https://github.com/pybamm-team/PyBaMM) and [DandeLiion](https://github.com/tinosulzer/DandeLiion). This capability effectively bridges microstructural details obtained from imaging to device-level performance predictions. OpenImpala is designed for excellent scalability on distributed memory systems, making the analysis of large, high-resolution datasets feasible.

---

## Table of Contents

* [Features](#features)
* [Getting Started (Recommended: Singularity)](#getting-started-recommended-singularity)
    * [Singularity Container](#singularity-container)
    * [Building with Singularity](#building-with-singularity)
    * [Running Tests with Singularity](#running-tests-with-singularity)
    * [Running the Application with Singularity](#running-the-application-with-singularity)
* [Native Installation (Advanced)](#native-installation-advanced)
    * [Dependencies](#dependencies)
    * [Building from Source (CMake)](#building-from-source-cmake)
* [Batch Processing (HPC)](#batch-processing-hpc)
* [Example `inputs` File](#example-inputs-file)
* [Output](#output)
* [Visualisation](#visualisation)
* [Applications & Related Publications](#applications--related-publications)
* [Continuous Integration](#continuous-integration)
* [Contributing](#contributing)
* [Citation](#citation)
* [License](#license)
* [Acknowledgements](#acknowledgements)
* [Contact & Support](#contact--support)

---

## Features

* Calculates effective transport properties (e.g., effective diffusivity, electrical/thermal conductivity) based on steady-state physics.
* Operates directly on segmented 3D image stacks (TIFF, HDF5, DAT support via internal readers).
* Massively parallel using MPI via the AMReX framework.
* Finite difference / finite volume method on the voxel grid.
* Output usable for parameterizing continuum models (e.g., PyBamm, DandeLiion).
* Includes tools for Volume Fraction calculation and basic image reading tests.

---

## Getting Started (Recommended: Singularity)

The easiest way to get started is by using the provided Singularity container, which includes all necessary dependencies and a pre-built environment.

### Singularity Container

A containerised build environment providing all dependencies is available on Sylabs Cloud. *(Note: Verify Sylabs link/availability if using the badge above)*.

The container includes: Centos 7, OpenMPI, Hypre, LibTiff, AMReX, HDF5, h5cpp, InfiniBand support, and a pre-built OpenImpala.

1.  **Install Singularity:** Ensure you have Singularity (version 3.x recommended) installed on your Linux system or HPC. See [Singularity documentation](https://sylabs.io/docs/).

2.  **Pull the Image:**
    ```bash
    singularity pull library://jameslehoux/default/openimpala:latest
    ```
    This downloads the `openimpala_latest.sif` image file to your current directory.

### Building with Singularity

While the container includes a pre-built version, you can also compile your local source code using the container's environment:

1.  **Start an interactive shell** within the container, mounting your local source code directory (replace `/path/to/local/openImpala` with the actual path).
    ```bash
    # Mount current directory (.) containing source code to /src inside container
    singularity shell --bind "$(pwd):/src" openimpala_latest.sif
    ```

2.  **Navigate and Build:** Inside the Singularity shell, go to your source directory and use `make`.
    ```bash
    Singularity> cd /src
    Singularity> make
    ```
    This will create a `build/` directory (relative to `/src`) containing object files and executables compiled using the container's tools.

### Running Tests with Singularity

After building (or using the pre-built version), you can run the tests.

* **Option 1: Inside Singularity Shell:** If you are already inside the shell (`singularity shell ...`):
    ```bash
    Singularity> cd /path/to/openImpala/build/tests # Use path to built tests
    Singularity> ./tTiffReader
    Singularity> ./tVolumeFraction
    Singularity> ./tTortuosity
    ```

* **Option 2: Using `singularity exec`:** Run directly from your host shell:
    ```bash
    singularity exec openimpala_latest.sif /openImpala/build/tests/tTiffReader
    singularity exec openimpala_latest.sif /openImpala/build/apps/Diffusion
    singularity exec openimpala_latest.sif /openImpala/build/tests/tVolumeFraction
    singularity exec openimpala_latest.sif /openImpala/build/tests/tTortuosity
    ```
    *(Note: Ensure paths like `/openImpala/build/...` correctly point to the executables *inside* the container image).*

### Running the Application with Singularity

The main application is `Diffusion`, configured via an `inputs` file.

1.  **Prepare `inputs` file:** Create or copy an `inputs` file (see [Example `inputs` File](#example-inputs-file) section below) on your *host* machine. Adjust parameters like `filename`, `data_path`, `results_dir` to point to locations accessible *from within the container* (often achieved by mounting host directories).

2.  **Run using `singularity exec`:** Use the `-B` flag (or `--bind`) to mount necessary host directories into the container.
    ```bash
    # Example: Mount current host directory to /host_pwd inside container
    # Assumes 'inputs' file and 'data/' directory are in the current host directory
    # Assumes results will be written relative to the mounted directory

    # Run sequentially
    singularity exec -B "$(pwd):/host_pwd" openimpala_latest.sif \
        /openImpala/build/apps/Diffusion /host_pwd/inputs

    # Run in parallel using MPI (requires MPI inside container + host setup)
    # Ensure OMP_NUM_THREADS=1 if using multiple MPI ranks
    export OMP_NUM_THREADS=1
    mpirun -np 4 singularity exec -B "$(pwd):/host_pwd" openimpala_latest.sif \
        /openImpala/build/apps/Diffusion /host_pwd/inputs
    ```
    *Modify the `-B` mount points (`host_path:container_path`) according to where your data/input/output directories reside on the host.*
    *Ensure paths used *inside* the `inputs` file (like `data_path`, `results_dir`) correspond to paths *within the container* (e.g., `/host_pwd/data/`, `/host_pwd/results/`).*

---

## Native Installation (Advanced)

Building natively requires manually installing all dependencies.

### Dependencies

* A modern C++ compiler supporting C++17 (e.g., GCC >= 7, Clang >= 6).
* A Fortran compiler compatible with your C++ compiler and MPI library (e.g., gfortran).
* CMake (version 3.10 or later recommended).
* An MPI library implementation (e.g., OpenMPI, MPICH, Intel MPI).
* **AMReX Library:** Core dependency ([https://github.com/AMReX-Codes/amrex](https://github.com/AMReX-Codes/amrex)). Ensure `AMREX_HOME` is set or AMReX is findable by CMake.
* **HYPRE Library:** Required for the default linear solver ([https://github.com/hypre-space/hypre](https://github.com/hypre-space/hypre)). Ensure `HYPRE_HOME` is set or Hypre is findable.
* **LibTIFF Library:** Required for reading TIFF input files (development package needed, e.g., `libtiff-dev`, `libtiff-devel`).
* **HDF5 Library:** Required for reading HDF5 input datasets (C and C++ bindings, development package needed). Needed if building with HDF5 support.
* *(Optional)* **Boost Filesystem:** May be required depending on internal path handling.
* *(Optional)* Other libraries depending on enabled features.

### Building from Source (CMake)

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/kramergroup/openImpala.git](https://github.com/kramergroup/openImpala.git) # <-- VERIFY/REPLACE URL
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
    * Set `CMAKE_CXX_COMPILER`/`CMAKE_Fortran_COMPILER` (e.g., to `mpicxx`, `mpif90`).

3.  **Compile the code:**
    ```bash
    make -j4 # Adjust job count
    ```

4.  **Run Tests (Optional):**
    ```bash
    make test # Or ctest
    # Or run individually: ./tests/tTiffReader etc.
    ```

5.  **Install (Optional):**
    ```bash
    make install
    ```
    The executable (e.g., `Diffusion`) will be in `build/apps/` or `../install/bin`. Ensure the location is in your `PATH`.

## Batch Processing (HPC)

The `Diffusion` application runs non-interactively, making it suitable for HPC batch jobs. An example SLURM script snippet using Singularity:

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20 # Request 20 cores on one node
#SBATCH --time=01:00:00
#SBATCH --job-name=openimpala_diffusion

module load singularity/3.2.1 # Load required singularity module on HPC

export OMP_NUM_THREADS=1 # Ensure OpenMP threading is disabled if using pure MPI

# Navigate to directory containing the 'inputs' file
cd /path/to/your/simulation/directory/

# Define path to singularity image (e.g., absolute path)
SIF_IMAGE=/path/on/hpc/to/openimpala_latest.sif

# Define path to executable inside the container
APP=/openImpala/build/apps/Diffusion

# Define path to inputs file (e.g., absolute path or relative to working dir)
INPUT_FILE=./inputs

# Run the calculation using mpirun and singularity exec
# Mount necessary directories (e.g., current directory for inputs/outputs)
mpirun -np <span class="math-inline">SLURM\_NTASKS singularity exec \-B "</span>(pwd)":"$(pwd)" $SIF_IMAGE $APP $INPUT_FILE

echo "Job Finished"
```
Note: Adjust module load, paths (`SIF_IMAGE`, `INPUT_FILE`, `cd`), and `mpirun` flags according to your specific HPC environment.

---

## Example `inputs` File

*(This section moved here for better flow after installation)*

The `Diffusion` application reads parameters from a text file (commonly named `inputs`). Create one based on this example (also found in `apps/diffusion/inputs` in the source tree):

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
direction = "X"                         # Compute for X, Y, Z ("X", "Y", "Z", "All", or "X Z" etc.)

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
