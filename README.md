# OpenImpala

OPEN source IMage based PArallelisable Linear Algebra solver

## Getting Started

The source code is available at https://github.com/kramergroup/openImpala.git

### Containerised compile environment

Singularity hub holds a containerised build environment suitable for running OpenImpala, which uses:

- Ubuntu
- OpenMPI
- Hypre
- LibTiff
- AMReX

In order to pull the singularity image from the hub, first ensure you have Singularity installed on your system, then run the following command:

```bash
singularity pull shub://jameslehoux/openimpala-singularity
```
This will pull the OpenImpala Singularity image to your current directory.

To operate the container interactively use the shell command, like so:

```bash
singularity shell openimpala-singularity_latest.sif
```
You are now ready to compile and use OpenImpala

### Building OpenImpala

It is recommended to build OpenImpala using the provided Singularity image. With the image opened interactively navigate to the main OpenImpala directory, here run the following command:

```bash
make
```

This will create a new folder, /build , which will be populated with the .o files and executables of OpenImpala. 

WARNING! There is a known bug when using the command ```make clean``` where it deletes files beyond the OpenImpala directory. This functionality currently does not work and should not be used.

### Testing Functionality

Once the make command is finished navigate to the test directory to check the functionality of the created executables:

```bash
cd /build/tests
./tTiffReader
```

which will open a sample Tiff file and assert the dimensions are as expected, printing the output.

```bash
./tVolumeFraction
```

which will open a sample 2 phase segmented tiff file and calculate the volume fraction of each phase.

```bash
./tTortuosity
```

which will open a sample 2 phase segmented tiff file and calculate the effective diffusion and tortuosity in the x direction.

## Diffusion

The main programme of 


## Visualisation

OpenImpala is built on the AMReX software framework. Output plot files can be visualised using a number of open-source visualisation packages e.g. Paraview, Visit, yt or AMRVis. 

For further information on how to view the visualisation data visit: https://amrex-codes.github.io/amrex/docs_html/Visualization.html

