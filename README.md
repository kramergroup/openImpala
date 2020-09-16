# OpenImpala

### OPEN source IMage based PArallelisable Linear Algebra solver

A common challenge from image-based modelling is the size of 3D tomography datasets, which can be of the order of several billion voxels. OpenImpala a data-driven, fully parallelisable, image-based modelling framework. 

3D datasets are used as the computational domain within a finite-differences based model, in order to solve the diffusion equation directly on the image dataset, removing the need for additional meshing. 

OpenImpala then calculates the equivalent homogenised transport coefficients for the given microstructure. These coefficients are written into parameterised files for direct compatibility with two popular continuum battery models: PyBamm and DandeLiion, facilitating the link between different scales of computational battery modelling. 

OpenImpala has been shown to scale well with an increasing number of computational cores on distributed memory architectures, making it applicable to large datasets typical of modern tomography.

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
cd build/tests
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

The main programme of OpenImpala calculates concentration gradients across a steady state diffusive problem for a 2 phase segmented dataset, in the 3 cartesian directions.

In order to use it, first ensure your dataset is stored within the /data repository, or use the correct filepath to navigate to your dataset, then, navigate to the apps directory:

```bash
cd build/apps
./Diffusion
```

OpenImpala will now ask the filename for the calculation to be run on, as an example, try:

```bash
SampleData_2Phase.tif
```

It  will now calculate steady state diffusion in each direction and print the results, as well as volume fraction.

In order to run the same calculation but using more processors, try:

```bash
mpirun -np 2 ./Diffusion
```

Compare the 'Total Run time' values between the two calculations, to check MPI is running correctly.


## Visualisation

OpenImpala is built on the AMReX software framework. Output plot files can be visualised using a number of open-source visualisation packages e.g. Paraview, Visit, yt or AMRVis. 

For further information on how to view the visualisation data visit: https://amrex-codes.github.io/amrex/docs_html/Visualization.html

## Batch

To submit a non-interactive job for use with the HPC batch queuing system, use DiffusionBatch.cpp

Edit the file and change the DATA_PATH property, declared in the header, to the datafile to be studied, e.g.:

```bash
#define DATA_PATH "../../data/SampleData_2Phase.tif"
```

Goes to:

```bash
#define DATA_PATH "../../data/SampleData_3Phase.tif"
```

Recompile the OpenImpala executables, and check the new executable functions correctly:

```bash
make
cd build/apps
./DiffusionBatch
```

The code should run with no required input from the user.

You can now use OpenImpala in an HPC batch job, an example file: 

```bash
module load singularity/3.2.1

export OMP_NUM_THREADS=1

cd /openimpala/build/apps/

mpirun -np 20 singularity exec openimpala-singularity_latest.sif ./DiffusionBatch
```

N.B. the OpenImpala singularity image needs to located in the same directory as the executable.



