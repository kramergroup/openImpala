# OpenImpala

OPEN source IMage based PArallelisable Linear Algebra solver

## Dockerized compile environment

The `docker` branch holds a container providing the reference enviroment, wich is based on:

- Fedora
- OpenMPI
- Hypre

There is an open [issue](https://github.com/open-mpi/ompi/issues/4948) with Open-MPI when using Docker container. This requires some additional rights. Use the following to start a functioning environment:

```bash
docker run --rm --cap-add=SYS_PTRACE -it -v "$(pwd):/src" amrex /bin/bash
```

To run with multiple MPI instances use somethings similar to 
```bash
mpirun -n 2 --allow-run-as-root <PROGRAM>
```
