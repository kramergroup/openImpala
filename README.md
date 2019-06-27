# OpenImpala - Docker builds

This branch holds dockerized amrex build environments for the master branch.

## Amrex Build Environment 

The default container (`Dockerfile`) provides a build environment for Amrex. It pulls
the latests Amrex source from their [git repository](https://github.com/AMReX-Codes/amrex), builds the Amrex library with embedded boundary support, but without MPI support, and constructs a minimal container with the compiled libraries. Only the header files are available in the runtime environment.

| Folder             | Description         |
| ------------------ | ------------------- |
| /usr/include/amrex | Header files        |
| /usr/lib64         | Amrex library files |

The runtime provides the same `gcc` and `gfortran` compilers that are used to compile the shared library. 

To compile and link statically use a command similar to

```bash
g++ -I/usr/include/amrex main.cpp /usr/lib64/libamrex.a -lgfortran -lm -ltiff -o main
```

> The order is important. All libraries need to come after the source (and if relevant object files).

## Amrvis Visualisation Tool

A stand-alone container to use Amrvis can be build using

```bash
docker build -t kramergroup/amrvis  -f Dockerfile.amrvis .
```

It is convinient to define an alias for this container using

```bash
alias amrvis='xhost + 127.0.0.1 > /dev/null && docker run -it -v $(pwd):/data -e DISPLAY=host.docker.internal:0 kramergroup/amrvis'
```

`Amvis` is an [X11](https://www.x.org/wiki/) application and needs access to an x-server to render output. There are different ways to provide access to the hosts xserver to the container. The above alias assumes that the X server is accepting network connections. This likely needs configuration, which depends on the host system.
