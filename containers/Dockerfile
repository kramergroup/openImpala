# Stage 1: Build dependencies and OpenImpala
# Use Rocky Linux 8 as the base
ARG ROCKY_VERSION=8
FROM rockylinux:${ROCKY_VERSION} AS builder

# Define arguments for versions (can be overridden during build)
ARG CMAKE_VERSION=3.28.3
ARG OPENMPI_VERSION=4.1.6
ARG HDF5_VERSION=1.12.3
ARG AMREX_VERSION=23.11
ARG HYPRE_VERSION=v2.30.0
ARG OPENIMPALA_REPO=https://github.com/kramergroup/openImpala.git # <-- VERIFY URL
ARG OPENIMPALA_BRANCH=working # <-- Default branch set to 'working'

# Set frontend to noninteractive (less relevant for dnf, but good practice)
ENV DEBIAN_FRONTEND=noninteractive

# Install build tools, compilers, and essential dependencies
RUN dnf update -y && \
    dnf install -y dnf-utils && \
    dnf config-manager --set-enabled powertools && \
    dnf install -y epel-release && \
    dnf update -y && \
    dnf install -y \
        gcc gcc-gfortran gcc-c++ make \
        wget git patch \
        python3 python3-pip \
        hostname \
        infiniband-diags libibverbs-devel \
        libtiff-devel boost-devel \
        which ca-certificates && \
    dnf clean all

# --- Install CMake (Recent Version) ---
WORKDIR /tmp/build_src
ARG CMAKE_VERSION
ENV CMAKE_INSTALL_PREFIX=/opt/cmake/${CMAKE_VERSION}
RUN wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz && \
    mkdir -p ${CMAKE_INSTALL_PREFIX} && \
    tar -xzf cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz --strip-components=1 -C ${CMAKE_INSTALL_PREFIX}
# Add CMake to PATH for this stage
ENV PATH=${CMAKE_INSTALL_PREFIX}/bin:${PATH}
RUN cmake --version

# --- Install OpenMPI (Recent Version) ---
ARG OPENMPI_VERSION
ENV OPENMPI_INSTALL_PREFIX=/usr/local
RUN wget https://download.open-mpi.org/release/open-mpi/v${OPENMPI_VERSION%.*}/openmpi-${OPENMPI_VERSION}.tar.gz --no-check-certificate && \
    tar -xzf openmpi-${OPENMPI_VERSION}.tar.gz && \
    cd openmpi-${OPENMPI_VERSION} && \
    ./configure \
       --prefix=${OPENMPI_INSTALL_PREFIX} \
       --enable-orterun-prefix-by-default \
       --enable-mpirun-prefix-by-default \
       --with-verbs \
       --enable-shared \
       --enable-static=no && \
    make -j$(nproc) install && \
    cd .. && \
    rm -rf openmpi-${OPENMPI_VERSION}* && \
    ldconfig # Update library cache

# --- Install HDF5 (with Parallel, Fortran, C++) ---
ARG HDF5_VERSION
ENV HDF5_INSTALL_PREFIX=/opt/hdf5/${HDF5_VERSION}
RUN wget https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-${HDF5_VERSION%.*}/hdf5-${HDF5_VERSION}/src/hdf5-${HDF5_VERSION}.tar.gz --no-check-certificate && \
    tar -xzf hdf5-${HDF5_VERSION}.tar.gz && \
    cd hdf5-${HDF5_VERSION} && \
    export CFLAGS="-O3 -march=native" && export CXXFLAGS="${CFLAGS}" && export FCFLAGS="${CFLAGS}" && \
    CC=mpicc CXX=mpicxx FC=mpif90 ./configure \
        --prefix=${HDF5_INSTALL_PREFIX} \
        --enable-parallel \
        --enable-fortran \
        --enable-fortran2003 \
        --enable-cxx \
        --enable-shared \
        --disable-static && \
    make -j$(nproc) install && \
    cd .. && \
    rm -rf hdf5-${HDF5_VERSION}*

# --- Install AMReX (Recent Stable Tag using CMake) ---
ARG AMREX_VERSION
ENV AMREX_INSTALL_PREFIX=/opt/amrex/${AMREX_VERSION}
RUN git clone --depth 1 --branch ${AMREX_VERSION} https://github.com/AMReX-Codes/amrex.git && \
    cd amrex && \
    mkdir build && cd build && \
    cmake .. \
        -DCMAKE_INSTALL_PREFIX=${AMREX_INSTALL_PREFIX} \
        -DCMAKE_BUILD_TYPE=Release \
        -DAMReX_MPI=ON \
        -DAMReX_OMP=ON \
        -DAMReX_EB=ON \
        -DCMAKE_CXX_COMPILER=mpicxx \
        -DCMAKE_C_COMPILER=mpicc \
        -DCMAKE_Fortran_COMPILER=mpif90 \
        -DCMAKE_CXX_FLAGS="-O3 -march=native" \
        -DCMAKE_C_FLAGS="-O3 -march=native" \
        -DCMAKE_Fortran_FLAGS="-O3 -march=native" && \
    make -j$(nproc) install && \
    cd ../.. && \
    rm -rf amrex

# --- Install HYPRE (Recent Stable Tag) ---
ARG HYPRE_VERSION
ENV HYPRE_INSTALL_PREFIX=/opt/hypre/${HYPRE_VERSION}
RUN git clone --depth 1 --branch ${HYPRE_VERSION} https://github.com/hypre-space/hypre.git && \
    cd hypre/src && \
    CC=mpicc CXX=mpicxx ./configure \
        --prefix=${HYPRE_INSTALL_PREFIX} \
        --with-MPI \
        --enable-shared \
        CFLAGS="-O3 -march=native" \
        CXXFLAGS="-O3 -march=native" && \
    make -j$(nproc) install && \
    cd ../.. && \
    rm -rf hypre

# --- Build OpenImpala (from specified branch) ---
WORKDIR /opt
ARG OPENIMPALA_REPO
ARG OPENIMPALA_BRANCH # Will use 'working' by default (set above), or override via --build-arg
# Set env vars needed for build
ENV HDF5_HOME=/opt/hdf5/${HDF5_VERSION}
ENV AMREX_HOME=/opt/amrex/${AMREX_VERSION}
ENV HYPRE_HOME=/opt/hypre/${HYPRE_VERSION}
ENV LD_LIBRARY_PATH=${HDF5_HOME}/lib:${AMREX_HOME}/lib:${HYPRE_HOME}/lib:/usr/local/lib:${LD_LIBRARY_PATH}
RUN echo "Cloning OpenImpala branch: ${OPENIMPALA_BRANCH} from ${OPENIMPALA_REPO}" && \
    git clone --depth 1 --branch ${OPENIMPALA_BRANCH} ${OPENIMPALA_REPO} openImpala && \
    cd openImpala && \
    echo "Building OpenImpala..." && \
    # Assuming 'make' is still the primary build method.
    # If using CMake: mkdir build && cd build && cmake .. && make -j$(nproc) install
    make -j$(nproc) && \
    # Clean up downloaded sources for dependencies
    rm -rf /tmp/build_src

#-------------------------------------------------------------------------------
# Stage 2: Final runtime image
#-------------------------------------------------------------------------------
FROM rockylinux:${ROCKY_VERSION} AS final

# Install only runtime dependencies
RUN dnf update -y && \
    dnf install -y \
        libibverbs \
        libtiff \
        boost-libs \
        python3 && \
    dnf clean all

# Copy built dependencies and OpenImpala from the builder stage
COPY --from=builder /usr/local /usr/local
COPY --from=builder /opt/hdf5 /opt/hdf5
COPY --from=builder /opt/amrex /opt/amrex
COPY --from=builder /opt/hypre /opt/hypre
COPY --from=builder /opt/openImpala /opt/openImpala

# Define arguments and environment variables for runtime
ARG HDF5_VERSION
ARG AMREX_VERSION
ARG HYPRE_VERSION
ARG OPENIMPALA_DIR=/opt/openImpala

ENV HDF5_HOME=/opt/hdf5/${HDF5_VERSION}
ENV AMREX_HOME=/opt/amrex/${AMREX_VERSION}
ENV HYPRE_HOME=/opt/hypre/${HYPRE_VERSION}
ENV OPENIMPALA_DIR=${OPENIMPALA_DIR}

# Add binaries to PATH (OpenMPI, HDF5 tools, OpenImpala apps/tests)
ENV PATH=${HDF5_HOME}/bin:/usr/local/bin:${OPENIMPALA_DIR}/build/apps:${OPENIMPALA_DIR}/build/tests:${PATH}

# Add libraries to LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=${HDF5_HOME}/lib:${AMREX_HOME}/lib:${HYPRE_HOME}/lib:/usr/local/lib:${LD_LIBRARY_PATH}

# Set CMAKE_PREFIX_PATH (may be useful if other tools interact with this container)
ENV CMAKE_PREFIX_PATH=${HDF5_HOME}:${AMREX_HOME}:${HYPRE_HOME}:/usr/local:${CMAKE_PREFIX_PATH}

# MPI runtime tuning (optional, same as Singularity recipe)
ENV OMPI_MCA_btl_vader_single_copy_mechanism=none
ENV OMPI_MCA_rmaps_base_oversubscribe=1

# Set locale
ENV LC_ALL=C
ENV LANG=C

# Update linker cache
RUN ldconfig

# Set default working directory
WORKDIR /data

# Default command (provides a shell)
CMD ["bash"]
