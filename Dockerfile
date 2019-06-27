FROM fedora as amrex-build

RUN yum update -y
RUN yum install -y gcc git make python gcc-c++ gcc-gfortran hostname
RUN yum install -y openmpi-devel
RUN mkdir -p /src

ENV PATH="/usr/lib64/openmpi/bin:${PATH}"

## Obtain source files
WORKDIR /src
RUN git clone https://github.com/AMReX-Codes/amrex.git && \
    cd /src/amrex && \
    ./configure --with-mpi yes --with-omp yes --enable-eb yes && \
    make && \
    make install

RUN curl -O https://computation.llnl.gov/projects/hypre-scalable-linear-solvers-multigrid-methods/download/hypre-2.11.2.tar.gz && \
    tar xvzf hypre-2.11.2.tar.gz && \
    cd hypre-2.11.2/src && \
    ./configure && \
    make && \
    make install

# -----------------------------------------------------------------------------------

FROM fedora

## Use --build-arg option to add SSH_PRIVATE_KEY in docker
## e.g. docker build -t amrex --build-arg SSH_PRIVATE_KEY="$(cat ~/.ssh/id_rsa)" .
ARG SSH_PRIVATE_KEY

RUN dnf install -y gcc git libtiff libtiff-devel make python gcc-c++ gcc-gfortran hostname openmpi-devel
RUN mkdir -p /src

COPY --from=amrex-build /src/amrex/tmp_install_dir/include /usr/include/amrex
COPY --from=amrex-build /src/amrex/tmp_install_dir/lib/* /usr/lib64
COPY --from=amrex-build /src/hypre-2.11.2/src/hypre/include /usr/include/hypre
COPY --from=amrex-build /src/hypre-2.11.2/src/hypre/lib/* /usr/lib64

ENV PATH="/usr/lib64/openmpi/bin:${PATH}"

RUN mkdir /root/.ssh/ && \
    touch /root/.ssh/id_rsa && \
    echo "${SSH_PRIVATE_KEY}" > /root/.ssh/id_rsa && \
    chmod 400 /root/.ssh/id_rsa && \
    touch /root/.ssh/known_hosts && \
    ssh-keyscan -t rsa gitlab.com > ~/.ssh/known_hosts && \
    rm -rf /root/.ssh

WORKDIR /src