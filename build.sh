#/bin/env bash

## Pass private ssh key to access private repo
docker build --build-arg SSH_PRIVATE_KEY="$(cat ~/.ssh/id_rsa)" -t amrex .