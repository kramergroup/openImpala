#!/usr/bin/env python

from pyoimp import initialize, finalize,isInitialized;
from time import sleep;

amrex = initialize()
print(amrex)
print(isInitialized())
finalize(amrex)
sleep(3)