#!/usr/bin/env python

import time, sys, os, io
# Suppress output for all ranks but one
if (os.environ["MPI_LOCALRANKID"] != 0):
    suppress_text = io.StringIO()
    sys.stdout = suppress_text

import unittest

from pyoimp import initialize, finalize, isInitialized, read_tiff_stack;

class TestThreshold(unittest.TestCase):

    def setUp(self) -> None:
        initialize()

    def test_initialized(self) -> None:

        voxels = read_tiff_stack("../data/SampleData_2Phase.tif",32)
        print("Number of boxes {}".format(voxels.size()))

    def tearDown(self) -> None:
        finalize()
        sys.stdout.flush()
        return super().tearDown()

if __name__ == '__main__':

    unittest.main()