"""
Single-processor version of MPI tools for Windows compatibility.
This version removes MPI dependencies and provides mock functions.
"""
import os, subprocess, sys
import numpy as np

def proc_id():
    """Single processor version - always returns 0"""
    return 0

def num_procs():
    """Single processor version - always returns 1"""
    return 1

def mpi_fork(n, bind_to_core=False):
    """Single processor version - does nothing"""
    pass

def msg(m, string=''):
    """Single processor version - just prints the message"""
    print(string)

def mpi_avg(x):
    """Single processor version - returns x as is"""
    return x

def mpi_statistics_scalar(x, with_min_and_max=False):
    """Single processor version - returns basic statistics"""
    x = np.asarray(x, dtype=np.float32)
    if with_min_and_max:
        return x, 0.0, x, x
    return x, 0.0