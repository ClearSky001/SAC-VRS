"""
Single-processor version of PyTorch MPI utilities for Windows compatibility.
This version removes MPI dependencies and provides mock functions.
"""
import torch

def setup_pytorch_for_mpi():
    """Single processor version - does nothing"""
    pass

def sync_params(module):
    """Single processor version - does nothing"""
    pass

def mpi_avg_grads(module):
    """Single processor version - does nothing"""
    pass