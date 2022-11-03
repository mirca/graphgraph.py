import numpy as np

from .operators import *

def degree_op_mat(p):
    eye = np.eye(p)
    return np.array([degree_op_T(eye[:,i]) for i in range(p)])
