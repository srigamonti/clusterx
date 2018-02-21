import numpy as np


def isclose(r1,r2,rtol=1e-4):
    return np.linalg.norm(np.subtract(r1,r2)) < rtol
