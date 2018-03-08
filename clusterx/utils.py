import numpy as np


def isclose(r1,r2,rtol=1e-4):
    return np.linalg.norm(np.subtract(r1,r2)) < rtol


def dict_compare(d1, d2):
    """
    parts taken from
    https://stackoverflow.com/questions/4527942/comparing-two-dictionaries-in-python
    """
    areeq = True

    if len(d1) != len(d2):
        return False

    d1_keys = set(d1.keys())
    d2_keys = set(d2.keys())
    intersect_keys = d1_keys.intersection(d2_keys)
    if len(d1) != len(intersect_keys):
        return False

    for k in d1_keys:
        for v1,v2 in zip(d1[k],d2[k]):
            if v1 != v2:
                return False
    
    return areeq
