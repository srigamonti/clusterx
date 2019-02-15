# Copyright (c) 2015-2019, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

from clusterx import utils
import numpy as np

def test_get_all_hnf():

    nrs = []
    for n in range(1,7):
        hnfs = utils.get_all_HNF(n)
        nrs.append(len(hnfs))

    assert (np.array(nrs) == np.array([1,7,13,35,31,91])).all()

    hnfs = utils.get_all_HNF(4,pbc=(1,1,0))

    hnfs_ref = [
                [[1, 0, 0],
                 [0, 4, 0],
                 [0, 0, 1]],
                [[1, 1, 0],
                 [0, 4, 0],
                 [0, 0, 1]],
                [[1, 2, 0],
                 [0, 4, 0],
                 [0, 0, 1]],
                [[1, 3, 0],
                 [0, 4, 0],
                 [0, 0, 1]],
                [[2, 0, 0],
                 [0, 2, 0],
                 [0, 0, 1]],
                [[2, 1, 0],
                 [0, 2, 0],
                 [0, 0, 1]],
                [[4, 0, 0],
                 [0, 1, 0],
                 [0, 0, 1]]
                ]

    isok = True

    # Now check that hnfs_ref is identical to hnfs
    for m1,m2 in zip(hnfs,hnfs_ref):
        for r1,r2 in zip(m1,m2):
            for i1,i2 in zip(r1,r2):
                if i1 != i2:
                    isok = False
                    break
            if not isok:
                break
        if not isok:
            break

    assert isok
