# Copyright (c) 2015-2019, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

from ase import Atoms
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

from clusterx.calculators.emt import ToyEMText
from clusterx.parent_lattice import ParentLattice
from clusterx.super_cell import SuperCell
from clusterx.structure import Structure
from clusterx.structures_set import StructuresSet
from clusterx.clusters.clusters_pool import ClustersPool
from clusterx.correlations import CorrelationsCalculator


def test_cluster_expansion():
    """Regression test for cluster expansion.

    After successful execution of the test, the generated structures and
    clusters pool may be visualized with the command::

        ase gui test_cluster_expansion_[...].json

    """

    cell = [[3, 0, 0], [0, 1, 0], [0, 0, 5]]
    positions = [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
    pbc = [True, True, False]

    pri = Atoms(["H", "H", "H"], positions=positions, cell=cell, pbc=pbc)
    su1 = Atoms(["C", "H", "H"], positions=positions, cell=cell, pbc=pbc)
    su2 = Atoms(["H", "He", "H"], positions=positions, cell=cell, pbc=pbc)
    su3 = Atoms(["H", "N", "H"], positions=positions, cell=cell, pbc=pbc)

    plat = ParentLattice(pri, substitutions=[su1, su2, su3], pbc=pbc)
    cpool = ClustersPool(plat, npoints=[1, 2, 3, 4], radii=[0, 2.3, 1.42, 1.42])
    cpool.write_clusters_db(cpool.get_cpool(), cpool.get_cpool_scell(), "cpool.json")
    corrcal = CorrelationsCalculator("trigonometric", plat, cpool)

    scell = SuperCell(plat, np.array([(1, 0, 0), (0, 3, 0), (0, 0, 1)]))
    strset = StructuresSet(plat)
    occupations = [
        [1, 2, 1, 6, 7, 1, 1, 2, 1],
        [6, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 2, 1, 1, 7, 1, 6, 7, 1],
        [1, 1, 1, 6, 1, 1, 1, 7, 1],
        [6, 7, 1, 6, 7, 1, 6, 2, 1],
        [1, 7, 1, 6, 2, 1, 1, 1, 1],
        [6, 1, 1, 1, 1, 1, 1, 2, 1],
        [1, 1, 1, 1, 7, 1, 6, 7, 1],
        [1, 2, 1, 6, 2, 1, 6, 2, 1],
        [6, 1, 1, 6, 2, 1, 1, 1, 1],
        [1, 1, 1, 6, 7, 1, 1, 2, 1],
        [1, 2, 1, 6, 2, 1, 1, 1, 1],
        [1, 7, 1, 1, 1, 1, 6, 2, 1],
        [6, 1, 1, 1, 7, 1, 1, 7, 1],
        [1, 2, 1, 6, 7, 1, 6, 2, 1],
        [1, 7, 1, 1, 2, 1, 1, 1, 1],
        [6, 7, 1, 1, 7, 1, 1, 1, 1],
        [6, 2, 1, 6, 1, 1, 6, 7, 1],
        [1, 7, 1, 1, 1, 1, 6, 2, 1],
        [6, 1, 1, 1, 7, 1, 1, 1, 1],
    ]
    for occupations_i in occupations:
        strset.add_structure(Structure(scell, occupations_i))

    strset.serialize(
        filepath="test_cluster_expansion_structures_set.json", overwrite=True
    )
    comat = corrcal.get_correlation_matrix(strset)
    strset.set_calculator(ToyEMText())
    energies = strset.compute_property_values()
    clsets = cpool.get_clusters_sets(grouping_strategy="size")
    fitter_cv = linear_model.LinearRegression(fit_intercept=True)

    cv = []
    rmse = []
    rows = np.arange(len(energies))
    ecis = []
    ranks = []
    sizes = []
    for iset, clset in enumerate(clsets):
        _comat = comat[np.ix_(rows, clset)]
        _cvs = cross_val_score(
            fitter_cv,
            _comat,
            energies,
            cv=LeaveOneOut(),
            scoring="neg_mean_squared_error",
        )
        cv.append(np.sqrt(-np.mean(_cvs)))
        fitter_cv.fit(_comat, energies)
        rmse.append(np.sqrt(mean_squared_error(fitter_cv.predict(_comat), energies)))
        ecis.append(fitter_cv.coef_)
        ranks.append(np.linalg.matrix_rank(_comat))
        sizes.append(len(clset))

    cv_ref = np.array([
        3.011734, 0.970003, 1.346359, 1.346359, 1.346359, 2.387488,
        2.674287, 3.207947, 3.054692, 3.367751, 3.946197
    ])
    rmse_ref = np.array([
        2.157325e+00, 2.983903e-01, 2.878580e-01, 2.878580e-01,
        2.878580e-01, 9.269026e-14, 9.946012e-15, 1.700114e-14,
        1.101484e-14, 9.315294e-15, 4.286140e-14
    ])
    ranks_ref = np.array([3, 9, 11, 11, 11, 16, 16, 16, 16, 16, 16])
    sizes_ref = np.array([3, 9, 11, 17, 21, 17, 23, 27, 20, 26, 30])

    np.testing.assert_allclose(cv, cv_ref, rtol=1e-6)
    np.testing.assert_allclose(rmse, rmse_ref, rtol=1e-6)
    np.testing.assert_array_equal(ranks_ref, ranks)
    np.testing.assert_array_equal(sizes_ref, sizes)
