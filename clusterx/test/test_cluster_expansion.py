# Copyright (c) 2015-2019, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

import clusterx as c
from clusterx.parent_lattice import ParentLattice
from clusterx.super_cell import SuperCell
from clusterx.structure import Structure
from clusterx.structures_set import StructuresSet
from clusterx.clusters.clusters_pool import ClustersPool
from clusterx.correlations import CorrelationsCalculator
#from clusterx.model import ModelConstructor
from ase import Atoms
import numpy as np
from clusterx.calculators.emt import EMT2
from clusterx.utils import isclose


def test_cluster_expansion():
    """Test generation of clusters pools.

    After successful execution of the test, the generated structures and
    clusters pool may be visualized with the command::

        ase gui test_cluster_expansion_[...].json

    """

    cell = [[3,0,0],
            [0,1,0],
            [0,0,5]]
    positions = [
        [0,0,0],
        [1,0,0],
        [2,0,0]]
    pbc = [True,True,False]

    pri = Atoms(['H','H','H'], positions=positions, cell=cell, pbc=pbc)
    su1 = Atoms(['C','H','H'], positions=positions, cell=cell, pbc=pbc)
    su2 = Atoms(['H','He','H'], positions=positions, cell=cell, pbc=pbc)
    su3 = Atoms(['H','N','H'], positions=positions, cell=cell, pbc=pbc)

    plat = ParentLattice(pri,substitutions=[su1,su2,su3],pbc=pbc)
    #cpool = ClustersPool(plat, npoints=[0,1,2,3,4], radii=[0,0,2.3,1.42,1.42])
    cpool = ClustersPool(plat, npoints=[1,2,3,4], radii=[0,2.3,1.42,1.42])
    cpool.write_clusters_db(cpool.get_cpool(),cpool.get_cpool_scell(),"cpool.json")
    corrcal = CorrelationsCalculator("trigonometric", plat, cpool)

    scell = SuperCell(plat,np.array([(1,0,0),(0,3,0),(0,0,1)]))
    #strset = StructuresSet(plat, filename="test_cluster_expansion_structures_set.json")
    strset = StructuresSet(plat)
    nstr = 20
    #for i in range(nstr):
    #    strset.add_structure(scell.gen_random(nsubs={}))
    strset.add_structure(Structure(scell,[1,2,1,6,7,1,1,2,1]),write_to_db=True)
    strset.add_structure(Structure(scell,[6,1,1,1,1,1,1,1,1]),write_to_db=True)
    strset.add_structure(Structure(scell,[1,2,1,1,7,1,6,7,1]),write_to_db=True)
    strset.add_structure(Structure(scell,[1,1,1,6,1,1,1,7,1]),write_to_db=True)
    strset.add_structure(Structure(scell,[6,7,1,6,7,1,6,2,1]),write_to_db=True)
    strset.add_structure(Structure(scell,[1,7,1,6,2,1,1,1,1]),write_to_db=True)
    strset.add_structure(Structure(scell,[6,1,1,1,1,1,1,2,1]),write_to_db=True)
    strset.add_structure(Structure(scell,[1,1,1,1,7,1,6,7,1]),write_to_db=True)
    strset.add_structure(Structure(scell,[1,2,1,6,2,1,6,2,1]),write_to_db=True)
    strset.add_structure(Structure(scell,[6,1,1,6,2,1,1,1,1]),write_to_db=True)
    strset.add_structure(Structure(scell,[1,1,1,6,7,1,1,2,1]),write_to_db=True)
    strset.add_structure(Structure(scell,[1,2,1,6,2,1,1,1,1]),write_to_db=True)
    strset.add_structure(Structure(scell,[1,7,1,1,1,1,6,2,1]),write_to_db=True)
    strset.add_structure(Structure(scell,[6,1,1,1,7,1,1,7,1]),write_to_db=True)
    strset.add_structure(Structure(scell,[1,2,1,6,7,1,6,2,1]),write_to_db=True)
    strset.add_structure(Structure(scell,[1,7,1,1,2,1,1,1,1]),write_to_db=True)
    strset.add_structure(Structure(scell,[6,7,1,1,7,1,1,1,1]),write_to_db=True)
    strset.add_structure(Structure(scell,[6,2,1,6,1,1,6,7,1]),write_to_db=True)
    strset.add_structure(Structure(scell,[1,7,1,1,1,1,6,2,1]),write_to_db=True)
    strset.add_structure(Structure(scell,[6,1,1,1,7,1,1,1,1]),write_to_db=True)

    strset.serialize(path="test_cluster_expansion_structures_set.json", overwrite=True)
    # Get the DATA(comat) + TARGET(energies)
    comat = corrcal.get_correlation_matrix(strset)
    strset.set_calculator(EMT2())
    energies = strset.calculate_property()

    #fitter_model = Fitter(method = "skl_LinearRegression")

    clsets = cpool.get_clusters_sets(grouping_strategy = "size")

    from sklearn.model_selection import LeaveOneOut
    from sklearn.model_selection import cross_val_score
    from sklearn import linear_model
    from sklearn.metrics import make_scorer, r2_score, mean_squared_error

    fitter_cv = linear_model.LinearRegression(fit_intercept=True, normalize=False)
    #fitter_cv = linear_model.LinearRegression(fit_intercept=False, normalize=False)

    cv = []
    rmse = []
    rows = np.arange(len(energies))
    ecis = []
    ranks = []
    sizes = []
    for iset, clset in enumerate(clsets):
        _comat = comat[np.ix_(rows,clset)]
        _cvs = cross_val_score(fitter_cv, _comat, energies, cv=LeaveOneOut(), scoring = 'neg_mean_squared_error')
        cv.append(np.sqrt(-np.mean(_cvs)))
        fitter_cv.fit(_comat,energies)
        rmse.append(np.sqrt(mean_squared_error(fitter_cv.predict(_comat),energies)))
        ecis.append(fitter_cv.coef_)
        ranks.append(np.linalg.matrix_rank(_comat))
        sizes.append(len(clset))

    #print(repr(cv))
    #print(repr(rmse))
    #print(repr(ranks))
    #print(repr(sizes))

    rcv = np.array([21.25070484696078, 21.192827866028548, 18.253021305083276, 18.25302130508316, 18.253021305083077, 26.826078769571485, 29.985365890016077, 33.07860588175105, 46.94157574864739, 49.37973761687287, 52.263685892043306])
    rrmse = np.array([16.901897707237335, 6.70078970932807, 3.999376257086459, 3.9993762570864626, 3.999376257086468, 4.388974798433372e-13, 1.6733997988876832e-13, 4.325945001471906e-13, 1.8694837133527457e-13, 2.5232271964659314e-13, 4.896588122807044e-13])
    rranks = np.array([3, 9, 11, 11, 11, 16, 16, 16, 16, 16, 16])
    rsizes = np.array([3, 9, 11, 17, 21, 17, 23, 27, 20, 26, 30])

    isok = isclose(cv,rcv) and isclose(rrmse, rmse) and (rranks == ranks).all() and (rsizes == sizes).all()
    assert(isok)
    #print(energies)
    """
        for train_index, test_index in loo.split(_comat):
            _comat_train = _comat[train_index]
            _energy_train = energies[train_index]

            #fitter_cv.fit(_comat_train,_energy_train)
    """

    # Select the clusters by cross-validation

    #clsets = cpool.get_sets(strategy = "size")


    """

    mctr = ModelConstructor(
        prop = "energy",
        training_set = strset,
        fitter = flr,
        clusters_selector,
        correlations_calculator = corrcal
    )
    """

    """
    reg = linear_model.LinearRegression()
    reg.fit(comat, energies)
    #LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
    print("Js",reg.coef_)

    reg = linear_model.Ridge(alpha = .001)
    reg.fit(comat, energies)
    print("Js l2: ",reg.coef_)
    print("intercept: ",reg.intercept_ )


    reg = linear_model.RidgeCV(alphas=[0.001, 0.005, 0.1])
    reg.fit(comat, energies)
    print("Js l2: ",reg.coef_)
    print("intercept: ",reg.intercept_ )
    print("alpha:  ", reg.alpha_)

    # Generate output
    print ("\n\n========Test writes========")
    atom_idxs, atom_nrs = cpool.get_cpool_arrays()
    scell = cpool.get_cpool_scell()
    cpool.write_clusters_db(cpool.get_cpool(),scell,"test_cluster_expansion_cpool.json")
    print("Correlation matrix:\n")
    print(np.array2string(comat,separator=",",max_line_width=1000))
    print ("===========================\n")

    print ("========Asserts========")

    #assert np.allclose([-0.33333333,0.,-0.,0.33333333,0.57735027,-0.33333333,-0.25,-0.,-0.25],comat,atol=1e-5)
    #print(comat)
    """
