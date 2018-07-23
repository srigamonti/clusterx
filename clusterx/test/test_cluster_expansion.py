import clusterx as c
from clusterx.parent_lattice import ParentLattice
from clusterx.super_cell import SuperCell
from clusterx.structure import Structure
from clusterx.structures_set import StructuresSet
from clusterx.clusters.clusters_pool import ClustersPool
from clusterx.correlations import CorrelationsCalculator
#from clusterx.model import ModelConstructor
from clusterx.fitter import Fitter
from ase import Atoms
import numpy as np
from clusterx.calculators.emt import EMT2

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
    cpool = ClustersPool(plat, npoints=[1,2,3,4], radii=[0,2.3,1.42,1.42])
    cpool.write_clusters_db(cpool.get_cpool(),cpool.get_cpool_scell(),"cpool.json")
    corrcal = CorrelationsCalculator("trigonometric", plat, cpool)

    scell = SuperCell(plat,np.array([(1,0,0),(0,3,0),(0,0,1)]))
    strset = StructuresSet(plat, filename="test_cluster_expansion_structures_set.json")
    nstr = 20
    print(scell.get_idx_subs())
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

    cvs = []
    rows = np.arange(len(energies))
    for clset in clsets:
        _comat = comat[np.ix_(rows,clset)]
        #_cvs = cross_val_score(fitter_cv, _comat, energies, cv=LeaveOneOut(), scoring = make_scorer(mean_squared_error))
        _cvs = cross_val_score(fitter_cv, _comat, energies, cv=LeaveOneOut(), scoring = 'neg_mean_squared_error')
        print("clusters set: ",clset)
        print('_cvs',_cvs)
        print('m_cvs',np.sqrt(-np.mean(_cvs)))
        print("")

    fitter_cv.fit(comat,energies)
    print("target",energies)
    print("predictions",fitter_cv.predict(comat))
    print("score",np.sqrt(mean_squared_error(fitter_cv.predict(comat),energies)))
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
