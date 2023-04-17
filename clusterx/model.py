# Copyright (c) 2015-2019, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

from clusterx.correlations import CorrelationsCalculator
from clusterx.estimators.estimator_factory import EstimatorFactory
from clusterx.clusters_selector import ClustersSelector
import numpy as np
import pickle
import os

class Model():
    """Model class

    **Parameters:**

    ``corrc``: CorrelationsCalculator object
        The correlations calculator corresponding to the optimal model.
    ``property_name``: String
        The methods of the ``Model`` class that take a ``StructureSet`` object as
        argument, will query property values named according to ``property_name``.
    ``estimator``: Estimator object
        An estimator object to predict the property values. Alternatively,
        The effective cluster interactions (ECIs) can be set.
    ``ecis``: Array of float
        Effective cluster iteractions (multiplied by the corresponding
        multiplicities). Overrides ``estimator`` object.
    ``filepath``: string (default: None)
        Overrides all the above. Used to initialize from file. Path of a json or pickle
        file containing a serialized Model object, as generated by the ``Model.serialize()`` method.
    ``json_db_filepath``: string (default: None)
        DEPRECATED, use ``filepath`` instead.
        Overrides all the above. Used to initialize from file. Path of a json
        file containing a serialized Model object, as generated by the ``Model.serialize()`` method.

    """
    def __new__(cls, *args, **kwargs):

        if len(args)==0 and len(kwargs)==0:
            inst = super(Model,cls).__new__(cls, *args, **kwargs)
            return inst
        
        elif "filepath" in kwargs:
            filepath = kwargs["filepath"]
            
            fext = os.path.splitext(filepath)[1][1:]
            if fext == "pickle":
                with open(filepath,"rb") as f:
                    inst = pickle.load(f)
                    
                return inst
        else:
            inst = super(Model,cls).__new__(cls)
            inst.initialize(*args,**kwargs)
            return inst
    

    def __init__(self, corrc = None, property_name = None, estimator = None, ecis = None, filepath = None, json_db_filepath = None):
        pass
    
    def initialize(self, corrc = None, property_name = None, estimator = None, ecis = None, filepath = None, json_db_filepath = None):
        self.pickle_file = None
        self._filepath_corrc = None
        if filepath is not None:
            fext = os.path.splitext(filepath)[1][1:]
            
            if fext == "pickle":
                self.pickle_file = filepath
                
            if fext == "json":
                json_db_filepath = filepath
                

        if json_db_filepath is not None:
            from ase.db import connect
            db = connect(json_db_filepath)

            from clusterx.correlations import CorrelationsCalculator
            self.corrc = CorrelationsCalculator(db = db)

            modict = db.metadata.get('model_parameters',None)
            if modict is None:
                import sys 
                sys.exit('Error: Initialization from json_db did not succeed.')

            self.ecis = modict.get('ECIs',[])
            self.property = modict.get('property_name',None)

        else:
            self.corrc = corrc
            self.ecis = ecis
            self.property = property_name

        self.estimator = estimator


    def serialize(self, filepath = None, fmt = None, db_name = None):
        """Write cluster expansion model to Json database

        **Parameter:**

        ``filepath``: string 
            Name of the file to store the Model instance. It can contain the relative or absolute path.
            If not given, and if db_name (deprecated) is None, filepathh is set to "cemodel.pickle" and fmt is overriden (set to pickle).
        ``fmt``: string
            It can take the values ``"pickle"`` or ``"json"``. 
            So far, object can only be re-initiated from a ``"pickle"`` file, therefore this is the preferred format. 
            If not given, it is inferred from the filepathh.
        ``db_name``: (DEPRECATED) string 
            Name of the json file containing the database
        """
        from pathlib import Path
        import os

        if filepath is None and db_name is None:
            filepath = "cemodel.pickle"

        if db_name is not None:
            filepath = db_name

        fext = os.path.splitext(filepath)[1][1:]

        
        if fmt is None:
            if fext == "pickle":
                fmt = "pickle"
            if fext == "json":
                fmt = "json_db"

        if fmt == "pickle":
            self.pickle_file = filepath


            stem = Path(filepath).stem
            dirname = os.path.dirname(filepath)
            self._filepath_corrc = os.path.join(dirname,stem+"_CCALC.pickle")
            self.corrc.serialize(filepath=self._filepath_corrc)
            
            with open(filepath,"wb") as f:
                pickle.dump(self,f)

                
        if fmt == "json_db":
            from ase.db.jsondb import JSONDatabase
            from subprocess import call
            call(["rm","-f",db_name])
            atoms_db = JSONDatabase(filename=db_name)

            cpool_atoms  = self.corrc._cpool.get_cpool_atoms()

            for atoms in cpool_atoms:
                atoms_db.write(atoms)

            #atoms_db.write(Atoms(symbols=None))
            modeldict = self.corrc._cpool.get_cpool_dict()

            corr_dict = {}
            corr_dict.update({'basis':self.corrc.basis})
            if self.corrc._lookup_table is not None:
                corr_dict.update({'lookup':True})
            else:
                corr_dict.update({'lookup':False})
            modeldict.update({'correlations_calculator':corr_dict})

            modict = {}
            modict.update({'property_name':self.property})

            if self.ecis is None:
                eff_inter = np.asarray([el for el in self.estimator.coef_])
                if (eff_inter[0] < 1.e-15) and (self.estimator.intercept_ > 1.e-15):
                    eff_inter[0] = float(self.estimator.intercept_)
                modict.update({'ECIs':eff_inter})
            else:
                modict.update({'ECIs':self.ecis})
            modeldict.update({'model_parameters':modict})

            atoms_db.metadata = modeldict
        
    def get_plat(self):
        """ Return parent lattice of the cluster expansion model.

        """
        return self.corrc._plat

    def get_ecis(self):
        """ Return array of effective cluster interactions (ECIs) of the model
        """
        if self.ecis is not None:
            return self.ecis
        else:
            return self.estimator.coef_

    def get_correlations_calculator(self):
        """ Return correlations calculator of the Model object
        """
        return self.corrc

    def predict(self,structure):
        """Predict property with the optimal cluster expansion model.

        **Parameters:**

        ``structure``: Structure object
            structure object to calculate property to.

        """
        corrs = self.corrc.get_cluster_correlations(structure)
        if self.estimator is not None:
            return self.estimator.predict(corrs.reshape(1,-1))[0]

        else:
            pv = 0
            for i in range(len(corrs)):
                pv = pv + self.ecis[i]*corrs[i]

            return pv

    def predict_swap_binary_linear(self,structure, ind1 = None, ind2 = None, correlation = False):
        """Predict property difference with the optimal cluster expansion model only for binary-linear)

        **Parameters:**

        ``structure``: Structure object
            structure object to calculate property difference to.

        ``ind1``: int
            index of first atom position has been swapped

        ``ind2``: int
            index of second atom position has been swapped

        """
        #corrsx = self.corrc.get_cluster_correlations(structure)
        try:
            cluster_orbits = self.corrc._cluster_orbits_mc
        except AttributeError:
            raise AttributeError("Cluster_orbit set is not predefined, look at the documentation.")
        corrs = np.zeros(len(cluster_orbits))

        sigma_ind1=structure.sigmas[ind1]
        sigma_ind2=structure.sigmas[ind2]

        for icl, cluster_orbit in enumerate(cluster_orbits):
            for cluster in cluster_orbit:
                cluster_atomic_idxs = cluster.get_idxs()
                if ind1 in cluster_atomic_idxs:
                    #cluster_alphas = cluster.alphas
                    sigmas = [structure.sigmas[cl_idx] for cl_idx in cluster_atomic_idxs]
                    cf1 = np.prod(np.array(sigmas))
                    sigmas[cluster_atomic_idxs.index(ind1)] = sigma_ind2
                    if ind2 in cluster_atomic_idxs:
                        sigmas[cluster_atomic_idxs.index(ind2)] = sigma_ind1
                    cf0 = np.prod(np.array(sigmas))
                    corrs[icl] += cf1
                    corrs[icl] += (-1)*cf0

                elif ind2 in cluster_atomic_idxs:
                    #cluster_alphas = cluster.alphas
                    sigmas = [structure.sigmas[cl_idx] for cl_idx in cluster_atomic_idxs]
                    cf1 = np.prod(np.array(sigmas))
                    sigmas[cluster_atomic_idxs.index(ind2)] = sigma_ind1
                    cf0 = np.prod(np.array(sigmas))
                    corrs[icl] += cf1
                    corrs[icl] += (-1)*cf0


            corrs[icl] /= len(cluster_orbit)
        corrs = np.around(corrs,decimals=12)
        if correlation:
            return corrs

        if self.estimator is not None:
            return self.estimator.predict(corrs.reshape(1,-1))[0]
        else:
            pv = 0
            for i in range(len(corrs)):
                pv = pv + self.ecis[i]*corrs[i]
            return pv


    def predict_swap(self, structure, ind1 = None, ind2 = None, correlation = False):
        """Predict property difference with the optimal cluster expansion model.

        **Parameters:**

        ``structure``: Structure object
            structure object to calculate property difference to.

        ``ind1``: int
            index of first atom position has been swapped

        ``ind2``: int
            index of second atom position has been swapped

        """
        #corrsx = self.corrc.get_cluster_correlations(structure)
        try:
            cluster_orbits = self.corrc._cluster_orbits_mc
        except AttributeError:
            raise AttributeError("Cluster_orbit set is not predefined, look at the documentation.")
        corrs = np.zeros(len(cluster_orbits))

        sigma_ind1=structure.sigmas[ind1]
        sigma_ind2=structure.sigmas[ind2]

        ems_ind1=structure.ems[ind1]
        ems_ind2=structure.ems[ind2]

        for icl, cluster_orbit in enumerate(cluster_orbits):
            for cluster in cluster_orbit:
                cluster_atomic_idxs = cluster.get_idxs()
                contribute = False
                if ind1 in cluster_atomic_idxs:
                    cluster_alphas = cluster.alphas
                    sigmas = [structure.sigmas[cl_idx] for cl_idx in cluster_atomic_idxs]
                    ems = [structure.ems[cl_idx] for cl_idx in cluster_atomic_idxs]

                    cf1 = 1.0
                    for i,alpha in enumerate(cluster_alphas):
                        cf1 *= self.corrc.site_basis_function(alpha, sigmas[i], ems[i])

                    clind =cluster_atomic_idxs.index(ind1)
                    sigmas[clind] = sigma_ind2
                    ems[clind] = ems_ind2
                    if ind2 in cluster_atomic_idxs:
                        clind =cluster_atomic_idxs.index(ind2)
                        sigmas[clind] = sigma_ind1
                        ems[clind] = ems_ind1

                    cf0 = 1.0
                    for i,alpha in enumerate(cluster_alphas):
                        cf0 *= self.corrc.site_basis_function(alpha, sigmas[i], ems[i])

                    corrs[icl] += cf1
                    corrs[icl] += (-1)*cf0

                elif ind2 in cluster_atomic_idxs:
                    cluster_alphas = cluster.alphas
                    sigmas = [structure.sigmas[cl_idx] for cl_idx in cluster_atomic_idxs]
                    ems = [structure.ems[cl_idx] for cl_idx in cluster_atomic_idxs]

                    cf1 = 1.0
                    for i,alpha in enumerate(cluster_alphas):
                        cf1 *= self.corrc.site_basis_function(alpha, sigmas[i], ems[i])

                    clind =cluster_atomic_idxs.index(ind2)
                    sigmas[clind] = sigma_ind1
                    ems[clind] = ems_ind1

                    cf0 = 1.0
                    for i,alpha in enumerate(cluster_alphas):
                        cf0 *= self.corrc.site_basis_function(alpha, sigmas[i], ems[i])

                    corrs[icl] += cf1
                    corrs[icl] += (-1)*cf0

            corrs[icl] /= len(cluster_orbit)
        corrs = np.around(corrs,decimals=12)

        if correlation:
            return corrs

        if self.estimator is not None:
            if (self.estimator.intercept_ > 1.e-15):
                inter=self.estimator.intercept_
                pv = np.subtract(self.estimator.predict(corrs.reshape(1,-1))[0],inter)
                return pv
            else:
                return self.estimator.predict(corrs.reshape(1,-1))[0]
        else:
            pv = 0
            for i in range(len(corrs)):
                pv = pv + self.ecis[i]*corrs[i]
            return pv

    def report_errors(self, sset):
        """Report fit and CV scores

        **Parameters**:

        ``sset``: StructuresSet object
            the scores are computed for the give structures set
        """
        errfit = self.get_errors(sset)
        errcv = self.get_cv_score(sset)

        print("\n+-----------------------------------------------------------+")
        print("|                Report of Fit and CV scores                |")
        print("+-----------------------------------------------------------+")
        print("|{0:<19s}|{1:^19s}|{2:^19s}|".format("","Fit","CV"))
        print("+-----------------------------------------------------------+")
        print("|{0:^19s}|{1:^19.5f}|{2:^19.5f}|".format("RMSE", errfit["RMSE"], errcv["RMSE-CV"]))
        print("|{0:^19s}|{1:^19.5f}|{2:^19.5f}|".format("MAE", errfit["MAE"], errcv["MAE-CV"]))
        print("|{0:^19s}|{1:^19.5f}|{2:^19.5f}|".format("MaxAE", errfit["MaxAE"], errcv["MaxAE-CV"]))
        print("+-----------------------------------------------------------+\n")

    def get_errors(self,sset):
        """Compute RMSE, MAE and MaxAE for model in structures set.
        """
        calc_vals = sset.get_property_values(property_name = self.property)
        predictions = sset.get_predictions(self)
        rmse = 0
        mae = 0
        maxae = 0
        i_sample = 0
        i_sample_max = 0
        for e,p in zip(calc_vals,predictions):
            rmse += (e-p)*(e-p)
            mae += np.abs(e-p)
            if np.abs(e-p) > maxae:
                i_sample_max = i_sample
                maxae = np.abs(e-p)
                
            i_sample += 1
                
        rmse = np.sqrt(rmse/len(calc_vals))
        mae = mae/len(calc_vals)

        return {"RMSE":rmse, "MAE": mae, "MaxAE":maxae, "sample_index_max":i_sample_max}

    def get_all_errors(self, sset, srtd = False):
        """Return absolute errors of every structure in a structures set.

        **Parameters:**

        ``sset``: StructuresSet object
            the errors are calculated for every structure in the structures set
        ``srtd``: Boolean (default: ``False``)
            if ``True`` the output errors are sorted from smallest to largest. 

        **Returns:**
        structured numpy array with fields::
            
            [('index','<i4'), ('error','<f4')]

        i.e., every element of the array contains a tuple with an integer, the structure index
        and a float, the absolute error
        """
        calc_vals = sset.get_property_values(property_name = self.property)
        predictions = sset.get_predictions(self)

        aes = []
        i_sample = 0
        for e,p in zip(calc_vals,predictions):
            aes.append((i_sample, np.abs(e-p)))
            i_sample += 1

        straes = np.array(aes, dtype=[('index','<i4'), ('error','<f4')])
        
        if srtd:
            saes = np.sort(straes, order='error')
            return saes
        else:
            return straes

    def get_cv_score(self,sset,fit_params=None):
        """Get leave-one-out cross-validation score over structures set.

        **Parameters:**

        ``fit_params``: dictionary
            Parameters to pass to the fit method of the estimator.
        """
        from sklearn.model_selection import cross_val_score, cross_validate, cross_val_predict
        from sklearn.model_selection import LeaveOneOut
        from sklearn.metrics import mean_squared_error
        from sklearn import linear_model
        
        X = self.corrc.get_correlation_matrix(sset)
        y = sset.get_property_values(self.property)

        # cross_val_score internally clones the estimator, so the optimal one in Model is not changed.
        cvs = cross_val_score(self.estimator, X, y, fit_params=fit_params, cv=LeaveOneOut(), scoring = 'neg_mean_squared_error')
        pred_cv = cross_val_predict(self.estimator, X, y, fit_params=fit_params, cv=LeaveOneOut())
        
        absolute_errors = np.sqrt(-cvs)
        cv = np.sqrt(-np.mean(cvs))
        maxae = np.amax(absolute_errors)
        mae = np.mean(absolute_errors)
        
        return {"RMSE-CV": cv, "MAE-CV": mae, "MaxAE-CV": maxae, "Predictions-CV":pred_cv}

class ModelBuilder():
    """Model class

    Objects of this class represent a cluster expansion model.

    **Parameters:**

    ``basis``: string (Default: ``"trigonometric"``)
        Basis set used to calculate the matrix of cluster correlations. For a detailed 
        description and possible values of this parameter, look at the documentation
        for parameter ``basis`` in :class:`CorrelationsCalculator <clusterx.correlations.CorrelationsCalculator>`

    ``selector_type``: string
        Cluster selector type. For the possible values, look at the
        documentation for the parameter ``selector_type`` in the 
        :class:`ClustersSelector <clusterx.clusters_selector.ClustersSelector>` class.

    ``selector_opts``: dictionary
        Cluster selector options. For the possible values, look at the
        documentation for the parameter ``selector_opts`` in the 
        :class:`ClustersSelector <clusterx.clusters_selector.ClustersSelector>` class.

    ``estimator_type``: string
        Estimator type. For the possible values, look at the documentation
        for the parameter ``estimator_type`` in the
        :class:`EstimatorFactory <clusterx.estimators.estimator_factory.EstimatorFactory>` class.

    ``estimator_opts``: dictionary
        Estimator options. For the possible values, look at the documentation
        for the parameter ``estimator_opts`` in the
        :class:`EstimatorFactory <clusterx.estimators.estimator_factory.EstimatorFactory>` class.

    ``filepath``: string (default: None)
        If not None, it must be the path to a pickle file generated with ModelBuilder.serialize(). All other 
        parameters are overriden
    """
    def __new__(cls, *args, **kwargs):
        
        if len(args)==0 and len(kwargs)==0:
            inst = super(ModelBuilder,cls).__new__(cls, *args, **kwargs)
            return inst
        
        elif "filepath" in kwargs:
            filepath = kwargs["filepath"]
            
            with open(filepath,"rb") as f:
                inst = pickle.load(f)
                
            if not isinstance(inst, cls):
                raise TypeError('Unpickled object is not of type {}'.format(cls))
            
            return inst
                
        else:
            inst = super(ModelBuilder,cls).__new__(cls)
            inst.initialize(*args,**kwargs)
            return inst


    def __init__(self,
                 basis="trigonometric",
                 selector_type="identity",
                 selector_opts={},
                 estimator_type="skl_LinearRegression",
                 estimator_opts={},
                 filepath = None):
        pass
    
    def initialize(self,
                   basis="trigonometric",
                   selector_type="identity",
                   selector_opts={},
                   estimator_type="skl_LinearRegression",
                   estimator_opts={},
                   filepath = None):

        self.basis = basis
        self.selector_type = selector_type
        self.selector_opts = selector_opts
        self.estimator_type = estimator_type
        self.estimator_opts = estimator_opts

        self.ini_comat = None
        
        self.opt_cpool = None
        self.opt_corrc = None
        self.opt_comat = None
        self.opt_estimator = None
        self.selector = None

    def get_selector(self):
        """
        Return selector used in build.

        When the ``build`` method is called, a ``ClustersSelector`` object
        is created to perform the cluster selection task. This selector
        can be obtained by calling this function.
        """
        return self.selector

    def get_initial_comat(self):
        """Return initial correlation matrix

        Return the initial correlation matrix, i.e. the full correlation matrix before 
        cluster selection is performed.
        """
        return self.ini_comat

    def get_opt_cpool(self):
        """
        Return optimal clusters pool found in build.

        When the ``build`` method is called, a ``ClustersSelector`` object
        is created to perform the cluster selection task. The selected clusters pool
        can be obtained by calling this function.
        """
        return self.opt_cpool

    def get_estimator(self):
        """Return estimator object used to create the optimal model
        """
        return self.opt_estimator

    def build(self, sset, cpool, prop, corrc = None, verbose = False):
        """Build optimal cluster expansion model

        Acts as a Model factory.

        **Parameters:**

        ``sset``: StructuresSet object
            structures set used for model training.

        ``cpool``: ClustersPool object
            clusters pool from which to select the best model using the method
            indicated in ``selector_type``.

        ``prop``: String
            Property name. Must be a valid name as stored in ``sset``. The list of names
            can be obtained using ``sset.get_property_names()``.

        ``corrc``: CorrelationsCalculator object (default: None)
            If not None, cpool and basis are overriden

        """
        self.sset = sset
        self.plat = cpool.get_plat()
        self.prop = prop

        if verbose: print("ModelBuilder: initialize correlations calculator")

        if corrc is None:
            self.cpool = cpool
            corrc = CorrelationsCalculator(self.basis, self.plat, self.cpool)
        else:
            self.cpool = corrc._cpool
            self.basis = corrc.basis
        
        if verbose: print("ModelBuilder: Build correlations matrix")
        self.ini_comat = corrc.get_correlation_matrix(self.sset, verbose=verbose)
        
        self.target = self.sset.get_property_values(property_name = self.prop)

        # Select optimal clusters using the clusters_selector module
        self.selector = ClustersSelector(basis=self.basis, method=self.selector_type, **self.selector_opts)
        self.opt_cpool = self.selector.select_clusters(sset, cpool, prop, comat = self.ini_comat)
        self.opt_comat = self.selector.optimal_comat
        
        self.opt_corrc = CorrelationsCalculator(self.basis, self.plat, self.opt_cpool)
        #if verbose: print("ModelBuilder: Build optimal correlations matrix")
        #self.opt_comat = self.opt_corrc.get_correlation_matrix(self.sset, verbose=verbose)

        # Find out the ECIs using an estimator
        self.opt_estimator = EstimatorFactory.create(self.estimator_type, **self.estimator_opts)
        self.opt_estimator.fit(self.opt_comat,self.target)

        return Model(self.opt_corrc, prop, estimator = self.opt_estimator)


    def serialize(self, filepath = "MODELBDR.pickle"):
        """ Serialize model into pickle file

        Only pickle format is supported.

        **Parameters:**
        
        ``filepath``: string (default: "MODELBDR.pickle")
            file path of the pickle file to serialize the model builder object
        """
        #from pathlib import Path
        #import os
        
        with open(filepath,"wb") as f:
            pickle.dump(self,f)


        
