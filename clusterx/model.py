# Copyright (c) 2015-2019, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

from clusterx.correlations import CorrelationsCalculator
from clusterx.estimators.estimator_factory import EstimatorFactory
from clusterx.clusters_selector import ClustersSelector
import numpy as np
import pickle
import os
import time
import warnings

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
    

    def __init__(self, corrc = None, property_name = None, estimator = None, ecis = None, filepath = None, json_db_filepath = None, standardize = False):
        pass
    
    def initialize(self, corrc = None, property_name = None, estimator = None, ecis = None, filepath = None, json_db_filepath = None, standardize = False):
        self.pickle_file = None
        self._filepath_corrc = None
        self.estimator = None

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
            self.property_name = modict.get('property_name',None)

            self.standardize =  modict.get('standardize',False)

        else:
            self.corrc = corrc
            self.ecis = ecis
            self.property_name = property_name
            self.standardize = standardize
            if standardize:
                from sklearn.preprocessing import StandardScaler
                self.stdscaler = StandardScaler()

        self._basis = None
        self._delta_e_calc = None
        if corrc is not None:
            self._basis =corrc.get_basis()
        if self.estimator is None:
            self.estimator = estimator
        self._mc = False
        self._num_mc_calls = 0
        self._mc_nclusters = 0
        self._mc_multiplicities = []
        self._mc_stime = 0
        self._mc_init_time = 0


    def reset_mc(self, mc = False):
        self._mc = mc
        self._num_mc_calls = 0
        self._mc_nclusters = 0
        self._mc_multiplicities = []
        self._mc_start_time = 0
        self._mc_init_time = 0
        self._mc_estimator_intercept = 0
        self._mc_estimator_coef = []
        


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

            modeldict = self.corrc._cpool.get_cpool_dict()

            corr_dict = {}
            corr_dict.update({'basis':self.corrc.basis})
            if self.corrc._lookup_table is not None:
                corr_dict.update({'lookup':True})
            else:
                corr_dict.update({'lookup':False})
            modeldict.update({'correlations_calculator':corr_dict})

            modict = {}
            modict.update({'property_name':self.property_name})

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
        warnings.warn("Using `get_plat()` is deprecated. Use `get_parent_lattice()` instead.", category=FutureWarning)
        return self.get_parent_lattice()
        
    def get_parent_lattice(self):
        """ Return parent lattice of the cluster expansion model.
        """
        return self.corrc._plat

    def get_ecis(self):
        """ Return array of effective cluster interactions (ECIs) of the model
        """
        if self.ecis is not None:
            return self.ecis
        else:
            if self.standardize:
                return self.estimator[-1].coef_
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
            if self.standardize:
                try:
                    corrs = self.stdscaler.transform(corrs)
                except:
                    import sys
                    sys.exit("StandardScaler of Model has not been fitted.")

            pv = 0
            for i in range(len(corrs)):
                pv = pv + self.ecis[i]*corrs[i]


            return pv


    def predict_swap(self, structure, ind1 = None, ind2 = None, correlation = False, site_types=[0]):
        """Predict property difference with the optimal cluster expansion model.

        **Parameters:**

        ``structure``: Structure object
            structure object to calculate property difference to.

        ``ind1``: int
            index of first atom position has been swapped

        ``ind2``: int
            index of second atom position has been swapped

        """
        if self._num_mc_calls == 0:
            self._mc_init_time = time.time()
            print("Info(Model): setting up dictionary of interactions.")
            
            try:
                cluster_orbits = self.corrc._cluster_orbits_mc
                self._mc_nclusters = len(cluster_orbits)
                for i in range(self._mc_nclusters):
                    self._mc_multiplicities.append(len(cluster_orbits[i]))
                
            except AttributeError:
                raise AttributeError("Cluster_orbits set has not been pre computed.")

            if self.standardize:
                import sys
                sys.exit("Predict swap does not support standardscaler")
                
            # Determine atom indexes for which to make the interactions list
            scell = structure.get_supercell()
            self._atom_indexes = []
            for st in site_types:
                for aidx in scell.get_atom_indices_for_site_type(st)[0]:
                    self._atom_indexes.append(aidx)

            # Determine ems
            self._ems = scell.get_ems()
                
                
            # Make a list of clusters
            self._clusters_list = []            

            icl = 0
            for cluster_index, cluster_orbit in enumerate(cluster_orbits):
                for cluster in cluster_orbit:
                    self._clusters_list.append({})

                    self._clusters_list[icl]["cluster_index"] = cluster_index
                    self._clusters_list[icl]["cluster_sites"] = cluster.get_idxs()
                    self._clusters_list[icl]["cluster_funcs"] = cluster.alphas
                    self._clusters_list[icl]["cluster_ems"] = self._ems.take(cluster.get_idxs())
            
                    
                    icl += 1
                    
            # Determine which interactions (clusters) contain every site
            self._interactions_dict = {}
            
            for ind in self._atom_indexes:
                self._interactions_dict[ind] = {}
                self._interactions_dict[ind]["interactions_list"] = []
                self._interactions_dict[ind]["cluster_sites_index_for_ind"] = []
                for icl in range(len(self._clusters_list)):
                    if ind in self._clusters_list[icl]["cluster_sites"]:
                        self._interactions_dict[ind]["interactions_list"].append(icl)
                        self._interactions_dict[ind]["cluster_sites_index_for_ind"].append(self._clusters_list[icl]["cluster_sites"].index(ind))

            if self.estimator is not None:
                self._mc_estimator_intercept = self.estimator.intercept_
                self._mc_estimator_coef = self.estimator.coef_

            if self._basis == 'binary-linear' or self._basis ==  'indicator-binary':
                self._delta_e_calc = self._compute_delta_e_binary_linear
            else:
                self._delta_e_calc = self._compute_delta_e
            
            self._num_mc_calls = 1
            self._mc_init_time -= time.time()
            self._mc_init_time = -self._mc_init_time
            self._mc_start_time = time.time()

        
        new_sigma = structure.sigmas[ind1]
        old_sigma = structure.sigmas[ind2]

        de1 = self._delta_e_calc(structure, ind1, old_sigma, new_sigma)

        sigma1 = structure.sigmas[ind1]
        sigma2 = structure.sigmas[ind2]
        structure.sigmas[ind1] = sigma2
        structure.sigmas[ind2] = sigma1

        de2 = self._delta_e_calc(structure, ind2, new_sigma, old_sigma)
        
        structure.sigmas[ind1] = sigma1
        structure.sigmas[ind2] = sigma2

        return de1 + de2

    def _compute_delta_e_binary_linear(self, structure, ind, old_sigma, new_sigma):
        sgn = new_sigma - old_sigma
        corrs = np.zeros(self._mc_nclusters)
        for icl in self._interactions_dict[ind]["interactions_list"]:
            cluster_index = self._clusters_list[icl]["cluster_index"]
            corrs[cluster_index] = 0

        for ifi, icl in zip(self._interactions_dict[ind]["cluster_sites_index_for_ind"],self._interactions_dict[ind]["interactions_list"]):
            cluster_index = self._clusters_list[icl]["cluster_index"]
            cluster_sites = self._clusters_list[icl]["cluster_sites"]
            sigmas = structure.sigmas.take(cluster_sites).copy()
            sigmas[ifi] = sgn
            ss = set(sigmas)

            if 0 not in ss:
                corrs[cluster_index] += sgn
            
        for i in range(self._mc_nclusters):
            corrs[i] /= self._mc_multiplicities[i]
            
        pv = 0
        
        for i in range(len(corrs)):
            pv = pv + self._mc_estimator_coef[i]*corrs[i]
        return pv

    def _compute_delta_e(self, structure, ind, old_sigma, new_sigma):

        corrs = np.zeros(self._mc_nclusters)
        for icl in self._interactions_dict[ind]["interactions_list"]:
            cluster_index = self._clusters_list[icl]["cluster_index"]
            corrs[cluster_index] = 0
            
        for icl in self._interactions_dict[ind]["interactions_list"]:
            cluster_index = self._clusters_list[icl]["cluster_index"]
            cluster_sites = self._clusters_list[icl]["cluster_sites"]
            cluster_funcs = self._clusters_list[icl]["cluster_funcs"]
            cluster_ems = self._clusters_list[icl]["cluster_ems"]
            nbodies = len(cluster_sites)
            sigmas = structure.sigmas.take(cluster_sites)
            
            cf = 1.0
            for i in range(nbodies):
                if (i == cluster_sites.index(ind)):
                    cf *= (self.corrc.site_basis_function(cluster_funcs[i], new_sigma, cluster_ems[i]) - self.corrc.site_basis_function(cluster_funcs[i], old_sigma, cluster_ems[i]))
                else:
                    cf *= self.corrc.site_basis_function(cluster_funcs[i], sigmas[i], cluster_ems[i])

            corrs[cluster_index] += cf

        for i in range(self._mc_nclusters):
            corrs[i] /= self._mc_multiplicities[i]
        corrs = np.around(corrs,decimals=12)

            
        if self.estimator is not None:
            # Intercept must be subctracted from computation of energy change.
            pv = self.estimator.predict(corrs.reshape(1,-1))[0] - self.estimator.intercept_
            return pv
        else:
            pv = 0
            for icl, corr in enumerate(corrs):
                pv = pv + self.ecis[icl]*corr
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
        calc_vals = sset.get_property_values(property_name = self.property_name)
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
        calc_vals = sset.get_property_values(property_name = self.property_name)
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
        from sklearn.model_selection import cross_val_score, cross_val_predict
        from sklearn.model_selection import LeaveOneOut
        
        X = self.corrc.get_correlation_matrix(sset)
        y = sset.get_property_values(self.property_name)

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

    ``standardize``: Boolean (defaut:``False``)
        Whether to standardize the input. Uses StandardScaler of scikit-learn.
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
                 filepath = None,
                 standardize = False):
        pass
    
    def initialize(self,
                   basis="trigonometric",
                   selector_type="identity",
                   selector_opts={},
                   estimator_type="skl_LinearRegression",
                   estimator_opts={},
                   filepath = None,
                   standardize = False):

        self.basis = basis
        self.selector_type = selector_type
        self.selector_opts = selector_opts
        self.estimator_type = estimator_type
        self.estimator_opts = estimator_opts
        self.standardize = standardize
        if "standardize" not in selector_opts.keys():
            selector_opts["standardize"] = standardize
        
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

        # Find out the ECIs using an estimator
        if not self.standardize:
            self.opt_estimator = EstimatorFactory.create(self.estimator_type, **self.estimator_opts)
        else:
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import make_pipeline

            self.opt_estimator = make_pipeline(StandardScaler(), EstimatorFactory.create(self.estimator_type, **self.estimator_opts))

        self.opt_estimator.fit(self.opt_comat,self.target)
        return Model(self.opt_corrc, prop, estimator = self.opt_estimator, standardize = self.standardize)

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


        
