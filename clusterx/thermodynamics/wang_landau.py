# Copyright (c) 2015-2019, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

## packages needed for WangLandau
from clusterx.structure import Structure

## packages needed for ConfigurationalDensityOfStates
import json
import math
import numpy as np
from copy import deepcopy

class WangLandau():
    """Wang Landau class

    **Description**:
        Objects of this class are used to obtain the configurational density of states 
        by performing samplings using the Wang Landau method. For details about the method, 
        see F. Wang and D.P. Landau, PRL 86, 2050 (2001). 

        It is initialized with:
        
        - a Model object, that enable to calculate the enrgy of a structure,
    
        - a SuperCell object, in which the sampling is performed,
    
        - secification of teh thermodynamic ensemble:

            If ``ensemble`` is 'canonical', the composition for the sampling is defined with ``nsubs``. In case               
            of multilattices, the sublattice for the sampling can be refined with ``sublattice_indices``.

            If ``ensemble`` is 'gandcanonical', the sublattice is defined with ``sublattices_indices``. 

    **Parameters**:

    ``energy_model``: Model object
        Model used for acceptance and rejection. Usually, the Model enables to
        calculate the total energy of a structure.

    ``scell``: SuperCell object
        Simulation cell in which the sampling is performed.

    ``ensemble``: string (default: ``canonical``) 
        ``canonical`` allows for swaps of atoms that conserve the concentration defined with ``nsubs``.

        ``grandcanonical`` allows for replacing atoms in the sub-lattices defined with ``sublattice_indices``. 
        (So far, ``grandcanonical`` is not yet implemented.)

    ``nsubs``: dictionary (default = None)
        Defines the number of substituted atoms in each sub-lattice of the
        Supercell in which the sampling is performed.

        The format of the dictionary is as follows::

            {site_type1:[n11,n12,...], site_type2:[n21,n22,...], ...}

        It indicates how many substitutional atoms of every kind (n#1, n#2, ...)
        may replace the pristine species for sites of site_type#
        (see related documentation in SuperCell object).

    ``sublattice_indices``: list of integers (default: None)
        Defines the sublattices for the grand canonical sampling. 
        Furthermore, it can be used to limit the canonical sampling 
        to a reduced number of sublattices. E.g. in the case of nsubs = {0:[4,6], 1:[4]}. Here, sublattices 0 and 1 
        contain substitutional sites, but only a sampling in sublattice 0 is wanted. Then, put ``sublattice_indices`` = [0].

    ``chemical_potentials``: dictionary (default: None)
        Define the chemical potentials used for samplings in the grand canonical ensemble.

    ``fileprefix``: string (default: 'cdos')
        Prefix for files in which the configurational density of states is saved.

    ``predict_swap``: boolean (default: False)
       If set to **True**, this parameter makes the sampling faster by calculating the correlation difference of the 
       proposed structure with respect to the previous structure.

    ``error_reset``: integer (default: None)
       If not **None**  and ``predict_swap`` equal to **True**, the correlations are calculated as usual (no differences) every n-th step.

    .. todo:
        Samplings in the grand canonical ensemble are not yet possible.

    """

    def __init__(self, energy_model, scell = None, nsubs = None, ensemble = "canonical", sublattice_indices = [], chemical_potentials = None, predict_swap = False, error_reset = None):
        self._em = energy_model
        self._scell = scell
        self._nsubs = nsubs
        #print(self._nsubs)

        if not sublattice_indices:
            try:
                self._sublattice_indices = [k for k in self._nsubs.keys()]

                if ensemble == "canonical":
                    for key in self._nsubs.keys():
                        if all([ subs == 0 for subs in self._nsubs[key] ]):
                            self._sublattice_indices.remove(key)

            except AttributeError:
                raise AttributeError("Index of sublattice is not properly assigned, look at the documentation.")
        else:
            self._sublattice_indices = sublattice_indices
        #print(self._sublattice_indices)

        if not self._sublattice_indices:
            import sys
            sys.exit('Indices of sublattice are not correctly assigned, look at the documatation.')
        self._predict_swap = predict_swap

        self._ensemble = ensemble
        self._chemical_potentials = chemical_potentials
        self._error_reset = error_reset

        if self._error_reset is not None:
            self._error_steps = int(self._error_reset)
            self._x = 1

        
    def wang_landau_sampling(self, energy_range=[-2,2], energy_bin_width=0.2, f_range=[np.round(math.exp(1),8), np.round(math.exp(1.0e-4),8)], update_method='square_root', flatness_conditions=[[0.5,np.round(math.exp(1e-1),1)],[0.80,np.round(math.exp(1e-3),3)],[0.90,np.round(math.exp(1e-5),5)],[0.95,np.round(math.exp(1e-7),7)],[0.98,np.round(math.exp(1e-8),8)]], initial_decoration = None, serialize = False, filename = "cdos.json", serialize_during_sampling = False, restart_from_file = False, **kwargs):
        """Perform Wang Landau simulation

        **Description**: 
            The Wang-Landau algorithm uses the fact that a random walk in energy space a probability proportional to 
            :math:`1/g(E)`, with g(E) being the configurational density of states, yields a flat histogram in energy 
            (details see: F. Wang and D.P. Landau, PRL 86, 2050 (2001)). 
            The energy of a visited structure is calcualted with ``energy_model``. 

            During the sampling, a new structure at step i is accepted with the probability given 
            by :math:`\min( 1, \exp( - g(E_i)/g(E_{i-1}) ) )`

            If a step is accepted, :math:`g(E_i)` of energy bin :math:`E_i` is updated with a modification factor :math:`f`.
            If a step is rejected, :math:`g(E_{i-1})` of the previous energy bin :math:`E_{i-1}` with a modification factor :math:`f`.

            The sampling procedure is a nested loop:
        
            - Inner loop: Generation of a flat histogram in energy for a fixed :math:`f`.

            - Outer loop: Gradual reduction of :math:`f` to increas the accuracy of the :math:`g(E)`.

            The initial modification factor is usually :math:`f=\exp(1)`. Since it is large, it ensures to reach all energy levels quickly. 
            In the standard procedure, the modification factor is reduces from :math:`f` to  :math:`\sqrt{f} ` for the next inner loop. 
            :math:`f` is a measure of the accuracy for :math:`g(E)`: The lower the value of :math:`f`, the higher the accuracy. The 
            sampling stops, if the next :math:`f` is below the threshold :math:`f_{final}`.

            A histogram is considered as flat, 
            if the lowest number of counts of all energy bins is above a given percentage of the mean value.
            The percentage can be set for each :math:`f`. Usually, the lower :math:`f`, the larger the percentage can be set. 

        **Parameters**:

        ``energy_range``: list of two floats [E_min, E_max] (default: [-2,2])
            Defines the energy range starting from energy E_min (center of first energy bin) 
            until energy E_max.

        ``energy_bin_width``: float (default: 0.2)
            Bin width w of each energy bin. 
            
            I.e., energy bins [E_min, E_min+w, E_min+2*w, ..., E_min+n*w ], if E_min+(n+1)*w would be larger than E_max.

        ``f_range``: list of two floats (default: [2.71828182, 1.00010000])
            List defines the initial modification factor :math:`f_1` and the threshold for the last modification factor :math:`f_{final}`. 
            I.e. [:math:`f_1`, :math:`f_{ final}`]
        
        ``update_method``: string (default: 'square_root')
            Defines the method of how the modification factor is reduced. (for now: only `square_root` implemented)

        ``flatness_conditions``: list 
            Defines the flatness condition via the percentage :math:`p` for each modification factor. I.e.
          
            [[:math:`p_1`, :math:`f_1`], [:math:`p_2`, :math:`f_2`], [:math:`p_3`, :math:`f_3`], ... , [:math:`p_{final}`, :math:`f_{final}`]]

            That means, the percentage is :math:`p_1` for all f > :math:`f_1`. If :math:`f_1 \geq f > f_2`, the flatness condition is defined via 
            :math:`p_2`, etc. 
            
            The default usually produces reasonable results and is defined as:
            Default: [[0.5, 1.1], [0.8, 1.001], [0.9, 1.00001],[0.95, 1.0000001], [0.98, 1.00000001]]

        ``initial_decoration``: list of integers
            Atomic numbers of the initial structure, from which the sampling starts.
            If **None**, sampling starts with a structure randomly generated.
        
        ``serialize``: boolean (default: False)
            If **True**, the ConfigurationalDensityOfStates object is serialized into a JSON file after the sampling. 

        ``filename``: string (default: ``cdos.json``)
            Name of a Json file in which the ConfigurationalDensityOfStates is serialized 
            after the sampling if ``serialize`` is **True**.

        ``serialize_during_sampling``: boolean (default: False)
            If **True**, the ConfigurationalDensityOfStates object is serialized every time after a flat histogramm is reached 
            (i.e. the inner loop is completed). This allows for studying the CDOS while the final :math:`f` is not yet reached. 

        ``**kwargs``: keyworded argument list, arbitrary length
            These arguments are added to the ConfigurationalDensityOfStates object that is initialized in this method.
        
        **Returns**: ConfigurationalDensityOfStates object
            Object contains the configurational density of states (CDOS) obtained from the last outer loop plus the CDOSs obtained 
            from the previous outer loops.
        
        """
        import math
        from clusterx.utils import poppush       

        if initial_decoration is not None:
            struc = Structure(self._scell, initial_decoration, mc = True)
            conc = struc.get_fractional_concentrations()
            nsites = struc.get_nsites_per_type()
            check_dict = {}
            for key in conc.keys():
                ns = nsites[key]
                nl = []
                for i,cel in enumerate(conc[key]):
                    if i == 0:
                        continue
                    else:
                        nl.append(int(cel*ns))
                check_dict.update({key:nl})

            from clusterx.utils import dict_compare
            bol = dict_compare(check_dict,self._nsubs)
            if not bol:
                import sys
                sys.exit("Number of substitutents does not coincides with them from the inital decoration.")
                              
        else:
            if self._nsubs is not None:
                struc = self._scell.gen_random(self._nsubs, mc = True)
            else:
                struc = self._scell.gen_random(mc = True)

        if restart_from_file:
            cd = ConfigurationalDensityOfStates(filename = filename, scell = self._scell, read = True, **kwargs )

            cdos = []
            for l,eb in enumerate(cd._energy_bins):
                cdos.append([eb,cd._cdos[l],0])
            cdos = np.asarray(cdos)
        
            if update_method == 'square_root':
                f = math.sqrt(cd._f)
                cd._update_method = update_method
            else:
                import sys
                sys.exit('Different update method for f requested. Please see documentation')

            if f_range[1] < cd._f_range[1]:
                cd._f_range = f_range
            else:
                f_range = cd._f_range
            
            checkf, indt = self.compare_flatness_conditions(cd._flatness_conditions, flatness_conditions)

            if not checkf:
                checkff, indtt = self.compare_flatness_conditions(flatness_conditions)
                if checkff:
                    flatness_conditions = cd._flatness_conditions
                else:
                    self._flatness_conditions = flatness_conditions
                    print('New flatness conditions are applied.')
            fi = 0
            while f < flatness_conditions[fi][1]:
                fi += 1
                if fi < len(flatness_conditions):
                    histogram_flatness = float(flatness_conditions[fi][0])
                else:
                    histogram_flatness = float(0.99)
                    break

        else:
            cd = ConfigurationalDensityOfStates(filename = filename, scell = self._scell, energy_range = energy_range, energy_bin_width = energy_bin_width, f_range = f_range, update_method = update_method, flatness_conditions = flatness_conditions, nsubs = self._nsubs, ensemble = self._ensemble, sublattice_indices = self._sublattice_indices, chemical_potentials = self._chemical_potentials, **kwargs )
            
            cdos=[]
            eb = energy_range[0]
            while eb < energy_range[1]:
                cdos.append([eb,1,0])
                eb = eb+energy_bin_width
            cdos = np.asarray(cdos)

            f = f_range[0]
            fi = 0
            histogram_flatness = flatness_conditions[fi][0]

        self._em.corrc.reset_mc(mc = True)
        e = self._em.predict(struc)
        #print(e)

        for k,d in enumerate(cdos):
            if e < d[0]:
                if k == 0:
                    inde = k
                else:
                    diffe1 = float(e-cdos[k-1][0])
                    diffe2 = float(d[0]-e)
                    if diffe1 < diffe2:
                        inde = k-1
                    else:
                        inde = k
                break

        cdos[inde][1] = cdos[inde][1]+math.log(f)
        cdos[inde][2] = cdos[inde][2]+1
        g = cdos[inde][1]
        
        while f > f_range[1]:

            struc, e, g, inde, cdos, hist_cond = self.flat_histogram(struc, e, g, inde, f, cdos, histogram_flatness)
            #print(self._nsubs)

            cd.add_cdos(cdos, f, histogram_flatness, hist_cond)
            
            if serialize_during_sampling:
                cd.serialize()
            
            for m,d in enumerate(cdos):
                if d[2] > 0:
                    cdos[m][2] = 0
        
            if update_method == 'square_root':
                f = math.sqrt(f)
            else:
                import sys
                sys.exit('Different update method for f requested. Please see documentation')
                
            if f < flatness_conditions[fi][1]:
                fi += 1
                if fi < len(flatness_conditions):
                    histogram_flatness = float(flatness_conditions[fi][0])
                else:
                    histogram_flatness = float(0.99)
                    

        if serialize:
            cd.serialize()
                    
        return cd
    
            
    def flat_histogram(self, struc, e, g, inde, f, cdos, histogram_flatness):

        hist_min = 0
        hist_avg = 1
        lnf = math.log(f)

        i = 0
        while (hist_min < histogram_flatness*hist_avg) or (i < 10):
            
            for i in range(500):
                struc, e, g, inde, cdos = self.dos_step(struc, e, g, inde, lnf, cdos)
                    
            hist_sum=0
            i=0
            for d in cdos:
                if float(d[2])>0:
                    i += 1
                    hist_sum = hist_sum+d[2]
                    if i == 1:
                        hist_min = d[2]
                    elif d[2] < hist_min:
                        hist_min = d[2]
            hist_avg = (hist_sum)/(1.0*i)
                                
        #print("\n")
        #print(f)
        
        #print(hist_min,hist_avg)
        #print("Flat histogram for f=%2.12f and flatness condition g_min > %1.2f * g_mean finished"%(f,histogram_flatness))
        
        return struc, e, g, inde, cdos, [hist_min, hist_avg]
                                                                                                                                                                                                                                                    
    def dos_step(self, struc, e, g, inde, lnf, cdos):
    
        ind1, ind2, site_type, rindices = struc.swap_random(self._sublattice_indices)

        if self._predict_swap:
            if self._error_reset:
                if (self._x > self._error_steps):
                    self._x = 1
                    e1 = self._em.predict(struc)
                else:
                    self._x += 1
                    de = self._em.predict_swap(struc, ind1 = ind1 , ind2 = ind2)
                    e1 = e + de
            else:
                de = self._em.predict_swap(struc, ind1 = ind1, ind2 = ind2)
                e1 = e + de
        else:
            e1 = self._em.predict(struc)

        for k,d in enumerate(cdos):
            if e1 < d[0]:
                if k == 0 :
                    g1 = d[1]
                    kinde = 0
                else:
                    diffe1 = float(e1-cdos[k-1][0])
                    diffe2 = float(d[0]-e1)
                    if diffe1 < diffe2:
                        g1 = cdos[k-1][1]
                        kinde = k-1
                    else:
                        g1 = d[1]
                        kinde = k
                break

        if g >= g1:
            accept_swap = True
            trans_prob = 1
        else:
            trans_prob = math.exp(g-g1)
            if np.random.uniform(0,1) <= trans_prob:
                accept_swap = True
            else:
                accept_swap = False

        if accept_swap:
            g = g1+lnf
            e = e1
            inde = kinde
            
            cdos[kinde][1] = g
            cdos[kinde][2] = cdos[kinde][2]+1
            
        else:
            g = g+lnf
            
            cdos[inde][1] = g
            cdos[inde][2] = cdos[inde][2]+1
            struc.swap(ind2, ind1, site_type = site_type, rindices = rindices)

        return struc, e, g, inde, cdos

    def compare_flatness_conditions(self,flatness1,flatness2 = [[0.5,np.round(math.exp(1e-1),1)],[0.80,np.round(math.exp(1e-3),3)],[0.90,np.round(math.exp(1e-5),5)],[0.95,np.round(math.exp(1e-7),7)],[0.98,np.round(math.exp(1e-8),8)]]):
        indt = -1
        flagcond = True
        for it,fct in enumerate(flatness1):
            for ix, fctx in enumerate(fct):
                if fctx != flatness2[it][ix]:
                    flagcond = False
                    indt = it
                    break

        return flagcond, indt

        
class ConfigurationalDensityOfStates():
    """ConfigurationalDensityOfStates class

    **Description**:
        Objects of this class are use to store and access the configurational density of states (CDOS) that 
        was generated from a Wang-Landau sampling. 
    
        Since the Wang-Landau algorithm is iterative, it contains the CDOS for each iteration of the outer loop. 

    **Parameters**:

    ``scell``: SuperCell object (default: None)
        Super cell in which the Wang-Landau sampling is performed.

    ``filename``: string (default: cdos.json)
        The trajectoy can be stored in a json file ``filename``.

    ``read``: boolean (default: False)
        If **True**, the CDOS is read from the Json file ``filename``.

    ``**kwargs``: keyword arguments

        Keyword arguments can be used to store additional information about the parameters used for 
        the WangLandau.wang_landau_sampling rountine. This will be saved in the Json file 
        ``filename`` under ``sampling_info``, if the object is serialized. 

    """

    def __init__(self, scell = None, filename="cdos.json", read = False, **kwargs):

        self._filename = filename
        self._scell = scell
        
        if read:        
            self.read()

        else:
            self._energy_bins = []
            self._cdos = []
            self._histogram = []

            self._stored_cdos = []
        
            self._scell = scell
            #self._save_nsteps = kwargs.pop("save_nsteps",10)
            #self._write_no = 0

            self._models = kwargs.pop("models",[])
            
            self._f = None
            self._flatness_condition = None
            self._histogram_minimum = None
            self._histogram_average = None

            # Patrameters from WangLandau.wang_landau_sampling routine
            self._energy_range = kwargs.pop('energy_range',None)
            self._energy_bin_width = kwargs.pop('energy_bin_width',None)
            self._f_range = kwargs.pop('f_range',None)
            self._update_method = kwargs.pop('update_method',None)
            self._flatness_conditions = kwargs.pop('flatness_conditions',None)
            self._nsubs = kwargs.pop('nsubs',None)
            self._ensemble = kwargs.pop('ensemble',None)
            self._sublattice_indices = kwargs.pop('sublattice_indices',None)
            self._chemical_potentials = kwargs.pop('chemical_potentials',None)
            self._keyword_arguments = kwargs
            
        self._normalized = False
        self._cdos_normalized = None

    def add_cdos(self, cdos, f, flatness_condition, hist_cond):
        """Add entry of configurational of states that is obtained after a flat histgram reached 
           (corresponding to a fixed modification factor :math:`f`).

        """
        if len(self._energy_bins) == 0:
            self._energy_bins = [c[0] for c in cdos]
            
        self._cdos = [c[1] for c in cdos]
        self._histogram = [c[2] for c in cdos]
        self._f = f
        self._flatness_condition = flatness_condition

        self._histogram_minimum = hist_cond[0]
        self._histogram_average = hist_cond[1]
        
        self._stored_cdos.append(dict([('cdos',self._cdos),('histogram',self._histogram),('modification_factor',f),('flatness_condition',flatness_condition),('histogram_minimum', hist_cond[0]),('histogram_average',hist_cond[1])]))

        
    def get_cdos(self, ln = False, normalization = True, discard_empty_bins = True, set_normalization_ln = None, modification_factor = None):
        """Returns the energy_bins and configurational density of states (CDOS) as arrays, respectively.
           
        **Parameters**:
            
        ``ln``: boolean (default: False)
            Decides whether the logarithm of CDOS is returned.
        
        ``normalization``: boolean (default: True)
            Decides whether the configurational density of states is normalized.
        
        ``discard_empty_bins``: boolean (default: True)
            Decides whether the energy bins that are not visited during the sampling are kept.

        ``set_normalization_ln``: float (default: None)
            If not **None**, the normalization is set by the user. The given value is substracted 
            from the logarithm of the CDOS for each bin.

        ``modification_factor``: float (default: None)
            If **None**, the CDOS from the last iteration is returned. If not **None**, the CDOS 
            corresponding to the given modification factor is returned.
                    
        """
        if modification_factor is None:
            g = self._cdos
        else:
            from utils import isclose
            for gj,gstored in enumerate(self._stored_cdos):
                if isclose(modification_factor,gstored['modification_factor'],tol = 1.0e-8):
                    g = gstored['cdos']
                
        if normalization:
            eb = []

            _gone = g[0]
            _gsum = np.sum([math.exp(gel-_gone) for gel in g])
            _lngsum = math.log(_gsum)
            _nsites = self._scell.get_nsites_per_type()
            _nkey = [int(k) for k in self._nsubs.keys()]
            for _nk in _nkey:
                #_ns = self._nsubs[str(_nk)]
                _ns = self._nsubs[_nk]
                _nsite = _nsites[_nk]
                
            import scipy.special as scipysp
            _binomcoeff = scipysp.binom( _nsite, _ns)
            gc = []
            for gi, ge in enumerate(g):
                if ge > 1.000000001:
                    if set_normalization_ln is not None:
                        gt = ge - float(set_normalization_ln)
                    else:
                        gt = ge - _gone - _lngsum + math.log(_binomcoeff)
                        
                    if not ln:
                        gc.append(math.exp(gt))
                    else:
                        gc.append(gt)
                    eb.append(self._energy_bins[gi])
                    
                else:
                    if not discard_empty_bins:
                        gc.append(float(0))
                        eb.append(self._energy_bins[gi])
                
            self._cdos_normalized = gc
            self._energy_bins_cdos_normalized = eb
            
            return eb, gc
        
        else:
            if discard_empty_bins:
                gc = []
                eb = []
                for gi,ge in enumerate(g):
                    if ge > 1.000000001:
                        if not ln:
                            gc.append(math.exp(ge))
                        else:
                            gc.append(ge)
                        eb.append(self._energy_bins[gi])
                        
                return eb, gc
            
            else:
                if not ln:
                    _expg = [math.exp(ge) for ge in g]
                    return self._energy_bins_bins, expg
                else:

                    return self._energy_bins_bins, g
        

    def calculate_thermodynamic_property(self, temperatures, boltzmann_constant, scale_factor = None, prop_name = "U", modification_factor = None):
        """Calculate the thermodynamic property with name ``prop_name`` for the temperature list given 
           with ``temperatures``. 

        **Parameters**:
        
        ``temperatures``: list of floats 
            Temperatures for which the thermodynamic property is calculated.

        ``boltzmann_constant``: float 
            Boltzmann constant

        ``scale_factor``: list of floats (default: None)
            List is used to adjust the factor :math:`k_B T` to the same units as the energiess of the energy bins have.
        
        ``prop_name``: string (default: U)
            Name of thermodynamic property.

            If **Z**, the partition function is calculated.

            If **U**, the internal energy is calculated. Units are the same as for the energy of the energy bins, [E].

            If **C_p**, the isobaric specific heat at zero pressure is calculated. Units :math:`k_B` [scale_factor].

            If **F**, the free energy is calculated. Units are the same as for the energy [E].

            If **S**, the entropy is calculated. Units :math:`k_B` [scale_factor].

        ``modification_factor``: float (default: None)
            If **None**, the CDOS from the last iteration is used for calculuting teh thermodynamic property. 
            If not **None**, the CDOS corresponding to the given modification factor is used.

        """
        e, g = self.get_cdos(ln = True, normalization = True, discard_empty_bins = True,  modification_factor = modification_factor)

        thermoprop = np.zeros(len(temperatures))
        e0 = float(e[0])
        kb = float(boltzmann_constant)

        scale = 1
        if scale_factor is not None:
            for scf in scale_factor:
                scale *= float(scf)
        scale = np.divide(1,scale)

        for i,t in enumerate(temperatures):
            u = 0
            z = 0
            if prop_name == "C_p":
                u2 = 0
                
            for ic,cc in enumerate(g):
                boltzf = math.exp((-1)*(e[ic]-e0)*scale/(1.0*kb*t))
                u += e[ic]*math.exp(cc)*boltzf
                if prop_name == "C_p":
                    u2 += e[ic]*e[ic]*math.exp(cc)*boltzf
                
                z += math.exp(cc)*boltzf
            
            u = np.divide(u, z)

            if prop_name == "U":
                thermoprop[i] = u
                
            elif prop_name == "Z":
                thermoprop[i] = z
                
            elif prop_name == "C_p":
                u2 = u2/(1.0*z)
                thermoprop[i] = (u2-u*u)*scale/(1.0*kb*kb*t*t)
                
            else:
                f = (-1)*kb*t/(1.0*scale)*(math.log(z))+e0
                if i == 0:
                    f1 = f
                elif i == 1:
                    f2 = f
                    _dt = float(t-temperatures[0])
                    df = (f2-f1)/(1.0*_dt)
                    f = f - i*df
                else:
                    f = f - i * df
                    
                if prop_name == "F":
                    thermoprop[i] = f
                elif prop_name == "S":
                    thermoprop[i] = (u-f)/(kb*t)
                else:
                    import sys
                    sys.exit("Thermodynamic property name ``prop_name`` not correctly defined. See Documentation.")

        return thermoprop
                
                
    def wang_landau_write_to_file(self, filename = None):
        self.serialize(filename = filename)

    def serialize(self, filename = None):
        """Write ConfigurationalDensityOfStates object containing the configurational density of states to Json file 
           with name ``filename``. If ``filename`` is not defined, it uses ``filename`` defined in the initialization 
           of ConfigurationalDensityOfStates object.

        """

        if filename is not None:
            self._filename = filename

        cdosdict = {}

        cdos_info = {}
        cdos_info.update({'energy_range':self._energy_range})
        cdos_info.update({'energy_bin_width':self._energy_bin_width})
        cdos_info.update({'energy_bins':self._energy_bins})
        cdos_info.update({'f_range':self._f_range})
        cdos_info.update({'update_method':self._update_method})
        cdos_info.update({'flatness_conditions':self._flatness_conditions})
        cdos_info.update({'nsubs':self._nsubs })
        cdos_info.update({'ensemble':self._ensemble})
        cdos_info.update({'sublattice_indices':self._sublattice_indices})
        cdos_info.update({'chemical_potentials':self._chemical_potentials})
        
        for key in self._keyword_arguments.keys():
            cdos_info.update({key:self._keyword_arguments[key]})

        cdos_info.update({'super_cell_definition':self._scell.as_dict()})

        cdosdict.update({'sampling_info':cdos_info})
        
        for j,st_c_dos in enumerate(self._stored_cdos):
            cdosdict.update({str(j):st_c_dos})

        from clusterx.thermodynamics.monte_carlo import NumpyEncoder
        
        with open(self._filename, 'w+', encoding='utf-8') as outfile:
            json.dump(cdosdict, outfile, cls=NumpyEncoder, indent = 2 , separators = (',',':'))

            
    def read(self, filename = None):
        """Read ConfigurationalDensityOfStates object from Json file with name ``filename``. 
           If ``filename`` is not defined, it uses ``filename`` defined in the initialization             
           of ConfigurationalDensityOfStates object.
        
        """
        if filename is not None:
            cdosfile = open(filename,'r')
            self._filename = filename
        else:
            cdosfile = open(self._filename,'r')

        data = json.load(cdosfile)

        cdos_info = data.pop('sampling_info',None)

        if cdos_info is not None:
            self._energy_range = cdos_info.pop('energy_range',None)
            self._energy_bin_width = cdos_info.pop('energy_bin_width',None)
            self._energy_bins = cdos_info.pop('energy_bins',None)
            self._f_range = cdos_info.pop('f_range',None)
            self._update_method = cdos_info.pop('update_method',None)
            self._flatness_conditions = cdos_info.pop('flatness_conditions',None)
            self._nsubs = cdos_info.pop('nsubs',None)
            self._ensemble = cdos_info.pop('ensemble',None)
            self._sublattice_indices = cdos_info.pop('sublattice_indices',None)
            self._chemical_potentials = cdos_info.pop('chemical_potentials',None)

            superdict = cdos_info.pop('super_cell_definition',None)
            if self._scell is None:
                from ase.atoms import Atoms
                from clusterx.parent_lattice import ParentLattice
                from clusterx.super_cell import SuperCell
                if superdict is not None:
                    nsp = sorted([int(el) for el in set(superdict['parent_lattice']['numbers'])])
                    species = []
                    for n in nsp:
                        species.append(superdict['parent_lattice']['numbers'][str(n)])
                    
                    _plat = ParentLattice(atoms = Atoms(positions = superdict['parent_lattice']['positions'], cell = superdict['parent_lattice']['unit_cell'], numbers=np.zeros(len(species)), pbc = np.asarray(superdict['parent_lattice']['pbc'])), sites  = np.asarray(species), pbc = np.asarray(superdict['parent_lattice']['pbc']))
                    
                self._scell = SuperCell(_plat, np.asarray(superdict['tmat']))
            else:
                _sdict = self._scell.as_dict()
                from clusterx.utils import dict_compare
                _testc = dict_compare(_sdict,superdict)
                if not _testc:
                    import sys
                    sys.exit('SuperCell object given in the initialization is not equivalent to the SuperCell object stored in read file.')
            self._keyword_arguments = cdos_info

        data_keys = sorted([int(el) for el in set(data.keys())])

        self._stored_cdos = []
        for key in data_keys:
            cdict = data[str(key)]
            self._stored_cdos.append(cdict)
            
        _last_cdos_entry = self._stored_cdos[-1]
        self._cdos = _last_cdos_entry['cdos']
        self._histogram = _last_cdos_entry['histogram']

        self._f = _last_cdos_entry['modification_factor']

        self._flatness_condition = _last_cdos_entry['flatness_condition']
        self._histogram_minimum = _last_cdos_entry['histogram_minimum']
        self._histogram_average = _last_cdos_entry['histogram_average']
            
        
#Should be in monte_carlo.py already
#class NumpyEncoder(json.JSONEncoder):
#    """ Special json encoder for numpy types
#    https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable/32850511
#
#    """
#    def default(self, obj):
#        if isinstance(obj, list):
#            return obj.tolist()
#        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
#            return int(obj)
#        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
#            return float(obj)
#        elif isinstance(obj,(np.ndarray,)):
#            return obj.tolist()
#
#        return json.JSONEncoder.default(self, obj)
