# Copyright (c) 2015-2019, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

## packages needed for MonteCarlo
import random
#from clusterx.structures_set import StructuresSet
from clusterx.structure import Structure

## packages needed for MonteCarloTrajectory
import json
import math
import numpy as np
from copy import deepcopy
import sys

class WangLandau():
    """WangLandau class

    Description:
        Generate the configurational density of states by using the Wang-Landau method

    Parameters:

    ``energy_model``: Model object
        Model used for acceptance and rejection. Usually, the Model enables to
        calculate the total energy of a given configuration.

    ``scell``: SuperCell object
        Super cell in which the sampling is performed.

    ``nsubs``: dictionary
        The format of the dictionary is as follows::

            {site_type1:[n11,n12,...], site_type2:[n21,n22,...], ...}

        It indicates how many substitutional atoms of every kind (n#1, n#2, ...)
        may replace the pristine species for sites of site_type#.

        The list of site types can be obtained with the method ``ParentLattice.get_idx_subs()``
        (see related documentation).

        Defines the number of substituted atoms in each sublattice
        Supercell in which the sampling is performed.

    ``filename``: string
        Trajectory can be written to a json file with the name ``filename``.

    ``last_visited_structure_name``: string
        The structure visited at the final step of the sampling can be saved to a json file.
        Default name: last-visited-structure-mc.json

    ``sublattice_indices``: list of int
        Sampled sublattices. Each index in the list gives the site_type defining the sublattice.
        If the list is empty (default), the site_type of the sublattices are read from ``nsubs``
        Non-substituted sublattices are excluded for canonical samplings.

    ``ensemble``: string
        "canonical" allows only for swapping of atoms inside scell
        "grandcanonical"  allows for replacing atoms within the given scell
        (the number of substitutents in each sublattice is not kept)

    .. todo:
        Samplings in the grand canonical ensemble are not yet possible.

    """

    def __init__(self, energy_model, scell, nsubs, fileprefix = "cdos", sublattice_indices = [], ensemble = "canonical", predict_swap = False, error_reset = False):
        self._em = energy_model
        self._scell = scell
        self._nsubs = nsubs
        print(self._nsubs)
        self._fileprefix = fileprefix

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
        print(self._sublattice_indices)

        if not self._sublattice_indices:
            import sys
            sys.exit('Indices of sublattice are not correctly assigned, look at the documatation.')
        self._predict_swap = predict_swap

        self._ensemble = ensemble

        self._error_reset = error_reset
        if self._error_reset:
            self._error_steps = int(100000)
            self._x = 1

    def wang_landau_sampling(self, energy_range=[-2,2], energy_bin_width=0.2, f_range=[math.exp(1), math.exp(1.0e-8)], update_method='square_root', flatness_conditions=[[0.5,math.exp(1e-1)],[0.80,math.exp(1e-3)],[0.90,math.exp(1e-5)],[0.98,math.exp(1e-8)]], initial_decoration = None, write_to_db = False):
        """Perform Wang Landau simulation

        **Description**: Perfom Wang-Landau algorithm for nmc sampling
             steps at scale factor :math:`k_B T`.  The total energy
             :math:`E` for visited structures in the sampling is
             calculated from the Model ``energ_model`` of the total
             energy. During the sampling, a new structure at step i is accepted
             with the probability given by :math:`\min( 1, \exp( - (E_i - E_{i-1})/(k_B T)) )`

        **Parameters**:

        ``scale_factor``: list of floats
            From the product of the float in the list, the scale factor for the energy :math:`k_B T` is obtained.

            E.g. [:math:`k_B`, :math:`T`] with :math:`k_B` as the Boltzmann constant and :math:`T` as the temperature for the Metropolis simulation.
            The product :math:`k_B T` defines the scale factor in the Boltzmann distribution.

            Note: The unit of the product :math:`k_B T` must be the same as for the total energy :math:`E`.

        ``nmc``: integer
            Number of sampling steps

        ``initial_decoration``: Structure object
            Sampling starts with the structure defined by this Structure object.
            If initial_structure = None: Sampling starts with a structure randomly generated.

        ``write_to_db``: boolean (default: False)
            Whether to add the structure to the json database (see ``filename`` parameter for MonteCarloTrajectory initialization)

        
        **Returns**: Configurational density of states for each f
            Array with length of the number of energy bins, :math:`E_0,E_1,...E_n`. 
            For each energy bin :math:`E_i`, the natural logarithm of configurational density at :math:`E_i`, :math:`ln g(E_i)`, 
            and the number of counts for each modification factor f is output in a file.

        .. todo:
            Besides list of floats, give option to set ``scale_factor`` as float too.
        """
        import math
        from clusterx.utils import poppush       

        if initial_decoration is not None:
            struc = Structure(self._scell, initial_decoration, mc = True)
            print(struc.get_atomic_numbers)
        else:
            struc = self._scell.gen_random(self._nsubs, mc = True)

        self._em.corrc.reset_mc(mc = True)
        e = self._em.predict(struc)
        print(e)

        cdos=[]
        eb = energy_range[0]
        while eb < energy_range[1]:
            cdos.append([eb,1,0])
            eb = eb+energy_bin_width
        cdos = np.asarray(cdos)

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
        
            
        f = f_range[0]
        fi = 0
        g = 1 + math.log(f)
        histogram_flatness = flatness_conditions[fi][0]

        cdos[inde][1] = cdos[inde][1]+math.log(f)
        cdos[inde][2] = cdos[inde][2]+1

        control_flag = True
        if self._error_reset:
            errorsteps = 50000
            x = 1

        while f > f_range[1]:

            struc, e, g, inde, cdos = self.flat_histogram(struc, e, g, inde, f, cdos, histogram_flatness)
            print(self._nsubs)

            filestring=self._fileprefix+"_modificationf-"+str(f)+"_histogram_flatness-"+str(histogram_flatness)+".json"
            cd = ConfigurationalDensityOfStates(filename=filestring,scell=self._scell, modification_factor = f, flatness_condition = histogram_flatness)
            cd._cdos = cdos
            cd.wang_landau_write_to_file()
            
            for m,d in enumerate(cdos):
                if d[2] > 0:
                    cdos[m][2] = 0
        
            if update_method == 'square_root':
                f=math.sqrt(f)
            else:
                sys.exit('different update method for f required')
            print(f,flatness_conditions[fi])    
                
            if f < flatness_conditions[fi][1]:
                fi += 1
                if fi < len(flatness_conditions):
                    histogram_flatness = float(flatness_conditions[fi][0])
                else:
                    histogram_flatness = float(0.99)
                    
        print("Loop over modification factor f finished")
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
                                
        print("\n")
        print(f)
        
        print(hist_min,hist_avg)
        print("Flat histogram for f=%2.12f and flatness condition g_min > %1.2f * g_mean finished"%(f,histogram_flatness))
        
        return struc, e, g, inde, cdos
                                                                                                                                                                                                                                                    
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
        
class ConfigurationalDensityOfStates():
    """ConfigurationalDensityOfStates class

    **Description**:
        Configurational density of states from a sampling performed in the supercell scell.
        Additional information about the sampling procedure, e.g. the modification factor, is stored.

    **Parameters**:

    ``scell``: SuperCell object
        Super cell in which the sampling is performed.

    ``filename``: string
        The trajectoy can be stored in a json file with the path given by ``filename``.

    ``**kwargs``: keyword arguments

        ``modification_factor``: float

        ``flatness_condition``: float

    """

    def __init__(self, scell = None, filename="cdos.json", **kwargs):
        self._cdos = []

        self._scell = scell
        self._save_nsteps = kwargs.pop("save_nsteps",10)
        self._write_no = 0

        self._models = kwargs.pop("models",[])

        self._filename = filename

        #Parameter for Wang Landau
        self._f = kwargs.pop("modification_factor",math.exp(1))
        self._flatness_condition = kwargs.pop("flatness_condition",0.50)


    def calculate_thermodynamics(self, quantity):
        """Calculate the thermodynamic for all decoration in the trajectory
        """
        pass
    
    def wang_landau_write_to_file(self, filename = None):
        """Write trajectory to file (default filename trajectory.json).
        """

        if filename is not None:
            self._filename = filename

        cdosdict={}
        #for j,dec in enumerate(self._trajectory):
        #    trajdic.update({str(j):dec})
        cdosdict.update({'modification_factor': self._f })
        cdosdict.update({'flatness_condiction': self._flatness_condition })
        cdosdict.update({'super_cell_definition': self._scell.as_dict()})
        cdosdict.update({"cdos": self._cdos})

        with open(self._filename, 'w+', encoding='utf-8') as outfile:
            json.dump(cdosdict, outfile, cls=NumpyEncoder, indent = 2 , separators = (',',':'))



class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types
    https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable/32850511

    """
    def default(self, obj):
        if isinstance(obj, list):
            return obj.tolist()
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)):
            return obj.tolist()

        return json.JSONEncoder.default(self, obj)
