# Copyright (c) 2015-2019, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

## packages needed for MonteCarlo
#from clusterx.structures_set import StructuresSet
from clusterx.structure import Structure

## packages needed for MonteCarloTrajectory
import json
import numpy as np
from copy import deepcopy

# needed for NumpyEncoder class at the end of this module
from ase.cell import Cell

class MonteCarlo():
    """Monte Carlo class

    **Description**:
        Objects of this class are used to perform Monte Carlo samplings.

        It is initialized with:

        - a Model object, that enables to calculate the energy of a structure, 

        - a SuperCell object, in which the sampling is performed, 
    
        - specification of the thermodynamic ensemble: 

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

    ``sublattice_indices``: list of integers (default = None)
        Defines the sublattices for the grand canonical sampling. 
        Furthermore, it can be used to limit the canonical sampling 
        to a reduced number of sublattices. E.g. in the case of nsubs = {0:[4,6], 1:[4]}. Here, sublattices 0 and 1 
        contain substitutional sites, but only a sampling in sublattice 0 is wanted. Then, put ``sublattice_indices`` = [0].

    ``chemical_potentials``: dictionary (default: None)
        Define the chemical potentials used for samplings in the grand canonical ensemble.
     
        The format of the dictionary is as follows

        {site_type1:[:math:`\Delta \mu_{11}`, :math:`\Delta \mu_{12}`,...], site_type2:[:math:`\Delta \mu_{21}`, :math:`\Delta \mu_{22}`,...],...}

        Here, :math:`\Delta \mu_{\#i}` is the chemical potential difference of substitutional species i relative 
        to the pristine species in sub-lattice with site_type#.

    ``models``: List of Model objects
        The properties returned from these Model objects are additionally 
        calculated during the sampling and stored with their corresponding ``prop_name`` 
        to the dictionary ``key_value_pairs`` for each visited structure during the sampling. 
        
        Properties from Model obejects can also be calculated after the sampling by using the MonteCarloTrajectory class.

    ``no_of_swaps``: integer (default: 1)
        Number of atom swaps/replacements per sampling step.

    ``predict_swap``: boolean (default: False)
       If set to **True**, this parameter makes the sampling faster by calculating the correlation difference of the 
       proposed structure with respect to the previous structure.

    ``error_reset``: integer (default: None)
       If not **None*  and ``predict_swap`` equal to **True**, the correlations are calculated as usual (no differences) every n-th step.

    .. todo:
        Samplings in the grand canonical ensemble are not yet possible.

    """

    def __init__(
            self,
            energy_model,
            scell,
            nsubs = None,
            ensemble = 'canonical',
            sublattice_indices = None,
            chemical_potentials = None,
            models = [],
            no_of_swaps = 1,
            predict_swap = False,
            error_reset = None,
            filename = "trajectory.json"
    ):
        self._em = energy_model
        self._scell = scell
        self._nsubs = nsubs
        self._filename = filename

        if sublattice_indices is None:
            try:
                self._sublattice_indices = [k for k in self._nsubs.keys()]
            
                if ensemble == 'canonical':
                    for key in self._nsubs.keys():
                        if all([ subs == 0 for subs in self._nsubs[key] ]):
                            self._sublattice_indices.remove(key)

            except AttributeError:
                raise AttributeError("Sublattice for the sampling is not properly assigned, look at the documentation.")
        else:
            self._sublattice_indices = sublattice_indices
                        
            if ensemble == 'canonical':
                for subind in self._sublattice_indices:
                    if subind not in self._nsubs.keys():                        
                        raise AttributeError("Sublattice for the sampling is not properly assigned, look at the documentation.")
                    
        if not self._sublattice_indices:
            import sys
            sys.exit('Sublattice for the sampling is not correctly assigned, look at the documatation.')

        self._models = []
        if models:
            self._models = models

        self._ensemble = ensemble
        self._no_of_swaps = no_of_swaps

        if self._no_of_swaps > 1:
            self._control_flag = False
        elif predict_swap == True:
            self._control_flag = True
        else:
            self._control_flag = False
            
        self._error_reset = error_reset

    def metropolis(
            self,
            no_of_sampling_steps = 100,
            scale_factor = [1.0],
            temperature = 1.0,
            boltzmann_constant = 1.0,
            initial_decoration = None,
            acceptance_ratio = None,
            serialize = False,
            filename = None,
            **kwargs
    ):
        """Perform Monte-Carlo Metropolis simulation

        **Description**: 
            Perfom Monte-Carlo Metropolis sampling for nmc sampling steps.

            During the sampling, a new structure at step i is accepted
            with the probability given by :math:`\min( 1, \exp( - (E_i - E_{i-1})/(k_B T)) )`

            The energy :math:`E_i` of visited structure at step i is calculated from the Model 
            ``energy_model``. The factor :math:`k_B T` is the product of the temperature :math:`T` 
            and the Boltzmann constant :math:`k_B` (also know as the thermal energy).

            Note: The units of the ``energy`` :math:`E` and the factor :math:`k_B T` must be the same. 
            With ``scale_factor``, :math:`k_B T` can be adjusted to the correct units (see below).

        **Parameters**:

        ``no_of_sampling_steps``: integer
            Number of sampling steps
        
        ``temperature``: float
            Temperature at which the sampling is performed.
        
        ``boltzmann_constant``: float
            Boltzmann constant 

        ``scale_factor``: list of floats
            List is used to adjust the factor :math:`k_B T` to the same units as the energy from ``energy_model``.
 
            All floats in list are multiply to the factor :math:`k_B T`.
            If list is empty, the factor :math:`k_B T` remains changed.
             
        ``initial_decoration``: list of integers
            Atomic numbers of the initial structure, from which the sampling starts.
            If ``None``, sampling starts with a structure randomly generated.

        ``acceptance_ratio``: float (default: None)
            Real number between 0 and 100. Represents the percentage of accepted moves.
            If not ``None``, the initial temperature will be adjusted to match the given
            acceptance ratio. The acceptance ratio during the simulation is computed using
            the last 100 moves.
        
        ``serialize``: boolean (default: False)
            Serialize the MonteCarloTrajectory object into a Json file after the sampling. 
        
        ``filename``: string (default: ``trajectory.json``)
            Name of a Json file in which the trajectory is serialized after the sampling if ``serialize`` is **True**.

        ``**kwargs``: keyworded argument list, arbitrary length
            These arguments are added to the MonteCarloTrajectory object that is initialized in this method.
        
        **Returns**: MonteCarloTrajectory object
            Trajectory containing the complete information of the sampling trajectory.

        """
        import math
        from clusterx.utils import poppush
        from tqdm import tqdm
        
        scale_factor_product = boltzmann_constant * temperature
        for el in scale_factor:
            scale_factor_product *= el

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
                        nl.append(int(round(cel*ns,0)))
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
                
        self._em.corrc.reset_mc(mc = True)
        e = self._em.predict(struc)


        if filename is not None:
            self._filename = filename
        else:
            filename = self._filename

        
        traj = MonteCarloTrajectory(
            self._scell,
            filename = filename,
            models = self._models,
            no_of_sampling_steps = no_of_sampling_steps,
            temperature = temperature,
            boltzmann_constant = boltzmann_constant,
            scale_factor = scale_factor,
            acceptance_ratio = acceptance_ratio,
            **kwargs)

        if self._models:
            key_value_pairs = {}
            for m, mo in enumerate(self._models):
                key_value_pairs.update({mo.property: mo.predict(struc)})
            traj.add_decoration(0, e, [], decoration = struc.decor, key_value_pairs = key_value_pairs)
        else:
            traj.add_decoration(0, e, [], decoration = struc.decor)

        if acceptance_ratio:
            nar = 100
            ar = acceptance_ratio
            hist = np.zeros(nar,dtype=int)

        if self._error_reset is not None:
            error_steps = int(self._error_reset)
            x = 1
            
        for i in tqdm(range(1,no_of_sampling_steps+1), total=no_of_sampling_steps, desc="MMC simulation"):
            indices_list = []

            for j in range(self._no_of_swaps):
                ind1, ind2, site_type, rindices = struc.swap_random(self._sublattice_indices)
                indices_list.append([ind1, ind2, [site_type, rindices]])

            if self._control_flag:
                if self._error_reset:
                    if (x > error_steps):
                        x = 1
                        e1 = self._em.predict(struc)
                    else:
                        x += 1
                        de = self._em.predict_swap(struc, ind1 = ind1 , ind2 = ind2, site_types = self._sublattice_indices)
                        e1 = e + de
                else:
                    de = self._em.predict_swap(struc, ind1 = ind1, ind2 = ind2, site_types = self._sublattice_indices)
                    e1 = e + de
                    
            else:
                e1 = self._em.predict(struc)

            if e >= e1:
                accept_swap = True
                boltzmann_factor = 0
            else:
                boltzmann_factor = math.exp((e-e1)/(scale_factor_product))

                if np.random.uniform(0,1) <= boltzmann_factor:
                    accept_swap = True
                else:
                    accept_swap = False

            if accept_swap:
                e = e1

                if self._models:
                    key_value_pairs = {}
                    for m, mo in enumerate(self._models):
                        key_value_pairs.update({mo.property: mo.predict(struc)})
                    traj.add_decoration(i, e, [[li[0],li[1]] for li in indices_list], key_value_pairs = key_value_pairs)

                else:
                    traj.add_decoration(i, e, [[li[0],li[1]] for li in indices_list])

                if acceptance_ratio:
                    ar = poppush(hist,1)

            else:
                for j in range(self._no_of_swaps-1,-1,-1):
                    struc.swap(indices_list[j][1],indices_list[j][0], site_type = indices_list[j][2][0], rindices = indices_list[j][2][1])

                if acceptance_ratio:
                    ar = poppush(hist,0)

            if acceptance_ratio:
                if i%10 == 0 and i >= nar:
                    scale_factor_product *= math.exp((acceptance_ratio/100.0-ar)/10.0)

        if serialize:
            traj.serialize()

        return traj



class MonteCarloTrajectory():
    """MonteCarloTrajectory class

    **Description**:
        Objects of this class are used to store and access information of the trajectory generated from  
        a Monte Carlo sampling performed with the MonteCarlo class. 
    
        It is initialized with the SuperCell object. 
        Alternative: If ``read`` is **True**, it is initalized from a Json file with name ``filename`` that 
        was created from a MonteCarloTrajectory object before by ``MonteCarloTrajectory.serialize()``.

    **Parameters**:

    ``scell``: SuperCell object (default: None)
        Super cell in which the sampling is performed.

    ``filename``: string (default: trajectory.json)
        The trajectoy can be stored in a json file with the path given by ``filename``.
    
    ``read``: boolean (default: False)
        If **True**, the trajectory is read from the Json file ``filename``.

    ``**kwargs``: keyword arguments

        ``models``: List of Model objects
         
        Further keyword arguments can be used to store additional information about the parameters used for 
        the MonteCarloTrajectory.metropolis rountine. This will be saved in the Json file ``filename`` 
        under ``sampling_info``, if the object is serialized.

    """

    def __init__(self, scell = None, filename="trajectory.json", read = False, **kwargs):
        # Load Boltzmann constant in eV / K from ASE
        from ase.units import kB

        self._filename = filename
        self._scell = scell
        
        if read:
            self.read()
            self._models = kwargs.pop('models',[])
            
        else:
            self._trajectory = []
            
            self._scell = scell
            self._save_nsteps = kwargs.pop('save_nsteps',10)
            self._write_no = 0
            
            
            self._models = kwargs.pop('models',[])
            
            self._nmc = kwargs.pop('no_of_sampling_steps',None)
            self._temperature = kwargs.pop('temperature',None)
            self._boltzmann_constant = kwargs.pop('boltzmann_constant',kB)
            self._scale_factor = kwargs.pop('scale_factor',None)
            self._acceptance_ratio = kwargs.pop('acceptance_ratio',None)
            self._keyword_arguments = kwargs

        
    def calculate_properties(self, models = [], prop_func = None, prop_name = None, **kwargs):
        """Calculate the property for all decorations in the trajectory. The property can be 
           calculated by a Model object or an external function ``prop_func``. 

        **Parameters**:
            ``models``: list (default: empty list)
        
            ``prop_func``: function (default: None)
                This function recieves as arguments the Structure object and the dictionary 
                of trajectory entry at step i, and additional keyword arguments given by ``**kwargs``.
        
            ``prop_name``: string (default: None)
                Name of property which is calculated by ``prop_func``.
            
            ``**kwargs``: keyworded argument list, arbitrary length
                Additional parameters for the function ``prop_func``   

        """
        for mo in models:
            if mo not in self._models:
                self._models.append(mo)

        predict_swap = kwargs.pop('predict_swap',False)
        print('predict swap ?', predict_swap)
        
        sx = Structure(self._scell, decoration = self._trajectory[0]['decoration'], mc = True)            
        for m,mo in enumerate(models):
            mo.corrc.reset_mc(mc = True)
        
        for t,tr in enumerate(self._trajectory):
            indices_list = tr['swapped_positions']
            for j in range(len(indices_list)):
                sx.swap(indices_list[j][0],indices_list[j][1])

            sdict={}
            if predict_swap == True:

                if t == 0:
                    movalue = np.zeros( len(models))
                    for m,mo in enumerate(models):
                        movalue[m] = mo.predict(sx)
                        sdict.update({mo.property: movalue[m]})
                else:
                
                    for m,mo in enumerate(models):
                        # only works for single swaps
                        dmo = mo.predict_swap(sx, ind1 = indices_list[0][0] , ind2 = indices_list[0][1], site_types = self._sublattice_indices)
                        movalue[m] = movalue[m]+dmo
                        sdict.update({mo.property: movalue[m]})

            else:
                for m,mo in enumerate(models):
                    sdict.update({mo.property: mo.predict(sx)})

            if prop_func is not None:
                sdict.update({prop_name: prop_func(sx,tr,**kwargs)})

            self._trajectory[t]['key_value_pairs'].update(sdict)
            
        
    def add_decoration(self, step, energy, indices_list, decoration = None, key_value_pairs = {}):
        """Add entry of the structure visited in the sampling to the trajectory.

        """
        if indices_list:
            self._trajectory.append(dict([('sampling_step_no', int(step)), ('energy', energy), ('swapped_positions', indices_list), ('key_value_pairs', key_value_pairs)]))
        else:
            self._trajectory.append(dict([('sampling_step_no', int(step)), ('energy', energy), ('swapped_positions', [[0,0]]) , ('decoration', deepcopy(decoration)), ('super_cell_definition', self._scell.as_dict()), ('key_value_pairs', key_value_pairs)]))


    def get_sampling_step_entry_at_step(self, nstep):
        """Return the entry, e.g. the dictionary stored for the n-th samplig step (``nstep``) in trajectory.

        """
        return self.get_sampling_step_entry(self.get_nid_sampling_step(nstep))
    

    def get_sampling_step_entry(self, nid):
        """Return the entry, e.g. the dictionary stored at index ``nid`` in trajectory.

        """
        return self._trajectory[nid]
    

    def get_structure_at_step(self, nstep):
        """Return structure in form of a Structure object, at the n-th sampling step (``nstep``) in trajectory.

        """
        return self.get_structure(self.get_nid_sampling_step(nstep))

    
    def get_structure(self, nid):
        """Return structure in form of a Structure object, at index ``nid`` in trajectory.

        """
        
        if nid < 0:
            _trajlength = len(self._trajectory)
            nid = int(_trajlength+nid)
            
        decoration = deepcopy(self._trajectory[0]['decoration'])
            
        for t,tr in enumerate(self._trajectory[0:nid+1]):
            indices_list = tr['swapped_positions']
            for j in range(len(indices_list)):
                idx1 = indices_list[j][0]
                idx2 = indices_list[j][1]
                decoration[idx1], decoration[idx2] = decoration[idx2], decoration[idx1]
                
        sx = Structure(self._scell, decoration = decoration)
        return sx
    

    def get_lowest_energy_structure(self):
        """Return structure in form of a Structure object with the lowest energy in trajectory.

        """
        _energies = self.get_energies()
        _emin = np.min(_energies)
        _nid = np.where(_energies == _emin)[0][0]

        return self.get_structure(_nid)

    
    def get_sampling_step_nos(self):
        """Return sampling step numbers of all entries in trajectory as array.

        """
        steps=[]
        for tr in self._trajectory:
            steps.append(tr['sampling_step_no'])

        return np.int_(steps)

    def get_sampling_step_no(self, nid):
        """Return sampling step number of entry at index ``nid`` in trajectory.

        """
        return self._trajectory[nid]['sampling_step_no']


    def get_nid_sampling_step(self, nstep):
        """Return entry index at the n-th sampling (``nstep``) step in trajectory.

        """
        steps = self.get_sampling_step_nos()
        nid = 0
        try:
            nid = np.where(steps == nstep)[0][0]
        except:
            if nstep > steps[-1]:
                nid = len(steps)-1
            else:
                for i,s in enumerate(steps):
                    if s > nstep:
                        nid = i-1
                        break
        return nid
    

    def get_energies(self):
        """Return energies of all entries in trajectory as array.

        """
        energies = []
        for tr in self._trajectory:
            energies.append(tr['energy'])

        return np.asarray(energies)


    def get_model_total_energies(self):
        
        return self.get_energies()
    

    def get_energy(self, nid):
        """Return energy of entry at index ``nid`` in trajectory.

        """
        return self._trajectory[nid]['energy']


    def get_model_total_energy(self, nid):
        
        self.get_energy(nid)
        

    def get_properties(self, prop):
        """Return the property ``prop`` from all entries in the trajectory as array.

        """
        if prop == 'energy':
            return self.get_energies()
        else:

            try:
                props = []
                for tr in self._trajectory:
                    props.append(tr['key_value_pairs'][prop])
                return np.asarray(props)

            except:
                if prop not in [mo.property for mo in self._models]:
                    print("Model of property is not given, look at the documentation.")
                else:
                    print("Property is not calculated, look at the documentation.")
                    

    def get_property(self, nid, prop):
        """Return property of entry at index ``nid`` in trajectory.

        """
        if prop == 'energy':
            return self.get_energy(nid)
        else:
            return self._trajectory[nid]['key_value_pairs'][prop]
        

    def calculate_average_property(self, prop_name = "U", no_of_equilibration_steps = 0, average_func = None, props_list = None, **kwargs):
        """Get averaged property of property with name ``prop_name`` after discarding at the start a given number of 
           equilibration steps. The average can only be obtained from a property that was already calculated before, 
           e.g. by MonteCarlo.Trajectory.calculate_properties(...). 

           Alternatively, an external function ``average_func`` can be used to calculate the average of one property 
           or several properties. The function ``average_func`` has as arguments an array containing the properties 
           used for the average at each sampling index, i.e. [[prob1_1,prop1_2,...,prop1_N], [prop2_1,prop2_2,...,prop2_N],...], 
           and optional keyword arguments. With ``prop_list``, the property names of prob1, prop_2, ... are defined.            
        
        **Parameters:**

        ``prop_name``: string (default: ``U``)
            Name of property that is averaged.
            If **C_p**, the isobaric specific heat at zero pressure is calculated.
            If **U**, the internal energy is calculated.
        
        ``no_of_equilibration_steps``: integer
            Number of equilibration steps at the start of the Monte-Carlo sampling that are discarded from the average.

        ``average_func``: function (default: ``None``)        
            If not **None**, the averaged property is obtained by this function. It gets as arguments an array and 
            optional keyword arguments. 

        ``**kwargs``: keyword arguments
           Additional arguments for ``average_func``.

        """
        step = 0
        prop_array = []
        prop_sum = 0

        if prop_name == "C_p" or prop_name == "U":
            prop_n = 'energy'
        else:
            prop_n = prop_name

        factor = 1.0
        if self._scale_factor is not None:
            for s in self._scale_factor:
                factor *= float(s)

        ind = 0
        maxind = len(self._trajectory)

        while step < self._nmc+1:
            if step < no_of_equilibration_steps:
                if ind < maxind:
                    if step == self._trajectory[ind]['sampling_step_no']:
                        if average_func is not None:
                            pv = []
                            for func_prop in props_list:
                                pv.append(float(self.get_property(ind,func_prop)))
                            prop = pv
                        else:
                            prop = float(self.get_property(ind,prop_n))
                          
                        ind += 1
                
            else:
                if ind < maxind:
                    if step == self._trajectory[ind]['sampling_step_no']:
                        if average_func is not None:
                            pv = []
                            for func_prop in props_list:
                                pv.append(float(self.get_property(ind,func_prop)))
                            prop = pv
                        else:
                            prop = float(self.get_property(ind,prop_n))
                            
                        ind += 1
                        
                prop_array.append(prop)
                if average_func is None:
                    prop_sum += prop
                                        
            step +=1

        if average_func is not None:
            prop_array = np.asarray(prop_array)            
            return average_func(prop_array.T, **kwargs)
        
        else:
            len_prop = len(prop_array)
            prop_average = np.divide(prop_sum,len_prop)
        
            if prop_name == "C_p":
                prop_array = np.divide(prop_array,factor)
                prop_average = np.divide(prop_average,factor)
                        
                ediff = 0
                for p in prop_array:
                    ediff += np.subtract(p,prop_average)*np.subtract(p,prop_average)

                const = 1.0*np.multiply((self._temperature)**2, (self._boltzmann_constant)**2)
            
                return np.divide(ediff, (const*len_prop*1/(1.0*factor)))
        
            else:
                return prop_average
        

    def get_nids(self, prop, value):
        """Return array of integers, that are the indices of the entries in trajectory for which the property 
           ``prop`` has the value ``value``.

        """
        arrayid = []

        if prop in ['sampling_step_no','swapped_positions','energy','decoration']:
            for i,tr in enumerate(self._trajectory):
                if tr[prop] == value:
                    arrayid.append(i)
        else:
            for i,tr in enumerate(self._trajectory):
                if tr['key_value_pairs'][prop] == value:
                    arrayid.append(i)

        return np.asarray(arrayid)

    def write_to_file(self, filename = None):
        
        self.serialize(filename = filename)
    
    def serialize(self, filename = None):
        """Write trajectory to Json file with name ``filename``. If ``filename`` is not defined, it uses 
           trajectory file name defined in the initialization of MonteCarloTrajectory object.

        """
        if filename is not None:
            self._filename = filename

        trajdic = {}
        traj_info = {}
        traj_info.update({'number_of_sampling_steps':self._nmc})
        traj_info.update({'temperature':self._temperature})
        traj_info.update({'boltzmann_constant':self._boltzmann_constant})
        if self._scale_factor is not None:
            traj_info.update({'scale_factor':self._scale_factor})
        if self._acceptance_ratio is not None:
            traj_info.update({'acceptance_ratio',self._acceptance_ratio})
            
        for key in self._keyword_arguments.keys():
            traj_info.update({key:self._keyword_arguments[key]})
                         
        trajdic.update({'sampling_info':traj_info})
        #add info about units of the energy, cluster expansion of the energy

        for j,dec in enumerate(self._trajectory):
            trajdic.update({str(j):dec})

        with open(self._filename, 'w+', encoding='utf-8') as outfile:
            json.dump(trajdic,outfile, cls=NumpyEncoder, indent = 2 , separators = (',',':'))

    def read(self, filename = None , append = False):
        """Read trajectory from the Json file with name ``filename``. If ``filename`` is not defined, it uses 
           trajectory file name defined in the initialization of the MonteCarloTrajectory.

        """
        if filename is not None:
            trajfile = open(filename,'r')
            self._filename = filename
        else:
            trajfile = open(self._filename,'r')

        data = json.load(trajfile)

        if not append:
            self._trajectory = []

        traj_info = data.pop('sampling_info',None)
        
        if traj_info is not None:
            self._nmc = traj_info.pop('number_of_sampling_steps',None)
            self._temperature = traj_info.pop('temperature',None)
            self._boltzmann_constant = traj_info.pop('boltzmann_constant',None)
            self._scale_factor = traj_info.pop('scale_factor',None)
            self._acceptance_ratio = traj_info.pop('acceptance_ratio',None)
            self._keyword_arguments = traj_info
            
        data_keys = sorted([int(el) for el in set(data.keys())])

        if 'model_total_energy' in data['0']:
            exchange = True
        else:
            exchange = False
            
        for key in data_keys:
            tr = data[str(key)]
            if exchange:
                tr['energy']=tr.pop('model_total_energy')
            self._trajectory.append(tr)

        if self._scell is None:
            from ase.atoms import Atoms
            from clusterx.parent_lattice import ParentLattice
            from clusterx.super_cell import SuperCell
            _trajz = self._trajectory[0]

            nsp = sorted([int(el) for el in set(_trajz['super_cell_definition']['parent_lattice']['numbers'])])
            species = []
            for n in nsp:
                species.append(_trajz['super_cell_definition']['parent_lattice']['numbers'][str(n)])

            _plat = ParentLattice(atoms = Atoms(positions = _trajz['super_cell_definition']['parent_lattice']['positions'], cell = _trajz['super_cell_definition']['parent_lattice']['unit_cell'], numbers=np.zeros(len(species)), pbc = np.asarray(_trajz['super_cell_definition']['parent_lattice']['pbc'])), sites  = np.asarray(species), pbc = np.asarray(_trajz['super_cell_definition']['parent_lattice']['pbc']))
            self._scell = SuperCell(_plat, np.asarray(_trajz['super_cell_definition']['tmat']))


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
        elif isinstance(obj,Cell):
            return obj.array


        return json.JSONEncoder.default(self, obj)
