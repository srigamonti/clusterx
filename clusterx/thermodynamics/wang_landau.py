# Copyright (c) 2015-2019, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

from clusterx.structure import Structure
import json
import math
import numpy as np
import scipy
from copy import deepcopy
import time
import datetime

def microcanonical_temperature(e, ln_g):
    """Compute microcanonical temperature

    The micrononical temperature is given by

    .. math:

        T = \frac{1}{k_B} \frac{1}{\frac{d ln(g(E))}{dE}}

    **Parameters**:

    ``e``: array of float
        energies in eV.

    ``ln_g``: array of float
        natural logarithm of the configurational density of states. 
    """
    from ase.units import kB as kb

    grad = np.gradient(ln_g,e)
    temp = map( lambda x: 1/(kb*x), grad)

    return list(temp)

def cdos_interpolation(
    energy, 
    log_cdos, show_plot=False, 
    plot_temperature=None, 
    plot_microcanonical_temperature=True, 
    **splrep_args
    ):
    """Perform interpolation of CDOS using splines
    
    Essentially wraps ``scipy.interpolate.splrep``.

    **Parameters**:

    ``energy``: array of float
        energies in eV

    ``log_cdos``: array of float
        natural logarithm of the configurational density of states

    ``show_plot``: boolean
        whether to plot the interpolated CDOS
    """
    from scipy import interpolate
    from ase.units import kB as kb

    e = energy
    log_g = log_cdos    
    log_g -= log_g[0] # Assume normalization g(E_0=0) = 1

    tck = interpolate.splrep(e, log_g, **splrep_args)
    e_itpl = np.arange(e[0],e[-1],(e[1]-e[0])/np.pi)
    log_g_itpl = interpolate.splev(e_itpl, tck)
    log_g_itpl -= log_g_itpl[0]
    
    if show_plot:
        import matplotlib.pyplot as plt

        if plot_temperature is None:
            plt.plot(e_itpl, log_g_itpl, label="CDOS(itpl)")
            plt.scatter(e, log_g, label="CDOS")
            if plot_microcanonical_temperature:
                temp = microcanonical_temperature(e, log_g)
                temp_itpl = microcanonical_temperature(e_itpl, log_g_itpl)
                plt.plot(e_itpl, temp_itpl, label="T(itpl)")
                plt.scatter(e, temp, label="T")
                #plt.ylim(-100,2000)
        else:
            plt.plot(e_itpl, log_g_itpl-e_itpl/(kb*plot_temperature), label="CDOS(itpl)")
            plt.scatter(e,log_g-e/(kb*plot_temperature), label="CDOS")

        plt.legend()
        plt.show()
        
    return e_itpl, log_g_itpl
        
def compute_thermodynamic_averages(temperatures, energy, log_cdos, filename=None):
    """Compute thermodynamic averages in the canonical ensemble as a function of temperature. 

    In disregard of the normalization used for the configurational density of states (CDOS) :math:`g(E)`,
    for numerical convenience this method internally normalizes :math:`g(E)` according to

    .. math::

        g(E_0=0) = 1

    For the output quantities other normalizations can be easily obtained considering the following 
    transformations:

    .. math::

        & g(E) \\rightarrow c g(E)

        & Z(T) \\rightarrow c Z(T)

        & S(T) \\rightarrow S(T) + k_B ln(c)

        & F(T) \\rightarrow F(T) - k_B T ln(c)

    **Parameters**:

    ``temperatures``: list of floats 
        Temperatures for which the thermodynamic property is calculated, in Kelvin.
    
    ``energy``: array of floats
        energies for which the CDOS is given. 

    ``log_cdos``: array of floats
        Natural logarithm of the CDOS for the given energies in ``energy`` array.

    """
    from ase.units import kB as kb
    import sys
    ln_maxfloat = math.log(sys.float_info.max)
    e = energy
    log_g = log_cdos
    
    thavg = np.zeros((4, len(temperatures)))

    for i, t_i in enumerate(temperatures):
        beta_i = 1.0/(kb*t_i)

        p = np.zeros(len(e))
        u = 0
        u2 = 0
        z = 0

        # Compute canonical probability as
        # P(E) = 1 / \sum_Ep exp(ln(Ep)-ln(E)-\beta * (Ep-E))
        for j,log_g_j in enumerate(log_g):
            pinv_j = 0
            for k,log_g_k in enumerate(log_g):
                exponent = log_g_k-log_g_j-beta_i*(e[k]-e[j])
                if exponent > ln_maxfloat or pinv_j == math.inf:
                    pinv_j = 0
                    break
                else:
                    pinv_j += math.exp(exponent)

            if pinv_j != 0:
                p[j] = 1/pinv_j

        # Here we use: ln(Z) = ln(g(E)) - \beta E - ln(P(E,T))
        j_max = np.argmax(p)
        ln_z = log_g[j_max] - beta_i * e[j_max] - math.log(p[j_max])

        f = - kb * t_i * ln_z

        for j,log_g_j in enumerate(log_g):
            u += e[j] * p[j]
            u2 += e[j]*e[j] * p[j]

        thavg[0, i] = u # Internal energy
        thavg[1, i] = ( u2 - u * u ) / ( kb * t_i * t_i ) # Specific heat
        thavg[2, i] = f # Helmholtz (or Gibbs at p=0) free energy
        thavg[3, i] = (u-f)/t_i # Entropy

    if filename is not None:
        np.savez(
            filename,
            temperature=temperatures,
            internal_energy=thavg[0],
            specific_heat=thavg[1],
            free_energy=thavg[2],
            entropy=thavg[3],
        )
        np.savetxt(
            filename+".txt",
            np.vstack((
                temperatures,
                thavg[0],
                thavg[1],
                thavg[2],
                thavg[3]
            )).T
        )
        
    return thavg
    
def make_energy_windows(inverse_overlap, n_windows, emin, emax, sought_energy_bin_width):
    """Make energy windows for WL sampling

    **Description**
        Determines boundaries of energy windows in an energy range for WL sampling.

    **Parameters**

    ``inverse_overlap``: integer
        Inverse of the fraction of overlap of two contiguous windows. 
        For instance, if ``inverse_overlap`` is 3, then 1/3rd of 
        an energy window overlaps with 1/3rd of the contiguous window.

    ``n_windows``: integer
        Number of windows.

    ``emin``: float
        Minimum energy

    ``emax``: float
        Maximum energy
    
    ``sought_energy_bin_width``: float
        Desired energy bin width. The actual energy bin width will be a bit higher, 
        to evenly fill the energy windows.

    **Returns**
        
        The function returns a dictionary with the following structure::

            {
                "inverse_overlap": float,
                "n_windows": int,
                "emin": float
                "emax": float
                "sought_energy_bin_width": float
                "energy_bin_width": float
                "windows":
                    [
                       [emin1, emax1],
                       [emin2, emax2],
                             ...
                       [eminn, emaxn]
                    ],
                "nbins_per_window_f",
                "nbins_per_window_i"
            }

    """
    n_delta = int( inverse_overlap + (n_windows - 1) * (inverse_overlap - 1) )
    delta = (emax - emin) / n_delta

    energy_bin_width = delta / np.floor(delta/sought_energy_bin_width)

    energy_windows = {}
    energy_windows["inverse_overlap"] = inverse_overlap
    energy_windows["n_windows"] = n_windows
    energy_windows["emin"] = emin
    energy_windows["emax"] = emax
    energy_windows["sought_energy_bin_width"] = sought_energy_bin_width
    energy_windows["energy_bin_width"] = energy_bin_width

    wins = np.zeros((n_windows,2))
    for i in range(n_windows):
        wins[i,0] = emin + (inverse_overlap - 1) * delta * i
        wins[i,1] = wins[i,0] + inverse_overlap * delta

    energy_windows["windows"] = wins
    
    energy_windows["nbins_per_window_f"] = (wins[0,1] - wins[0,0])/energy_bin_width
    energy_windows["nbins_per_window_i"] = round((wins[0,1] - wins[0,0])/energy_bin_width)

    return energy_windows

def merge_windows(filepaths = [], wliteration = -1, e_factor = 1, show_plot=False, plot_microcanonical_temperature=False, plot_temperature=None, filename=None):
    """Merge CDOSs from a parallel WL run into a single CDOS for postprocessing
    """
    eps_energy = 1e-3
    cdoss = []

    for filepath in filepaths:
        with open(filepath,'r') as f:
            data = json.load(f)
        cdoss.append(data)

    n_cdos = len(cdoss)

    energy = []
    log_g = []
    energy_window = []
    energy_windows = []
    log_g_window = []
    log_g_windows = []
    histogram = []
    histogram_window = []
    histogram_norm = []
    microcanonical_temp = []
    microcanonical_temps = []
    energy_ranges = []

    wl_itrn = []
    for i in range(n_cdos):        
        wl_itrn.append(str(wliteration))
        if wliteration == -1:
            wl_itrn[i] = str(len(cdoss[i])-2)

    for i in range(n_cdos):
        log_g_window.append(cdoss[i][wl_itrn[i]]["cdos"])
        energy_window.append(cdoss[i]["sampling_info"]["energy_bins"])
    
    matching_deltas = []
    matching_deltas.append(cdoss[0][wl_itrn[0]]["cdos"][0])
    for i in range(1, n_cdos):
        deltas = []
        for j1, e1 in enumerate(cdoss[i-1]["sampling_info"]["energy_bins"]):
            for  j2, e2 in enumerate(cdoss[i]["sampling_info"]["energy_bins"]):
                if np.abs(e1-e2) < eps_energy:
                    deltas.append(cdoss[i][wl_itrn[i]]["cdos"][j2]-cdoss[i-1][wl_itrn[i-1]]["cdos"][j1])

        if i == 0:
            matching_deltas.append(np.mean(deltas))
        else:
            matching_deltas.append(np.mean(deltas)+matching_deltas[i-1])

    for i in range(n_cdos):

        histogram_window.append(cdoss[i][wl_itrn[i]]["histogram"])
        hsum = np.sum(cdoss[i][wl_itrn[i]]["histogram"])
        nbins = len(cdoss[i][wl_itrn[i]]["histogram"])
        microcanonical_temp.append(
            microcanonical_temperature(
                cdoss[i]["sampling_info"]["energy_bins"],
                cdoss[i][wl_itrn[i]]["cdos"]
                )
            )

        energy_ranges.append(cdoss[i]["sampling_info"]["energy_bins"])


        for e, g, h, t in zip(
            cdoss[i]["sampling_info"]["energy_bins"], 
            cdoss[i][wl_itrn[i]]["cdos"], 
            cdoss[i][wl_itrn[i]]["histogram"],
            microcanonical_temp[i]
            ):
            energy.append(e)
            log_g.append(g-matching_deltas[i])
            histogram.append(h)
            histogram_norm.append(h/hsum * nbins)
            microcanonical_temps.append(t)

    energy_unique = np.unique(energy)

    log_g_unique = np.zeros(len(energy_unique))
    for i, e in enumerate(energy_unique):
        g_avg = 0
        count = 0
        for e2, g in zip(energy, log_g):
            if np.abs(e-e2) < eps_energy:
                g_avg += g
                count += 1
        if count != 0:
            log_g_unique[i] = g_avg / count

    energy_unique *= e_factor

    from ase.units import kB as kb
    log_boltzmann = np.zeros(len(energy))
    log_boltzmann_unique = np.zeros(len(energy_unique))
    if plot_temperature is not None:
        log_boltzmann = np.array(energy)/(kb*plot_temperature)
        log_boltzmann_unique = np.array(energy_unique)/(kb*plot_temperature)

    if filename is not None:
        from ase.units import kB as kb

        for i in range(n_cdos):
            for e, g in zip(cdoss[i]["sampling_info"]["energy_bins"], cdoss[i][wl_itrn[i]]["cdos"]):
                energy_windows.append(e)
                log_g_windows.append(g)

        np.savez(
            filename,
            temperature_for_log_canonical_probability=plot_temperature,
            energy=energy_unique,
            log_cdos=log_g_unique,
            log_canonical_probability=log_g_unique - log_boltzmann_unique,
            energy_windows=energy_windows,
            log_cdos_windows=log_g_windows, 
            histogram=histogram, 
            histogram_norm=histogram_norm,
            microcanonical_temperature=microcanonical_temps,
            energy_window = np.array(energy_window, dtype=object),
            log_cdos_window = np.array(log_g_window, dtype=object),
            histogram_window = np.array(histogram_window, dtype=object),
            microcanonical_temperatures=np.array(microcanonical_temp, dtype=object),
            energy_ranges = np.array(energy_ranges, dtype=object),
            allow_pickle=True
        )

        np.savetxt(
            filename+".txt",
            np.vstack((
                energy_unique,
                log_g_unique,
                energy_windows,
                log_g_windows,
                histogram,
                histogram_norm,
                microcanonical_temps    
            )).T
        )

    if show_plot:
        import matplotlib.pyplot as plt

        plt.scatter(np.array(energy) * e_factor, np.array(log_g) - log_boltzmann, label='All data')
        plt.scatter(energy_unique, log_g_unique - log_boltzmann_unique, label='Merged data')

        if plot_microcanonical_temperature:
            temp = microcanonical_temperature(energy_unique, log_g_unique)
            plt.scatter(energy_unique, temp)

        plt.show()

    return energy_unique, log_g_unique
        
class WangLandau():
    """Wang Landau class

    **Description**:
        Objects of this class are used to obtain the configurational density of states 
        by performing samplings using the Wang Landau method. For details about the method, 
        see F. Wang and D.P. Landau, PRL 86, 2050 (2001). 

        It is initialized with:
        
        - a Model object, that enables to calculate the energy of a structure,
    
        - a SuperCell object, in which the sampling is performed,
    
        - specification of the thermodynamic ensemble:

            If ``ensemble`` is 'canonical', the composition for the sampling is defined with ``nsubs``. In case               
            of multilattices, the sublattice for the sampling can be defined with ``sublattice_indices``.

            If ``ensemble`` is 'gandcanonical', the sublattice is defined with ``sublattices_indices``. 

    **Parameters**:

    ``energy_model``: :class:`Model <clusterx.model.Model>` object
        CE model for predicting the internal energy of the supercell (see ``scell`` below)
        in [eV]. If other units are employed, see parameter ``energy_factor`` below.

    ``energy_factor``: float (default: 1.0)
        Multiplicator factor (:math:`s`) applied to the predictions of ``energy_model`` 
        (:math:`E_{CE}`). Set this factor such that :math:`s *  E_{CE}` gives
        the total energy, in [eV], of the supercell (``scell`` below).
        For instance, if the CE model predicts the energy in meV/atom, set this parameter 
        to 1000*Nsc, where Nsc is the total number of atoms of the supercel (``len(scell)``). 
        If CE predicts the energy in Hartree
        per parent lattice, set ``energy_factor`` to 27.2114*Npl, where Npl is 
        the number of parent lattices contained in the supercell (this can be obtained 
        with :meth:`scell.get_index() <clusterx.super_cell.SuperCell.get_index()>`). 

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

    def __init__(
            self,
            energy_model,
            energy_factor=1.0,
            scell = None,
            nsubs = None,
            ensemble = "canonical",
            sublattice_indices = [],
            chemical_potentials = None,
            predict_swap = False,
            error_reset = None
    ):
        self._em = energy_model
        self._ef = energy_factor
        self._scell = scell
        self._nsubs = nsubs

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

        if not self._sublattice_indices:
            import sys
            sys.exit('Indices of sublattice are not correctly assigned, look at the documatation.')
        self._predict_swap = predict_swap

        self._ensemble = ensemble
        self._chemical_potentials = chemical_potentials
        self._error_reset = error_reset
        self._n_mc_steps_total = 0
        self._start_time = time.time()
        self._elapsed_time = 0
        
        if self._error_reset is not None:
            self._error_steps = int(self._error_reset)
            self._x = 1


    def _wls_create_initial_structure(
        self, 
        initial_decoration, 
        emin, 
        emax, 
        prob_dist = "gaussian",
        trans_prob = 1e-3, 
        itmax = int(1e8),
        nitsampling = 1,
        nproc = 0
    ):
        import sys
        emean = None
        scale = None
        if prob_dist == "gaussian":
            from scipy.stats import norm
            
            emean = (emax+emin)/2.0
            scale = emax - emin

        if initial_decoration is not None:
            struc = Structure(self._scell, initial_decoration, mc = True)
        else:
            if self._nsubs is not None:
                struc = self._scell.gen_random_structure(self._nsubs, mc = True)
            else:
                struc = self._scell.gen_random_structure(mc = True)

        e = self._em.predict(struc) * self._ef

        if e >= emin and e <= emax and nitsampling == 1:
            return struc
        else:
            cou = 0
            scou = 0
            countsampling = False
            f = open(f"init_str_search_nproc_{nproc}.txt", "w")
            while cou < itmax and scou < nitsampling:
                f.write(f"{nproc}\t{cou}\t{emin}\t{emax}\t{e}\n")
                cou += 1
                if countsampling:
                    scou += 1
                    
                ind1, ind2, site_type, rindices = struc.swap_random(self._sublattice_indices)
                de = self._em.predict_swap(struc, ind1 = ind1 , ind2 = ind2, site_types = self._sublattice_indices) * self._ef
                e1 = e + de
                if e1 >= emin and e1 <= emax:
                    countsampling = True
                    accept_swap = True
                else:
                    if (e1 > emax and de < 0) or (e1 < emin and de > 0):
                        accept_swap = True
                    else:
                        if prob_dist == "gaussian":
                            trans_prob = norm.pdf(e1, emean, scale)

                        accept_swap = np.random.uniform(0,1) <= trans_prob

                if not cou % 1000:
                    f.flush()
                    print(f"searching struc {cou}, {emin:2.9f} {e:2.9f} {de} {ind1:d} {ind2:d} {emax:2.9f}")
                    
                if accept_swap:
                    e = e1
                else:
                    struc.swap(ind2, ind1, site_type = site_type, rindices = rindices)
            f.close()
            if cou >= itmax:
                print("WangLandau: maximum number of iterations for searching initial structure reached. Aborting simulation.")
                return struc
            return struc


    def _wls_init_from_file(self, cd, update_method, flatness_conditions, f_range):
        cdos = []
        for l,eb in enumerate(cd._energy_bins):
            cdos.append([eb,cd._cdos[l],0])
        cdos = np.asarray(cdos)

        # Update f
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

        return cdos, f, histogram_flatness, f_range, flatness_conditions, fi
    

    def _wls_init_from_scratch(self, energy_range, energy_bin_width, flatness_conditions, f_range):
        cdos=[]
        eb = energy_range[0]
        while eb < energy_range[1]:
            cdos.append([eb,0.0,0]) # log(cdos=1.0)=0.0
            eb += energy_bin_width
        cdos = np.asarray(cdos)

        f = f_range[0]
        fi = 0
        histogram_flatness = flatness_conditions[fi][0]

        return cdos, f, histogram_flatness, fi
    

    def _wls_locate_histogram_bin(self, e, energies, energy_bin_width):
        emin = energies[0]
        emax = energies[-1] + energy_bin_width

        is_outside_interval = False
        
        if e < emin:
            return  None, True
        elif e >= emax:
            return  None, True
        else:
            for i in range(len(energies)):
                if e >= energies[i] and e < energies[i] + energy_bin_width:
                    return  i, False

    def _wls_update_modification_factor(self, f, update_method):
        if update_method == 'square_root':
            f = math.sqrt(f)
        else:
            import sys
            sys.exit('Different update method for f requested. Please see documentation')
        return f

    
    def _plot_hist(self, figure, ax, cdos, energy_bin_width):
        from clusterx.visualization import _wls_normalize_histogram_for_plotting

        ener_arr = cdos[:,0].copy()
        cdos_arr = _wls_normalize_histogram_for_plotting(cdos[:,1], shift_y_first_nonzero=True)
        hist_arr = _wls_normalize_histogram_for_plotting(cdos[:,2])
        ones_arr = np.ones(len(ener_arr))

        ax.clear()
        ax.bar(ener_arr, ones_arr, width=energy_bin_width*0.92, color="silver")
        ax.bar(ener_arr, hist_arr, width=energy_bin_width*0.82)
        ax.bar(ener_arr, cdos_arr, width=energy_bin_width*0.35)
        ax.relim()
        figure.canvas.draw()
        figure.canvas.flush_events()

        
    def wang_landau_sampling(
            self,
            energy_range=[-2,2],
            energy_bin_width=0.2,
            f_range=[math.exp(1), math.exp(1e-4)],
            update_method='square_root',
            flatness_conditions=[
                [0.5,math.exp(1e-1)],
                [0.80,math.exp(1e-3)],
                [0.90,math.exp(1e-5)],
                [0.95,math.exp(1e-7)],
                [0.98,math.exp(1e-8)]
            ],
            initial_decoration = None,
            serialize = False,
            filename = "cdos.json",
            serialize_during_sampling = False,
            restart_from_file = False,
            plot_hist_real_time = False,
            acc_prob_init_structure = 1e-3,
            acc_prob_dist_init_structure = "gaussian",
            itmax_init_structure=int(1e8),
            nproc=0,
            **kwargs
    ):
        """Perform Wang Landau simulation

        **Description**: 
            The Wang-Landau algorithm uses the fact that a Markov chain with transition probability 
            :math:`P_{1 -> 2} = \min[ 1, g(E_1)/g(E_2) ]`, with 
            :math:`g(E)`  being the configurational density of states, yields a flat histogram in energy 
            [see F. Wang and D.P. Landau, PRL 86, 2050 (2001)]. This fact is employed to 
            iteratively modify :math:`g(E)`, which is unknown and initially set to 1, in order to achieve 
            a flat histogram of the visited energies. 
            The accuracy of the found :math:`g(E)` will depend on how flat is the histogram at the 
            end of the run and other parameters, as explained below.

            The energy of a visited structure is calcualted with the CE model ``energy_model``. 

            During the sampling, a new structure at step i is accepted with the probability given 
            by :math:`\min[ 1, g(E_{i-1})/g(E_i) ]`

            If a step is accepted, :math:`g(E_i)` of energy bin :math:`E_i` is updated with a modification factor :math:`f`.
            If a step is rejected, :math:`g(E_{i-1})` of the previous energy bin :math:`E_{i-1}` with a modification factor :math:`f`.

            The sampling procedure is a nested loop:
       
            - Inner loop: Generation of a flat histogram in energy for a fixed :math:`f`.

            - Outer loop: Gradual reduction of :math:`f` to increase the accuracy of :math:`g(E)`.

            The initial modification factor is usually :math:`f=\exp(1)`. Since it is large, it ensures to reach all energy levels quickly. 
            In the standard procedure, the modification factor is reduced from :math:`f` to  :math:`\sqrt{f} ` for the next inner loop. 
            :math:`f` is a measure of the accuracy for :math:`g(E)`: The lower the value of :math:`f`, the higher the accuracy. The 
            sampling stops, if the next :math:`f` is below the threshold :math:`f_{final}`.

            A histogram is considered as flat, 
            if the lowest number of counts of all energy bins is above a given percentage of the mean value.
            The percentage can be set for each :math:`f`. Usually, the lower :math:`f`, the larger the percentage can be set. 

        **Parameters**:

        ``energy_range``: list of two floats [E_min, E_max] (default: [-2,2])
            Defines the energy range, in [eV], starting from energy E_min (center of first energy bin) 
            until energy E_max.

        ``energy_bin_width``: float (default: 0.2)
            Bin width w of each energy bin, in [eV]. 
            
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

        ``acc_prob_init_structure``: float (default: 1e-3)
            When searching for an initial structure inside a given energy window, this is the acceptance probability
            for moves which whose energy difference with the upper window bound increases or for moves whose energy
            difference with the lower window bound decreases.

        ``itmax_init_structure``: integer
            Maximum number of trials to search for initial structure inside given energy window.

        ``**kwargs``: keyworded argument list, arbitrary length
            These arguments are added to the ConfigurationalDensityOfStates object that is initialized in this method.
        
        **Returns**: ConfigurationalDensityOfStates object
            Object contains the configurational density of states (CDOS) obtained from the last outer loop plus the CDOSs obtained 
            from the previous outer loops.
        
        """
        import math
        from clusterx.utils import poppush
        import sys
        
        self._em.corrc.reset_mc(mc = True)
        
        struc = self._wls_create_initial_structure(
            initial_decoration, 
            energy_range[0], 
            energy_range[1], 
            trans_prob = acc_prob_init_structure,
            prob_dist = acc_prob_dist_init_structure,
            itmax = itmax_init_structure,
            nproc = nproc
        )
        
        if restart_from_file:
            cd = ConfigurationalDensityOfStates(
                filename = filename,
                scell = self._scell,
                read = True,
                **kwargs
            )
            cdos, f, histogram_flatness, f_range, flatness_conditions, fi = self._wls_init_from_file(cd, update_method, flatness_conditions)
            
        else:
            cd = ConfigurationalDensityOfStates(
                filename = filename,
                scell = self._scell,
                energy_range = energy_range,
                energy_bin_width = energy_bin_width,
                f_range = f_range,
                update_method = update_method,
                flatness_conditions = flatness_conditions,
                nsubs = self._nsubs,
                ensemble = self._ensemble,
                sublattice_indices = self._sublattice_indices,
                chemical_potentials = self._chemical_potentials,
                **kwargs
            )
            cdos, f, histogram_flatness, fi = self._wls_init_from_scratch(energy_range, energy_bin_width, flatness_conditions, f_range)

        e = self._em.predict(struc) * self._ef

        energies = cdos[:,0]
        ibin, is_inside_range = self._wls_locate_histogram_bin(e, energies, energy_bin_width)

        cdos[ibin][1] += math.log(f)
        cdos[ibin][2] += 1
        g = cdos[ibin][1]
        

        if plot_hist_real_time:
            import matplotlib.pyplot as plt
            plt.ion()
            figure, ax = plt.subplots(figsize=(10, 8))

        while f > f_range[1]:
            print("----------------------------------------")
            print("Info (Wang-Landau): Running WL sampling.")
            print(f"Info (Wang-Landau): Modification factor: {f}")
            print(f"Info (Wang-Landau): Histogram flatness: {histogram_flatness}")
            
            struc, e, g, ibin, cdos, hist_cond, niter = self.flat_histogram(struc, e, g, ibin, f, cdos, histogram_flatness, energy_bin_width)
            
            print(f"Info (Wang-Landau): Number of MC steps: {niter}")
            
            self._n_mc_steps_total += niter
            cd.store_cdos(cdos, f, histogram_flatness, hist_cond, niter, self._n_mc_steps_total, self._start_time)

            if plot_hist_real_time:
                self._plot_hist(figure, ax, cdos, energy_bin_width)

            if serialize_during_sampling:
                cd.serialize()

            f = self._wls_update_modification_factor(f, update_method)
                        
            if f < flatness_conditions[fi][1]:
                fi += 1
                if fi < len(flatness_conditions):
                    histogram_flatness = float(flatness_conditions[fi][0])
                else:
                    histogram_flatness = float(0.99)
                    

        if serialize:
            cd.serialize()
                    
        return cd

    def _wls_get_hist_min_and_avg(self, hist):
        hist_sum = 0
        count_nonzero_bins = 0
        for h in hist:
            if float(h)>0:
                count_nonzero_bins += 1
                hist_sum += h
                if count_nonzero_bins == 1:
                    hist_min = h
                elif h < hist_min:
                    hist_min = h
        n_nonzero_bins = count_nonzero_bins
        hist_avg = hist_sum/(1.0*n_nonzero_bins)

        return hist_min, hist_avg, n_nonzero_bins

    def flat_histogram(self, struc, e, g, inde, f, cdos, histogram_flatness, energy_bin_width):
        hist_min = 0
        hist_avg = 1
        n_nonzero_bins = 0
        lnf = math.log(f) # Natural logarithm of f

        cdos[:,2] = 0 # Initialize histogram

        niter = 0
        niter_per_sweep = 100000
        nonzero_bins_thresh = 5

        print(f"Building flat histogram.")
        print(f" {'Mod. factor':12s} | {'MIN':8s} | {'AVG':10s} | {'Flatness':8s} | {'Tgt. Flat.':11s} |  {'No. of Bins':12s} | {'N iter.':15s} |  {'emin':11s} |  {'emax':11s} ")
        while (hist_min < histogram_flatness*hist_avg) or (n_nonzero_bins < nonzero_bins_thresh):
            print(f" {f:12.9f} | {int(hist_min):8d} | {hist_avg:10.2f} | {hist_min/hist_avg:8.3f} | {histogram_flatness:11.3f} |  {n_nonzero_bins:12d} | {niter:15d} | {cdos[0,0]:11.3f} | {cdos[-1,0]:11.3f}")
            
            struc, e, g, inde, cdos = self.dos_steps(struc, e, g, inde, lnf, cdos, niter_per_sweep, energy_bin_width)
            niter += niter_per_sweep
            
            hist_min, hist_avg, n_nonzero_bins = self._wls_get_hist_min_and_avg(cdos[:,2])
            if n_nonzero_bins < 10:
                print(f"{inde:4d} | {cdos[0,0]:11.3f} | {e:11.3f} | {cdos[-1,0]:11.3f}")

        print(f" {f:12.9f} | {int(hist_min):8d} | {hist_avg:10.2f} | {hist_min/hist_avg:8.3f} | {histogram_flatness:11.3f} |  {n_nonzero_bins:12d} | {niter:15d}")
        
        return struc, e, g, inde, cdos, [hist_min, hist_avg], niter

    
    def dos_steps(self, struc, e, g, inde, lnf, cdos, niter, energy_bin_width):
        energies = cdos[:,0]
        accept_swap = False
        for n in range(niter):
            ind1, ind2, site_type, rindices = struc.swap_random(self._sublattice_indices)

            # Compute new energy
            e1 = None
            if self._predict_swap:
                if self._error_reset:
                    if self._x > self._error_steps:
                        self._x = 1
                        e1 = self._em.predict(struc) * self._ef
                    else:
                        self._x += 1
                        de = self._em.predict_swap(struc, ind1 = ind1 , ind2 = ind2, site_types = self._sublattice_indices) * self._ef
                        e1 = e + de
                else:
                    de = self._em.predict_swap(struc, ind1 = ind1, ind2 = ind2, site_types = self._sublattice_indices) * self._ef
                    e1 = e + de
            else:
                e1 = self._em.predict(struc) * self._ef

            ibin, is_outside_interval = self._wls_locate_histogram_bin(e1, energies, energy_bin_width)


            if is_outside_interval:
                accept_swap = False
            else:
                g1 = cdos[ibin][1]

                if g >= g1:
                    accept_swap = True
                else:
                    trans_prob = math.exp(g-g1)
                    if np.random.uniform(0,1) <= trans_prob:
                        accept_swap = True
                    else:
                        accept_swap = False

            if accept_swap:
                e = e1
                inde = ibin
            else:
                struc.swap(ind2, ind1, site_type = site_type, rindices = rindices)

            cdos[inde][1] += lnf
            cdos[inde][2] += 1
            g = cdos[inde][1]
            
        return struc, e, g, inde, cdos

    def compare_flatness_conditions(
            self,
            flatness1,
            flatness2 = [
                [0.5, math.exp(1e-1)],
                [0.80, math.exp(1e-3)],
                [0.90, math.exp(1e-5)],
                [0.95, math.exp(1e-7)],
                [0.98, math.exp(1e-8)]
            ]
    ):
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

    def store_cdos(self, cdos, f, flatness_condition, hist_cond, n_mc_steps = 0, n_mc_steps_total = 0, start_time = 0):
        """Add entry of configurational of states that is obtained after a flat histgram reached 
           (corresponding to a fixed modification factor :math:`f`).
        """

        if len(self._energy_bins) == 0:
            self._energy_bins = cdos[:,0].copy()
            
        self._cdos = cdos[:,1].copy()
        self._histogram = cdos[:,2].copy()
        self._f = f
        self._flatness_condition = flatness_condition

        self._histogram_minimum = hist_cond[0]
        self._histogram_average = hist_cond[1]
        elapsed_time = time.time() - start_time
        
        self._stored_cdos.append(
            dict(
                [
                    ('cdos', self._cdos),
                    ('histogram', self._histogram),
                    ('modification_factor', f),
                    ('flatness_condition', flatness_condition),
                    ('histogram_minimum', hist_cond[0]),
                    ('histogram_average', hist_cond[1]),
                    ('n_mc_steps', n_mc_steps),
                    ('n_mc_steps_total', n_mc_steps_total),
                    ('elapsed_time [s]', elapsed_time),
                    ('elapsed_time [(d.)h.m.s]', str(datetime.timedelta(seconds=elapsed_time))),
                    ('n_mc_per_second', n_mc_steps_total/elapsed_time),

                ]
            )
        )
        
    def get_cdos(
            self,
            ln = False,
            normalization = True,
            normalization_type = 0,
            set_normalization_ln = None,
            modification_factor = None,
            discard_empty_bins = True
    ):
        """Returns the energy_bins and configurational density of states (CDOS) as arrays, respectively.
           
        **Parameters**:
            
        ``ln``: boolean (default: False)
            Whether to return the natural logarithm of the CDOS. 
        
        ``normalization``: boolean (default: True)
            Whether to normalize the configurational density of states.
        
        ``normalization_type``: integer (default: ``0``)
            If ``normalization`` is true, type of normalization applied. Possible values are:

                * ``0``: :math:`g(E_{min}) = 1`
                * ``1``: :math:`\sum_E g(E) = \sum_{subl.} Binom(N_{sites}, n_{subs})`,
                    i.e. the total weight of the histogram equals the total number of configurations.
                * ``2``: :math:`\sum_E g(E) = e^{F}`, 
                    where :math:`F` is a custom normalization factor
                    given by the argument ``set_normalization_ln``, see below.
        
        ``set_normalization_ln``: float (default: None)
            If not **None**, the normalization is set by the exponential of the given value. 
            See option ``2`` of ``normalization_type``.

        ``discard_empty_bins``: boolean (default: True)
            Whether to keep the energy bins that are not visited during the sampling.

        ``modification_factor``: float (default: None)
            If **None**, the CDOS from the last iteration is returned. If not **None**, the CDOS 
            corresponding to the given modification factor is returned.                    
        """
        if modification_factor is None:
            modification_factor = self._stored_cdos[-1]['modification_factor']
            ln_g = self._cdos.copy()
        else:
            from clusterx.utils import isclose
            for gj,ln_gstored in enumerate(self._stored_cdos):
                if isclose(modification_factor,ln_gstored['modification_factor'],rtol = 1.0e-8):
                    ln_g = ln_gstored['cdos'].copy()
                    
        ln_mod_fac = math.log(modification_factor)
        
        if normalization:
            eb = []

            ln_norm_factor = 0 
            if normalization:

                first_i_nonzero = 0
                while ln_g[first_i_nonzero] < ln_mod_fac:
                    first_i_nonzero += 1
                ln_g0 = ln_g[first_i_nonzero]
                
                if normalization_type == 0:
                    ln_norm_factor = -ln_g0
                    
                elif normalization_type == 1:
                    # gsum = sum_E g(E)/g0
                    ln_gsum = math.log( np.sum([math.exp(ln_ge-ln_g0) for ln_ge in ln_g]) )

                    # nsites_dict = {0:16, 1:3} there is in total 16 sites in sublattice 0 and 3 sites in sublattice 1.
                    # self._nsubs = {0: [8], 1:[2]} there are 8 substituent atoms in sublattice 0 and 2 substituent atoms in sublattice 1.
                    nsites_dict = self._scell.get_nsites_per_type()
                    sublattices = [int(k) for k in self._nsubs.keys()]

                    ln_binomcoeff = 0
                    for sublattice in sublattices:
                        nsubs = self._nsubs[str(sublattice)][0] # [8]
                        nsites = nsites_dict[sublattice] # 16
                        # ln[binom(m,n)] = ln[Gamma(m+1)]-ln[Gamma(n+1)]-ln[Gamma(m-n+1)]
                        ln_binomcoeff += scipy.special.gammaln(nsites+1)-scipy.special.gammaln(nsubs+1)-scipy.special.gammaln(nsites-nsubs+1)
                        
                    ln_norm_factor = ln_binomcoeff - ln_g0 - ln_gsum

                elif normalization_type == 2:
                    # gsum = sum_E g(E)/g0
                    ln_gsum = math.log( np.sum([math.exp(ln_ge-ln_g0) for ln_ge in ln_g]) )

                    ln_norm_factor = set_normalization_ln - ln_g0 - ln_gsum

                
            gc = []
            for i, ln_gi in enumerate(ln_g):
                if ln_gi >= ln_mod_fac:
                    ln_gi_norm = ln_gi + ln_norm_factor
                    if not ln:
                        gc.append(math.exp(ln_gi_norm))
                    else:
                        gc.append(ln_gi_norm)
                    eb.append(self._energy_bins[i])
                    
                else:
                    if not discard_empty_bins:
                        gc.append(float(0))
                        eb.append(self._energy_bins[i])
                
            self._cdos_normalized = gc
            self._energy_bins_cdos_normalized = eb
            
            return eb, gc
        
        else:
            if discard_empty_bins:
                gc = []
                eb = []
                for i,ge in enumerate(ln_g):
                    if ge >= ln_mod_fac:
                        if not ln:
                            gc.append(math.exp(ge))
                        else:
                            gc.append(ge)
                        eb.append(self._energy_bins[i])
                        
                return eb, gc
            
            else:
                if not ln:
                    _expg = [math.exp(ge) for ge in ln_g]
                    return self._energy_bins, _expg
                else:

                    return self._energy_bins, ln_g
        

    def calculate_thermodynamic_property(self, temperatures, prop_name = "U", modification_factor = None):
        """Calculate the thermodynamic property with name ``prop_name`` for the temperature list given 
        list ``temperatures``. 

        In disregard of the normalization used for the configurational density of states :math:`g(E)`,
        for numerical convenience this method internally normalizes :math:`g(E)` according to
        
        .. math::
        
            g(E_0=0) = 1

        For the output quantities other normalizations can be easily obtained considering the following 
        transformations:

        .. math::
        
            & g(E) \\rightarrow c g(E)

            & Z(T) \\rightarrow c Z(T)

            & S(T) \\rightarrow S(T) + k_B ln(c)

            & F(T) \\rightarrow F(T) - k_B T ln(c)

        **Parameters**:
        
        ``temperatures``: list of floats 
            Temperatures for which the thermodynamic property is calculated, in Kelvin.

        ``prop_name``: string (default: U)
            Name of thermodynamic property.

            If **U**, the internal energy is calculated.

            If **C_p**, the isobaric specific heat at zero pressure is calculated. 

            If **F**, the free energy is calculated.

            If **S**, the entropy is calculated.

        ``modification_factor``: float (default: None)
            If **None**, the CDOS from the last iteration is used for calculuting the thermodynamic property. 
            If not **None**, the CDOS corresponding to the given modification factor is used.

        """
        from ase.units import kB as kb
        import sys
        ln_maxfloat = math.log(sys.float_info.max)
        e, log_g = self.get_cdos(ln = True, normalization = True, discard_empty_bins = True,  modification_factor = modification_factor)
        thermoprop = np.zeros(len(temperatures))

        e = np.array(e)
        log_g = np.array(log_g)
        log_g -= log_g[0] # Assume normalization g(E_0=0) = 1
        e -= e[0]
        
        for i, t_i in enumerate(temperatures):
            beta_i = 1.0/(kb*t_i)

            p = np.zeros(len(e))
            u = 0
            u2 = 0
            z = 0

            # Compute canonical probability as
            # P(E) = 1 / \sum_Ep exp(ln(Ep)-ln(E)-\beta * (Ep-E))
            for j,log_g_j in enumerate(log_g):
                pinv_j = 0
                for k,log_g_k in enumerate(log_g):
                    exponent = log_g_k-log_g_j-beta_i*(e[k]-e[j])
                    if exponent > ln_maxfloat or pinv_j == math.inf:
                        pinv_j = 0
                        break
                    else:
                        pinv_j += math.exp(exponent)

                if pinv_j != 0:
                    p[j] = 1/pinv_j

            # Here we use: ln(Z) = ln(g(E)) - \beta E - ln(P(E,T))
            j_max = np.argmax(p)
            ln_z = log_g[j_max] - beta_i * e[j_max] - math.log(p[j_max])
            
            f = - kb * t_i * ln_z
                    
            for j,log_g_j in enumerate(log_g):
                
                u += e[j] * p[j]
                
                u2 += e[j]*e[j] * p[j]
                    
            if prop_name == "U":
                thermoprop[i] = u
                
            elif prop_name == "Cp" or prop_name == "C_p":
                thermoprop[i] = ( u2 - u * u ) / ( kb * t_i * t_i )
                
            elif prop_name == "F":
                thermoprop[i] = f
                
            elif prop_name == "S":
                thermoprop[i] = (u-f)/t_i
                
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
                    
                    _plat = ParentLattice(
                        atoms = Atoms(
                            positions = superdict['parent_lattice']['positions'],
                            cell = superdict['parent_lattice']['unit_cell'],
                            numbers=np.zeros(len(species)),
                            pbc = np.asarray(superdict['parent_lattice']['pbc'])
                        ),
                        sites  = species,
                        pbc = np.asarray(superdict['parent_lattice']['pbc'])
                    )
                    
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
