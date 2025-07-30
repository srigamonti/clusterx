# Copyright (c) 2015-2025, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

import warnings

import numpy as np


class MonteCarloLite:
    r"""Monte Carlo class for binary single sublattice CE models

    **Description**:
        Objects of this class are used to perform Monte Carlo samplings.

        It is initialized with:

        - a Model object, that enables to calculate the energy of a structure,

        - a SuperCell object, in which the sampling is performed,

        - specification of the thermodynamic ensemble:

            If ``ensemble`` is 'canonical', the composition for the sampling is defined with ``nsubs``. In case
            of multilattices, the sublattice for the sampling can be refined with ``sublattice_indices``.

            If ``ensemble`` is 'grandcanonical', the sublattice is defined with ``sublattices_indices``.

    **Parameters**:

    ``energy_model``: Model object
        Model used for acceptance and rejection. Usually, the Model enables to
        calculate the total energy of a structure.

    ``scell``: SuperCell object
        Simulation cell in which the sampling is performed.

    ``energy_scale_factor``: float or None
        the energy given by energy_model times the energy_scale_factor must give the total
        energy in the simulation supercell. If None, it is assumed that the energy_model
        gives the energy per parent lattice and energy_scale_factor is set to scell.get_index().

    ``ensemble``: string (default: ``canonical``)
        ``canonical`` allows for swaps of atoms that conserve the concentration defined with ``nsubs``.

        ``grandcanonical`` allows for replacing atoms in the sub-lattices defined with ``sublattice_indices``.
        (So far, ``grandcanonical`` is not yet implemented.)

    .. todo:
        Samplings in the grand canonical ensemble are not yet possible.

    """

    def __init__(
        self,
        energy_model,
        scell,
        energy_scale_factor=None,
        boltzmann_constant=1.0,
        mcsetup_filepath=None,
    ):
        self._emodel = energy_model
        self._scell = scell
        self._energy_scale_factor = (
            energy_scale_factor
            if energy_scale_factor is not None
            else scell.get_index()
        )
        self._kb = boltzmann_constant

        if not scell.is_nary(2):
            raise ValueError("MonteCarloLite works only for binary materials.")

        self._substitutional_sublattice = None
        for k, v in scell.get_sublattice_types().items():
            if len(v) == 2:
                self._substitutional_sublattice = int(k)
                break

    def metropolis(
        self,
        temperature=None,
        n_mc_steps=None,
        n_flips_per_mc_step=None,
        ensemble=None,
        n_substitutions=None,
        chemical_potential=None,
        mcrun_filepath=None,
        n_error_reset=None,
        initial_structure=None,
        **kwargs,
    ):
        r"""Perform Monte-Carlo Metropolis simulation

        **Description**:
            Perfom Monte-Carlo Metropolis sampling for
            ``no_of_sampling_steps`` sampling steps.

            During the sampling, a new structure at step i is accepted
            with the probability given by :math:`\min( 1, \exp( - (E_i - E_{i-1})/(k_B T)) )`

            The energy :math:`E_i` of visited structure at step i is calculated from the Model
            ``energy_model``. The factor :math:`k_B T` is the product of the temperature :math:`T`
            and the Boltzmann constant :math:`k_B`.

            Note: The units of the ``energy`` :math:`E` and the factor :math:`k_B T` must be the same.
            With ``scale_factor``, :math:`k_B T` can be adjusted to the correct units (see below).

        **Parameters**:

        ``n_mc_steps``: integer
            Number of sampling steps

        ``temperature``: float
            Temperature at which the sampling is performed.

        ``n_flips``: integer
            Number of species flips per sampling step. Must be an even number for canonical ensemble.
            Defaults to ``1`` for ``grandcanonical`` and to ``2`` form ``canonical``.

        ``n_substitutions``: integer (default = None)
            Defines the number of substituted atoms in the sublattice

        ``chemical_potential``: dictionary (default: None)
            Define the chemical potential in the grand canonical ensemble.

        ``mcrun_filepath``: string (default: ``trajectory.json``)
            Name of a Json file in which the trajectory is serialized after the sampling if ``serialize`` is **True**.

        ``n_error_reset``: integer (default: None)
            If not **None*  and ``predict_swap`` equal to **True**, the correlations are calculated as usual (no differences) every n-th step.

        ``initial_structure``: Structure object
            Initial structure, from which the sampling starts.
            If ``None``, sampling starts from a random structure.

        ``**kwargs``: keyworded argument list, arbitrary length
            These arguments are added to the MonteCarloTrajectory object that is initialized in this method.

        **Returns**: MonteCarloTrajectory object
            Trajectory containing the complete information of the sampling trajectory.

        """
        import math

        from tqdm import tqdm

        from clusterx.utils import poppush

        if n_flips is None:
            n_flips = 1 if ensemble == "grandcanonical" else 2

        if initial_structure is None:
            if n_substitutions is not None:
                struc = self._scell.gen_random_structure(n_substitutions)
            else:
                struc = self._scell.gen_random_structure()
        else:
            struc = initial_structure

        self._emodel.corrc.reset_mc(mc=True)
        e = self._emodel.predict(struc)

        mcrun_filepath
        if mcrun_filepath is None:
            warnings.warn(
                "No file path provided for storing the MC run. Trajectory will not be saved to a file.", UserWarning
            )

        mcrun = MCRun(self._scell, temperature, ensemble, mcrun_filepath)

        mcrun.sigmas.append(tuple(struc.get_sigmas()))
        mcrun.energies.append(e)

        if n_error_reset is not None:
            error_steps = int(n_error_reset)
            x = 1

        for i in tqdm(
            range(1, n_mc_steps + 1),
            total=n_mc_steps,
            desc="MMC simulation",
        ):
            indices_list = []

            for j in range(n_flips):
                ind1, ind2, site_type, rindices = struc.swap_random([self._substitutional_sublattice])
                indices_list.append([ind1, ind2, [site_type, rindices]])

            if self._control_flag:
                if self._error_reset:
                    if x > error_steps:
                        x = 1
                        e1 = self._em.predict(struc)
                    else:
                        x += 1
                        de = self._em.predict_swap(
                            struc,
                            ind1=ind1,
                            ind2=ind2,
                            site_types=self._sublattice_indices,
                        )
                        e1 = e + de
                else:
                    de = self._em.predict_swap(
                        struc, ind1=ind1, ind2=ind2, site_types=self._sublattice_indices
                    )
                    e1 = e + de

            else:
                e1 = self._em.predict(struc)

            if e >= e1:
                accept_swap = True
                boltzmann_factor = 0
            else:
                boltzmann_factor = math.exp((e - e1) / (scale_factor_product))

                if np.random.uniform(0, 1) <= boltzmann_factor:
                    accept_swap = True
                else:
                    accept_swap = False

            if accept_swap:
                e = e1

                if self._models:
                    key_value_pairs = {}
                    for m, mo in enumerate(self._models):
                        key_value_pairs.update({mo.property_name: mo.predict(struc)})
                    traj.add_decoration(
                        i,
                        e,
                        [[li[0], li[1]] for li in indices_list],
                        key_value_pairs=key_value_pairs,
                    )

                else:
                    traj.add_decoration(i, e, [[li[0], li[1]] for li in indices_list])

                if acceptance_ratio:
                    ar = poppush(hist, 1)

            else:
                for j in range(self._no_of_swaps - 1, -1, -1):
                    struc.swap(
                        indices_list[j][1],
                        indices_list[j][0],
                        site_type=indices_list[j][2][0],
                        rindices=indices_list[j][2][1],
                    )

                if acceptance_ratio:
                    ar = poppush(hist, 0)

            if acceptance_ratio:
                if i % 10 == 0 and i >= nar:
                    scale_factor_product *= math.exp(
                        (acceptance_ratio / 100.0 - ar) / 10.0
                    )

        if serialize:
            traj.serialize()

        return traj


class MCRun:
    def __init__(self, scell, temperature, ensemble, mcrun_filepath):
        self.scell = scell
        self.temperature = temperature
        self.ensemble = ensemble
        self.mcrun_filepath = mcrun_filepath

        self.sigmas = []
        self.energies = []
