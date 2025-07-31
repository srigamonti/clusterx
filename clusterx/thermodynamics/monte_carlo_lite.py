# Copyright (c) 2015-2025, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.
from __future__ import annotations

import os
import pickle
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

    @classmethod
    def from_file(cls, filepath: str) -> MonteCarloLite:
        """
        Load a MonteCarloLite object from a serialized file.

        Supported formats:
        - `.pickle`: Full Python object serialization using `pickle`.

        Parameters
        ----------
        filepath : str
            Path to the serialized file. The format is inferred from the file extension.

        Returns
        -------
        Structure
            A reconstructed Structure object based on the contents of the file.

        Raises
        ------
        ValueError
            If the file extension is not supported.
        """
        file_ext = os.path.splitext(filepath)[1].lower()
        if file_ext == ".pickle":
            return cls._load_from_pickle(filepath)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")

    @staticmethod
    def _load_from_pickle(filepath: str) -> MonteCarloLite:
        with open(filepath, "rb") as f:
            return pickle.load(f)

    def metropolis(
        self,
        temperature=None,
        n_mc_steps=None,
        n_clics=1,
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

        ``n_clics``: integer
            Number of species flips (grandcanonical) or species swaps (canonical) per sampling step.
            Defaults to 1.

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

        if initial_structure is None:
            if n_substitutions is not None:
                struc = self._scell.gen_random_structure(n_substitutions)
            else:
                struc = self._scell.gen_random_structure()
        else:
            struc = initial_structure

        scaledbeta = self._energy_scale_factor / self._kb / temperature

        self._emodel.corrc.reset_mc(mc=True)
        e = self._emodel.predict(struc)

        mcrun_filepath
        if mcrun_filepath is None:
            warnings.warn(
                "No file path provided for storing the MC run. Trajectory will not be saved to a file.",
                UserWarning,
            )

        mcrun = MCRun(self._scell, temperature, ensemble)

        mcrun.accepted_steps.append(0)
        mcrun.sigmas.append(tuple(struc.get_sigmas()))
        mcrun.energies.append(e)

        for i in tqdm(
            range(1, n_mc_steps + 1),
            total=n_mc_steps,
            desc="MMC simulation",
        ):
            indices_list = []

            mcrun.clics.append([])

            # make MC move
            atom_indices = []
            new_sigmas = []
            for j in range(n_clics):
                if ensemble == "grandcanonical":
                    atom_index, sigma_initial, sigma_final = struc.flip_random(
                        self._substitutional_sublattice
                    )
                    mcrun.clics[-1].append(
                        {
                            "atom_index": atom_index,
                            "sigma_i": sigma_initial,
                            "sigma_f": sigma_final,
                        }
                    )
                    atom_indices.append(atom_index)
                    new_sigmas.append(sigma_final)
                elif ensemble == "canonical":
                    atom_index1, sigma_initial1, sigma_final1 = struc.flip_random(
                        self._substitutional_sublattice
                    )
                    atom_index2, sigma_initial2, sigma_final2 = struc.flip_random(
                        self._substitutional_sublattice,
                        sigma_initial=sigma_final1,
                        sigma_final=sigma_initial1,
                    )

                    mcrun.clics[-1].append(
                        {
                            "atom_index1": atom_index1,
                            "atom_index2": atom_index2,
                            "sigma_1i": sigma_initial1,
                            "sigma_1f": sigma_final1,
                            "sigma_2i": sigma_initial2,
                            "sigma_2f": sigma_final2,
                        }
                    )
                    atom_indices.append(atom_index1)
                    new_sigmas.append(sigma_final1)
                    atom_indices.append(atom_index2)
                    new_sigmas.append(sigma_final2)

            comps = struc._comps[self._substitutional_sublattice]
            print(comps)
            comps[sigma_initial] -= 1
            comps[sigma_final] += 1
            print(comps)
            # compute new energy
            if n_error_reset is not None and i % n_error_reset == 0:
                e1 = self._emodel.predict(struc)

            else:
                de = 0
                for atom_index, sigma in zip(atom_indices, new_sigmas):
                    de += self._emodel.predict_flip(
                        struc,
                        atom_index=atom_index,
                        new_sigma=sigma,
                        site_types=[self._substitutional_sublattice],
                    )

                e1 = e + de
                print(e1, de)

            if e >= e1:
                accept_swap = True
                boltzmann_factor = 0
            else:
                boltzmann_factor = math.exp((e - e1) * scaledbeta)

                if np.random.uniform(0, 1) <= boltzmann_factor:
                    accept_swap = True
                else:
                    accept_swap = False

            if accept_swap:
                e = e1

                print("before", struc._comps[self._substitutional_sublattice])
                print(atom_indices, new_sigmas)
                struc.update_arrays(atom_indices=atom_indices, new_sigmas=new_sigmas)
                print("after", struc._comps[self._substitutional_sublattice])
                mcrun.accepted_steps.append(i)
                mcrun.sigmas.append(tuple(struc.get_sigmas()))
                mcrun.energies.append(e)

        if mcrun_filepath is not None:
            mcrun.serialize(filepath=mcrun_filepath)

        return mcrun

    def serialize(self, filepath="mcsetup.pickle", fmt="pickle"):
        """Save the structure to a file in the specified format.

        Parameters
        ----------
        fmt : str, optional
            File format for output (default is "pickle").
        filepath : str, optional
            Path to the output file (default is "mcsetup.pickle").

        """
        file_ext = (
            os.path.splitext(filepath)[1].lower().lstrip(".")
        )  # remove leading dot
        fmt = fmt or file_ext  # use file extension as format if fmt is not provided

        if fmt == "pickle":
            with open(filepath, "wb") as f:
                pickle.dump(self, f)
        else:
            raise NotImplementedError(f"Serialization format '{fmt}' is not supported.")


class MCRun:
    def __init__(self, scell, temperature, ensemble):
        self.scell = scell
        self.temperature = temperature
        self.ensemble = ensemble

        self.accepted_steps = []
        self.sigmas = []
        self.energies = []
        self.clics = []

    def serialize(self, filepath: str = None):
        """Serialize the MCRun object using pickle."""
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def from_file(cls, filepath: str) -> "MCRun":
        """Deserialize an MCRun object from a pickle file."""
        with open(filepath, "rb") as f:
            return pickle.load(f)

    def __repr__(self):
        return (
            f"<MCRun(ensemble={self.ensemble}, T={self.temperature}, "
            f"n_steps={len(self.clics)}, n_accepted_steps={len(self.accepted_steps)})>"
            f"acceptance_ratios={len(self.clics) / len(self.accepted_steps) if len(self.accepted_steps) != 0 else 'Undefined'})>"
        )
