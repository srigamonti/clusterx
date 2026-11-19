# Copyright (c) 2015-2026, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

from typing import Optional

import plac
from ase.io import write

from clusterx.cli.config_utils import cmd_message
from clusterx.dbinterfaces.get_nomad_data import Dataset
from clusterx.dbinterfaces.get_nomad_oasis_access_token import (
    get_nomad_oasis_access_token,
)
from clusterx.structures_set import StructuresSet

commands = ["build_structures_set_from_nomad"]


@plac.pos("dataset_id", help="NOMAD dataset ID from which to retrieve entries.")
@plac.opt(
    "pagination_page_size",
    abbrev="ps",
    help="Maximum number of dataset entries to retrieve.",
    type=int,
)
@plac.opt(
    "base_url",
    abbrev="u",
    help="Base URL of the NOMAD / NOMAD OASIS API.",
    type=str,
)
@plac.opt(
    "path_to_dotenv",
    abbrev="e",
    help="Path to .env file containing OASIS_USER_NAME and OASIS_PASSWORD.",
    type=str,
)
@plac.opt(
    "username",
    abbrev="n",
    help="Username for the NOMAD / NOMAD OASIS account.",
    type=str,
)
@plac.opt(
    "userpassword",
    abbrev="p",
    help="Password for the NOMAD / NOMAD OASIS account.",
    type=str,
)
@plac.opt(
    "sset_filepath",
    abbrev="ss",
    help="Output path for the serialized StructuresSet database.",
    type=str,
)
@plac.opt(
    "atoms_db_filepath",
    abbrev="ad",
    help="Output path for the ASE database containing relaxed atoms.",
    type=str,
)
@plac.opt(
    "save_atoms_db",
    abbrev="sad",
    help="If True, write relaxed atoms to the ASE database.",
    type=bool,
)
@plac.opt(
    "parent_lattice_filepath",
    abbrev="pl",
    help="Output path for the serialized parent lattice.",
    type=str,
)
@plac.opt(
    "save_parent_lattice",
    abbrev="spl",
    help="If True, serialize the dataset parent lattice.",
    type=bool,
)
@plac.opt("include_oasis_entry_id", abbrev="oe", help="Add NOMAD entry IDs as the oasis_entry_id property.", type=bool)
@plac.opt("include_total_energy_ev", abbrev="te", help="Add total energies as the total_energy_ev property.", type=bool)
@plac.opt("include_band_gap_ev", abbrev="bg", help="Add band gaps as the band_gap_ev property.", type=bool)
@plac.opt(
    "include_lattice_params",
    abbrev="lp",
    help="Add raw a, b, c, alpha, beta, and gamma lattice parameters.",
    type=bool,
)
@plac.opt(
    "include_normalized_lattice_params",
    abbrev="nlp",
    help="Add normalized a_norm, b_norm, c_norm, alpha_norm, beta_norm, and gamma_norm properties.",
    type=bool,
)
def build_structures_set_from_nomad(
    dataset_id: str,
    pagination_page_size: int = 100,
    base_url: str = "https://sol-oasis.physik.hu-berlin.de/nomad-oasis/api/v1/",
    path_to_dotenv: Optional[str] = None,
    username: Optional[str] = None,
    userpassword: Optional[str] = None,
    sset_filepath: str = "sset.db",
    atoms_db_filepath: str = "atoms_relaxed.db",
    save_atoms_db: bool = False,
    parent_lattice_filepath: str = "plat.json",
    save_parent_lattice: bool = False,
    include_oasis_entry_id: bool = False,
    include_total_energy_ev: bool = False,
    include_band_gap_ev: bool = False,
    include_lattice_params: bool = False,
    include_normalized_lattice_params: bool = False,
):
    """Build a StructuresSet and related files from a NOMAD dataset.

    Dataset properties are added only when their corresponding ``include_*``
    option is set to ``True``. All property options default to ``False``.
    ``include_lattice_params`` adds the raw ``a``, ``b``, ``c``, ``alpha``,
    ``beta``, and ``gamma`` values. ``include_normalized_lattice_params`` adds
    the corresponding parent-lattice-normalized values with ``_norm`` suffixes.

    The parent lattice is serialized only when ``save_parent_lattice`` is
    ``True``. Relaxed atoms are written only when ``save_atoms_db`` is ``True``.
    Credentials can be supplied directly or loaded from a dotenv file;
    otherwise, the command prompts.

    Examples
    --------
    .. code-block:: bash

        $ cell build_structures_set_from_nomad DATASET_ID \\
            --path-to-dotenv=.env --sset-filepath=sset.db

    In a TOML input file, explicitly enable only the properties you want::

        [build_structures_set_from_nomad]
        dataset_id = "DATASET_ID"
        include_total_energy_ev = true
        include_band_gap_ev = true
        include_lattice_params = true
        include_normalized_lattice_params = true
        include_oasis_entry_id = false
        save_atoms_db = true
        save_parent_lattice = true
    """
    cmd_message("head")
    token = get_nomad_oasis_access_token(
        base_url=base_url,
        path_to_dotenv=path_to_dotenv,
        username=username,
        userpassword=userpassword,
        prompt=True,
    )
    dataset = Dataset(
        dataset_id=dataset_id,
        token=token,
        pagination_page_size=pagination_page_size,
        base_url=base_url,
    )

    structures = dataset.get_structure_objects()
    if not structures:
        raise ValueError(f"NOMAD dataset '{dataset_id}' did not contain any structures.")

    parent_lattice = structures[0].get_parent_lattice()
    if parent_lattice is None:
        raise ValueError(f"NOMAD dataset '{dataset_id}' has no parent lattice.")

    if save_atoms_db:
        atoms_objects = dataset.get_atoms_objects()
        if len(atoms_objects) != len(structures):
            raise ValueError(
                f"NOMAD dataset '{dataset_id}' returned {len(atoms_objects)} atoms for {len(structures)} structures."
            )
        write(filename=atoms_db_filepath, images=atoms_objects)

    sset = StructuresSet(parent_lattice)
    sset.add_structures(structures)

    energies = dataset.get_total_energies() if include_total_energy_ev else None
    band_gaps = dataset.get_band_gaps() if include_band_gap_ev else None
    entry_ids = dataset.get_entry_ids() if include_oasis_entry_id else None

    lattice_params = dataset.get_lattice_constants() if include_lattice_params else None
    normalized_lattice_params = (
        dataset.get_normalized_lattice_constants() if include_normalized_lattice_params else None
    )

    property_values = {
        "oasis_entry_id": entry_ids,
        "total_energy_ev": [energy for energy in energies] if energies is not None else None,
        "band_gap_ev": band_gaps,
    }
    lattice_parameter_names = ("a", "b", "c", "alpha", "beta", "gamma")
    if lattice_params is not None:
        property_values.update(zip(lattice_parameter_names, lattice_params))
    if normalized_lattice_params is not None:
        property_values.update(
            (f"{name}_norm", values)
            for name, values in zip(lattice_parameter_names, normalized_lattice_params)
        )
    properties = {name: values for name, values in property_values.items() if values is not None}
    entry_count = len(structures)
    mismatched_lengths = {name: len(values) for name, values in properties.items() if len(values) != entry_count}
    if mismatched_lengths:
        raise ValueError(
            f"NOMAD dataset '{dataset_id}' returned mismatched property counts "
            f"for {entry_count} structures: {mismatched_lengths}"
        )

    for property_name, values in properties.items():
        sset.set_property_values(
            property_name=property_name,
            property_vals=values,
            update_db=True,
        )

    sset.serialize(filepath=sset_filepath, ase_db_type="db", overwrite=True)
    if save_parent_lattice:
        parent_lattice.serialize(parent_lattice_filepath)
    print(f"StructuresSet serialized to {sset_filepath}")
    if save_atoms_db:
        print(f"Relaxed atoms database written to {atoms_db_filepath}")
    if save_parent_lattice:
        print(f"Parent lattice serialized to {parent_lattice_filepath}")
    return sset
