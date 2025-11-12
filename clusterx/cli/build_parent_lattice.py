import plac
from ase.build import bulk

from clusterx.parent_lattice import ParentLattice

commands = ["build_parent_lattice"]


@plac.annotations(
    species=("Comma-separated list of species, e.g., Si,Ge", "option", "s", str),
    filepath=(
        "File path where to serialize the generated parent lattice",
        "option",
        "f",
        str,
    ),
)
def build_parent_lattice(species: str = "Si,Ge", filepath="plat.json"):
    """Build parent lattice

    This command constructs a single-sublattice parent lattice. It generates a lattice
    for the first species using ASE's :func:`ase.build.bulk` function
    provided in the input list and creates a :class:`~clusterx.parent_lattice.ParentLattice`
    object where all lattice sites are associated with the same set of possible
    species.

    The resulting parent lattice is serialized to a JSON file.

    Parameters
    ----------
    species : str, optional
        Comma-separated list of chemical species to include in the parent lattice.
        The first species is used to generate the underlying atomic structure.
        Default is ``"Si,Ge"``.
    filepath : str, optional
        File path to which the generated :class:`~clusterx.parent_lattice.ParentLattice`
        is serialized. Default is ``"plat.json"``.

    Examples
    --------
    .. code-block:: bash

        $ clusterx build_parent_lattice -s Si,Ge -f plat.json

    Notes
    -----
    This command is primarily intended for demonstration or testing.
    For realistic parent lattices, users should define them explicitly using
    the :class:`~clusterx.parent_lattice.ParentLattice` interface.
    """
    species_list = species.split(",")
    pristine_atoms = bulk(species_list[0])
    nsites = len(pristine_atoms)
    plat = ParentLattice(pristine_atoms, symbols=[species_list] * nsites)

    plat.serialize(filepath)
