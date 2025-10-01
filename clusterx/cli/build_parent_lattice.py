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

    Build single sublattice parent lattice for simple demonstration purposes.
    Uses "bulk" module of ASE.
    """
    species_list = species.split(",")
    pristine_atoms = bulk(species_list[0])
    nsites = len(pristine_atoms)
    plat = ParentLattice(pristine_atoms, symbols=[species_list] * nsites)

    plat.serialize(filepath)
