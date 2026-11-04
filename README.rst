CELL
====

CELL (aka clusterX) is a python package for building Cluster Expansion models of
simple and complex alloys and performing thermodynamical analyses of materials.

Webpage: http://sol.physik.hu-berlin.de/cell/

Inspect saved data
------------------

The CLI provides formatted summaries of saved CELL objects and correlation
matrices. Each command takes the file to inspect as its positional argument::

    cell print_info_plat plat.json
    cell print_info_cpool cpool.json
    cell print_info_comat comat.npz
    cell print_info_model model.pickle
    cell print_info_ccalc ccalc.pickle
    cell print_info_sset sset.xyz
    cell print_info_derivative_structures dss.pickle

``print_info_dss`` is a short alias for ``print_info_derivative_structures``.
The previous names ``print_plat_info``, ``print_cpool_info``,
``summarize_derivative_structures``, and ``summarize_dsgenerator`` remain available.

Structure-set reports support ``.db``, ``.json``, and CELL extended ``.xyz`` files.
Model reports show stored coefficients, fitted pipeline and ensemble details,
and the associated correlation calculator when available. They can inspect a
standalone model pickle as well as the usual model/calculator file pair.
Every parent-lattice table lists the number of supported species per sublattice,
including vacancies and spectator sublattices.
These commands read the artifacts without fitting models or regenerating orbits.
