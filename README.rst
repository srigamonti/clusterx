CELL
====

CELL (aka clusterX) is a python package for building Cluster Expansion models of
simple and complex alloys and performing thermodynamical analyses of materials.

Webpage: http://sol.physik.hu-berlin.de/cell/

Importing ``clusterx`` and starting ``cell`` do not read or write the legacy
``cellconfig.json`` file. Run parameters come from command-line options or TOML
input files, and individual commands manage their output artifacts.
The unused ``clusterx.config`` module and its ``CellConfig`` API have been
removed, along with the package attributes ``config``, ``cfg``, ``PRECISION``, and
``T2D``.

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

Ordered command workflows
-------------------------

To repeat a command with other commands in between, give each invocation a unique
step name and select its implementation with ``command``. This uses standard TOML
single-bracket tables::

    [export_before]
    command = "convert_dss_to_sset"
    dss_filepath = "dss.pickle"
    sset_filepath = "before.xyz"

    [plot]
    command = "plot_columns"
    source_filepath = "dss.pickle"
    xcolumn = "f_mix_pred_full"
    ycolumn = "f_mix_pred_pruned"

    [export_after]
    command = "convert_dss_to_sset"
    dss_filepath = "dss.pickle"
    sset_filepath = "after.xyz"

Run this with ``cell --input_file workflow.toml``. By default, named steps execute
in file order. Set ``ignore = true`` inside a step to skip it. To select or reorder
steps, place ``do = ["export_before", "plot", "export_after"]`` at the top of the
file, before any table header. Command names and argument signatures are checked
before execution begins.

Existing ``[command]`` tables and ``[[command]]`` arrays remain supported and can
be mixed with named steps. Legacy array entries run consecutively as one group;
use named steps when individual invocations must be interleaved. A registered
command's own table still passes its options directly, so an actual argument
named ``command`` remains supported for custom commands.
