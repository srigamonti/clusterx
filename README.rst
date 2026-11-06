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

Running tests
-------------

With the project environment active, run tests from the repository root::

    cell test
    cell test -n test_parent_lattice
    python -m pytest -q clusterx/test/test_parent_lattice.py

Each test automatically runs in its own temporary working directory. Generated
structures, databases, and other artifacts stay there for inspection; pytest
retains temporary directories from the last three runs by default.

``cell test`` reports to the terminal. To generate XML and HTML reports
explicitly, use pytest::

    python -m pytest clusterx/test --junitxml=.test-reports/testlog.xml --html=.test-reports/testlog.html

For development, install the optional tools, including Ruff::

    python -m pip install -e '.[dev]'

Run this command from the repository root, writing ``'.[dev]'`` exactly as
shown. The ``.`` refers to the package in the current directory, and ``[dev]``
selects the optional development dependencies declared in ``setup.py`` (currently
Ruff). The ``-e`` option installs CELL in editable mode, so source changes take
effect without reinstalling the package.

For environments managed with uv, use ``uv pip install -e '.[dev]'``.
Run ``ruff check`` on changed Python files; it uses the repository's
``ruff.toml`` configuration.

Affine model predictions
------------------------

``Model`` automatically folds supported fitted affine estimators into effective
coefficients in correlation space and a complete intercept. This includes
``ColumnExpander``, ``PolynomialFeatures`` of degree zero or one,
``StandardScaler``, ``SelectFromModel``, and passthrough steps, followed by
``LinearRegression``, ``Ridge``, ``RidgeCV``, ``Lasso``, ``LassoCV``, ``ElasticNet``,
or ``ElasticNetCV``. Multi-output coefficients are supported. Multiplicities
are not applied again.

The fitted estimator remains available as ``model.estimator`` for reference.
Other transformations, estimator subclasses, and nonlinear polynomials use
the original estimator prediction path. Local affine energy differences use
only the effective coefficients, so offsets from centering and bias features
cancel correctly.

The folded coefficients are cached lazily and are not saved in pickle files.
Existing saved models enable the optimization automatically when loaded.
Replacing the estimator or a pipeline step, or refitting with new final
``coef_``/``intercept_`` objects, refreshes the cache automatically. After
editing fitted arrays in place or changing transformer/selector settings with
``set_params()``, call ``model.invalidate_prediction_cache()``. Starting a new
Monte Carlo run also clears the cache through ``reset_mc()``.

For a reproducible comparison with the fitted pipeline, run::

    python benchmarks/affine_prediction.py pruned_model.pickle --reference-model full_model.pickle

The benchmark uses one numerical-library thread, a half-filled 4x4x3 supercell,
100 warm-up moves, and three timed repetitions of 5,000 proposed swaps. It
checks prediction errors, composition, final occupations, and both MC RNG states.

With ``predict_swap=True``, local cluster calculations access occupations directly
without allocating occupation and species-count arrays for each cluster.
With ``reduce_structure=True``, periodic neighborhoods are extracted in one
gather. Cached offsets and temporary neighborhood storage depend on the reduced
cell, so their size stays fixed when the full supercell grows for a fixed model.
The offset cache refreshes automatically when either grid shape changes.

Reduced local correlation changes also use one compiled loop over packed
interaction arrays. This cache contains only reduced-cell geometry and is
limited to 16 MiB of numeric storage; larger pools use the per-cluster
calculation. Unreduced calculations do not create a packed cache. The cache
is discarded on MC reset or rebuilding interactions and is rebuilt after
loading a saved model. Switching supercell transformations rebuilds interactions
automatically. Cluster order, normalization, and rounding are preserved.
