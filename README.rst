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

Nonlinear Monte Carlo
---------------------

For models nonlinear in their correlations, including polynomial degrees two
and higher, use ``predict_swap=True`` and pass ``nonlinear=True`` to
``MonteCarlo.metropolis``. This also supports ``reduce_structure=True``.
Recognized affine models keep their direct energy-difference path even when
``nonlinear=True`` is requested.

The nonlinear path initializes the global correlation vector and caches its
model prediction. Each proposal adds the local correlation changes and evaluates
the trial vector once. Recognized polynomial pipelines use a compiled sparse
monomial evaluator, with fitted expansion, feature selection, scaling, bias,
and feature order folded into its cached parameters. Other supported linear
transformations retain the affine fast path; unsupported pipelines use sklearn's
fitted predictor. Only accepted moves replace the cached correlations and
prediction. The compiled cache is limited to 16 MiB and is rebuilt after loading
a model or invalidating fitted prediction parameters. No degree-specific dense
coefficient tensors are constructed.

Reduced initialization streams one parent-cell neighborhood at a time, sums
correlation contributions, and normalizes once before evaluating the model.
It uses the same capped reduced-cell geometry cache as local moves. Added MC
workspace consists of correlation vectors and reduced neighborhoods, independent
of the full atom count for a fixed model. The fitted estimator still has its
own workspace requirements, which depend on feature count and polynomial degree.
Reduced trajectories continue to report energies relative to the initial state;
the cached absolute prediction is kept separately.

Long proposal runs use bounded temporary temperature arrays by default. For
example, run two million proposals in batches of 20,000 while retaining one
continuous chain::

    mc = MonteCarlo(model, scell, nsubs={0: [480]}, predict_swap=True, seed=42)
    trajectory = mc.metropolis(
        no_of_sampling_steps=2_000_000,
        temperature=300,
        max_steps_per_batch=20_000,
        nonlinear=True,
    )

``max_steps_per_batch`` defaults to 100,000 and bounds only temporary
per-proposal temperature storage; structure and nonlinear correlation state
are retained across those internal batches. Use ``serialize=False`` (the
default) and periodic XYZ checkpoints when the accepted-state trajectory itself
must not grow with run length. To continue with a later temperature schedule,
call ``metropolis(..., continue_chain=True)`` on the same object. This retains
the configuration, random streams, nonlinear correlations, cached energy, and
acceptance-rate adaptation. Call ``mc.reset_chain(seed=...)`` to start fresh.
After changing or refitting the model, reset the chain before continuing so its
cached energy uses the updated fitted model.

To start from the final complete frame in an XYZ/extxyz file, pass
``resume_from`` when constructing ``MonteCarlo``::

    mc = MonteCarlo(
        model, scell, nsubs={0: [480]}, predict_swap=True,
        resume_from="previous-temperature.extxyz", seed=43,
        save_atoms_every=10_000, atoms_filename="anneal.extxyz",
    )
    trajectory = mc.metropolis(500_000, temperature=250, nonlinear=True)

Resume checks the cell, periodicity, ordered site positions, allowed species,
and canonical composition before building the ``Structure``. It uses a new
random stream initialized from ``seed``; XYZ does not contain enough state to
reproduce the original proposal sequence exactly. New XYZ checkpoints include
all sites and can be used for later resumes. Use ``reset_chain(resume_from=...)``
to explicitly discard an in-memory chain and start from another saved frame.

Reduced local updates require that a cluster does not wrap onto the same atom
more than once. Such small periodic cells raise an error requesting larger
periodic repeat lengths; the local flip calculation cannot treat these images
as independently occupied sites.

``Model.get_correlations(structure, reduce_structure=True)`` and
``Model.predict(structure, reduce_structure=True)`` use this streaming calculation
after initializing the reduced model and structure, as Monte Carlo does.

Annealing and continuation from TOML
-----------------------------------

The maintained engine is selected by ``command = "metropolis"`` and
``task = "runmc"`` (also ``runMC`` or ``run-monte-carlo``). It accepts ordered
``runs`` stages, using one ``MonteCarlo`` instance throughout the schedule.
Each stage must give a positive ``temperature`` and exactly one length:
``n_mc_steps`` proposals or ``n_mc_sweeps`` sweeps. One sweep is as many
proposals as there are substitutional sites in the whole supercell, summed
over all substitutional sublattices, including when sampling is restricted to
selected sublattices. Spectators do not count. A proposal contains ``n_clics``
swaps or flips (default one). Fractional sweeps are allowed when they give an
integer proposal count. Stage lengths cannot also be supplied at the top level.
A single-temperature call can instead specify its length and temperature
directly; omitting both length options retains the historical 100 proposals.

For a fresh run, save the following as ``anneal.toml`` in a directory containing
the saved gamma model, with the parent-lattice path shown below available.
This example uses the 1,008,000-atom supercell from the perovskite workflow:
604,800 I/Br substitutional sites, with 302,400 Br atoms. No fitting step is
needed. Execute with ``cell --input_file anneal.toml``::

    do = ["anneal"]

    [anneal]
    command = "metropolis"
    task = "runmc"
    model_filepath = "f_mix_2_pruned_model.pickle"
    plat_filepath = "../../data/gamma_parent_lattice/plat_I-Br_gamma_sym.json"
    sc_shape = [42, 40, 30]
    n_substitutions = {"0" = [302400]}
    ensemble = "canonical"
    predict_swap = true
    reduce_structure = true
    nonlinear = true
    random_seed = 42
    energy_normalization = "per_atom"
    boltzmann_constant = 8.617330337217213e-5  # ASE kB in eV/K
    max_steps_per_batch = 100000
    serialize = false
    save_atoms_endpoints = true
    save_atoms_every = 0
    atoms_filename = "anneal.xyz"
    atoms_mode = "x"

    [[anneal.runs]]
    temperature = 300.0
    n_mc_sweeps = 40

    [[anneal.runs]]
    temperature = 250.0
    n_mc_sweeps = 40

The model's ``f_mix`` is in eV/atom. ``energy_normalization = "per_atom"``
passes ``scale_factor=[len(scell)]`` to the Python API. Alternatives are
``"total"`` (factor one) and ``"per_parent_cell"`` (number of parent cells).
The existing ``energy_scale_factor`` convention is preserved: the CLI passes
``scale_factor=[1 / energy_scale_factor]``, so a per-atom model requires the
reciprocal of the total atom count. Set only one of ``energy_normalization``
and ``energy_scale_factor``. If neither is set, the historical default is the
number of parent cells, and ``boltzmann_constant`` still defaults to one.

To continue from the final complete frame of ``anneal.xyz``, create a separate
``continue-anneal.toml`` and run ``cell --input_file continue-anneal.toml``::

    do = ["continue_anneal"]

    [continue_anneal]
    command = "metropolis"
    task = "runmc"
    model_filepath = "f_mix_2_pruned_model.pickle"
    plat_filepath = "../../data/gamma_parent_lattice/plat_I-Br_gamma_sym.json"
    sc_shape = [42, 40, 30]
    n_substitutions = {"0" = [302400]}
    ensemble = "canonical"
    predict_swap = true
    reduce_structure = true
    nonlinear = true
    random_seed = 43
    resume_from = "anneal.xyz"
    energy_normalization = "per_atom"
    boltzmann_constant = 8.617330337217213e-5
    max_steps_per_batch = 100000
    serialize = false
    save_atoms_endpoints = true
    save_atoms_every = 0
    atoms_filename = "anneal-continued.xyz"
    atoms_mode = "x"

    [[continue_anneal.runs]]
    temperature = 200.0
    n_mc_sweeps = 40

    [[continue_anneal.runs]]
    temperature = 150.0
    n_mc_sweeps = 40

Resume restores the physical configuration and rebuilds nonlinear correlations
and model energy before sampling. It starts a new seeded random stream.
The continuation schedule is always explicit; completed or remaining sweeps
are never inferred from the source file. The cell, periodicity, ordered
positions, optional sublattice tags, allowed species, and canonical composition
must match. Full-site XYZ/extxyz files are required, including spectators.
An incomplete final XYZ frame is ignored; if explicitly appending to that same
file, its incomplete tail is removed before writing new frames.
For a single-temperature continuation without ``runs``, supply both
``temperature`` and one of ``n_mc_steps`` or ``n_mc_sweeps`` explicitly.

Each CLI step constructs a fresh MC object. Within its schedule, all later
stages use ``continue_chain=True`` without passing another initial decoration,
preserving site-index maps, both random streams, nonlinear correlation and
energy caches, and acceptance-rate adaptation. A fresh named step without
``resume_from`` starts a new configuration; Python callers can also explicitly
call ``reset_chain()``. ``continue_chain`` is routed to the first Python call
if supplied, but does not share an in-memory chain between separate CLI steps.

``save_atoms_endpoints = true`` writes the actual starting configuration and
each temperature endpoint as full-site XYZ frames suitable for OVITO.
``save_atoms_every`` independently adds checkpoints at a fixed number of
proposals since this chain started. It is independent of
``max_steps_per_batch``. Coincident interval/endpoint frames and shared stage
boundaries are written once. Checkpoint writes do not change the proposal
sequence. Output metadata contains:

- ``temperature``: the current stage temperature.
- ``mc_step``: proposals since this fresh start or XYZ resume, beginning at zero.
- ``mc_sweeps``: ``mc_step`` divided by ``mc_sweep_sites``, the total number of
  substitutional sites.
- ``mc_stage``: one-based temperature-stage number in this invocation.
- ``mc_stage_step``: proposals within that stage.
- ``mc_counter_origin``: ``fresh_start`` or ``xyz_resume``.

These counters describe the current chain session, and never claim cumulative
work performed before the source checkpoint. ``atoms_mode = "x"`` refuses an
existing output file, ``"w"`` explicitly overwrites it, and ``"a"`` appends.
The mode is applied once at the start; later stages append. The default
``"auto"`` preserves the Python API's fresh-overwrite/resume-append behavior.
The continuation example uses a distinct output file to retain source results.

``serialize`` retains its historical CLI default of ``true``. Set it to
``false`` for large runs to avoid accumulating accepted-move histories.
When serializing a schedule, JSON histories use
``mc-trajectory_stage0001.json``, ``mc-trajectory_stage0002.json``, etc.;
``traj_filepath`` selects the base path, and each stage may override it.
The older single-run filename behavior and MonteCarloLite tasks are unchanged.

The same options are available directly through plac. Structured values use
JSON, and boolean options take explicit ``true``/``false`` values, for example::

    cell metropolis -task runmc -mofp model.pickle -plfp plat.json \
        -scsh '[8,1,1]' -nsubs '{"0":[4]}' \
        -runs '[{"temperature":300,"n_mc_sweeps":2},{"temperature":250,"n_mc_steps":100}]' \
        -energy_norm per_atom -k_B 8.617330337217213e-5 \
        -serialize false -nonlinear true -batch 100000 \
        -endpoints true -atoms anneal.xyz -atoms_mode x -rs 42
