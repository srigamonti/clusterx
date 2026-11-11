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

    python benchmarks/affine_prediction.py pruned_model.pickle --reference-model full_model.pickle \
        --energy-factor 960 --temperature 600 --boltzmann-constant 8.617330337217213e-5

The benchmark uses one numerical-library thread, a half-filled 4x4x3 supercell,
100 warm-up moves, and three timed repetitions of 5,000 proposed swaps. It
checks prediction errors, composition, final occupations, and both MC RNG states.
The example factor assumes that this supercell has 960 atoms and the model
predicts eV/atom. The benchmark requires the physical inputs explicitly.

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

Monte Carlo energy and temperature units
----------------------------------------

Every maintained ``MonteCarlo.metropolis`` call requires three explicit inputs:
``energy_factor``, ``temperature``, and ``boltzmann_constant``. The CLI also
requires ``energy_factor`` and ``boltzmann_constant``; give temperature directly
or in every schedule stage. There is no default physical unit system and no
inferred normalization. The user is responsible for making all three consistent.

For canonical sampling, CELL uses::

    delta_E_total = energy_factor * (model_prediction_after - model_prediction_before)
    acceptance_probability = min(1, exp(-delta_E_total / (boltzmann_constant * temperature)))

``delta_E_total`` MUST be the TOTAL energy change of the entire simulation
supercell, and MUST have the same energy unit as ``boltzmann_constant * temperature``.
``energy_factor`` is one finite positive number, multiplying energy, never
temperature or k_B. It must include both the conversion from the model's
normalization to the whole supercell and any energy-unit conversion. Lists,
reciprocal factors, and automatic atom-count selection are not supported.
For temperature in K and k_B in eV/K:

==========================  =====================================
Model prediction            User-supplied energy_factor
==========================  =====================================
eV per atom                 N_atoms
meV per atom                N_atoms / 1000
eV per parent cell          N_parent_cells
Total supercell energy, eV  1
==========================  =====================================

The relevant atom/site count is part of the USER'S model normalization. For
example, energy per all lattice sites and energy per occupied atom differ when
vacancies are present. CELL cannot determine the correct count from a fitted
model, the property name, or the supercell geometry. Likewise, Kelvin and eV
are examples, not assumptions: another consistent energy/temperature system
is valid. Even an already extensive energy model needs an explicit factor 1.

Example for a model known by the user to predict eV per atom::

    from ase.units import kB  # eV/K

    trajectory = mc.metropolis(
        100_000, energy_factor=float(len(scell)),
        temperature=300.0, boltzmann_constant=kB,
    )

For semigrand sampling, the existing convention adds the supplied
chemical-potential COST change to ``delta_E_total``. These costs must already
have the same total-energy unit as k_B*T and are not scaled by energy_factor.
To represent the usual E - sum(mu*N), supply negative physical chemical
potentials (including a reference value for pristine species). Both the
unconditional acceptance check and exponential use this same combined change.

``target_rate`` actively adapts the inverse temperature. With this option, the
nominal temperature is not a fixed sampling temperature; use fixed-temperature
production sampling without adaptation for fluctuation-based thermodynamics.

Trajectory ``get_energies()`` and ``get_energy()`` return raw MODEL predictions.
``get_model_total_energies()`` and ``get_model_total_energy()`` apply energy_factor.
Reduced MC retains its relative initial-state energy zero in either accessor.
The average property ``U`` is the scaled total supercell energy. The historical
``C_p`` property computes TOTAL configurational heat capacity,
``Var(E_total)/(k_B*T**2)``, in energy per temperature unit. It requires canonical
fixed-temperature sampling without adaptation, contains no pressure/volume
contribution, and is not per atom or divided by k_B. Normalize that result
explicitly if needed.
These averages require the accepted-move history from ``serialize=True``;
endpoint-only XYZ output does not record all samples needed for fluctuations.

The old ``scale_factor`` list, reciprocal ``energy_scale_factor``, and
``energy_normalization`` shortcuts have been removed. Update input files and
Python calls to one multiplicative ``energy_factor`` and explicit k_B and T.
Removed or misspelled sampling arguments now fail instead of becoming metadata;
use the explicit ``metadata`` dictionary for descriptive trajectory information.
CE model pickle files are unaffected. New trajectory JSON stores energy_factor.
When loading an older trajectory lacking that field, supply the factor explicitly
to ``MonteCarloTrajectory(..., read=True, energy_factor=...)``; its old conversion
settings are not inferred or migrated automatically.

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
        energy_factor=float(len(scell)),  # This example assumes a model in eV/atom.
        temperature=300,
        boltzmann_constant=8.617330337217213e-5,  # eV/K
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
    trajectory = mc.metropolis(
        500_000, energy_factor=float(len(scell)),  # Model in eV/atom.
        temperature=250, boltzmann_constant=8.617330337217213e-5, nonlinear=True,
    )

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
    energy_factor = 1008000.0  # USER conversion: eV/atom -> total supercell eV
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

This particular model's ``f_mix`` is in eV/atom, so the explicitly supplied
factor is the TOTAL atom count, 1,008,000, not the 604,800 substitutional sites.
The temperature is in K and the supplied k_B is in eV/K. If model units,
normalization, or supercell size change, the user must recalculate this factor.

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
    energy_factor = 1008000.0  # USER conversion: eV/atom -> total supercell eV
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
- ``energy_factor`` and ``boltzmann_constant``: the user-supplied numerical
  conversion and k_B, without an inferred unit label.
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
The single-run filename behavior is unchanged. MonteCarloLite sampling tasks
are discontinued; use the maintained ``runmc`` task.

The same options are available directly through plac. Structured values use
JSON, and boolean options take explicit ``true``/``false`` values, for example::

    cell metropolis -task runmc -mofp model.pickle -plfp plat.json \
        -scsh '[8,1,1]' -nsubs '{"0":[4]}' \
        -runs '[{"temperature":300,"n_mc_sweeps":2},{"temperature":250,"n_mc_steps":100}]' \
        -ef 8.0 -k_B 8.617330337217213e-5 \
        -serialize false -nonlinear true -batch 100000 \
        -endpoints true -atoms anneal.xyz -atoms_mode x -rs 42

This direct-command example assumes a one-atom parent cell, hence eight atoms
in that supercell, and model energies in eV/atom. Supply the factor appropriate
to the actual model and supercell used.

Configurational free energy of mixing
------------------------------------

``compute_free_energy_of_mixing`` calculates binary canonical mixing free
energies by thermodynamic integration. It holds the supplied CE fixed throughout
each integration and uses the maintained ``MonteCarlo`` sampler. Affine and nonlinear models can
use local swap predictions, including ``reduce_structure=True``. One binary
substitutional site type is supported; any spectator sites remain pristine.
Species A and B are the first and second entries in that site's ordered
allowed-species list (sigmas 0 and 1).

For N binary substitutional sites and NB = x*N B atoms, define
beta = 1/(k_B*T). The calculation evaluates:

.. math::

    \beta\Delta f_{\mathrm{mix}}(x,\beta)
    = -\frac{\ln\binom{N}{N_B}}{N}
      + \int_0^\beta\left[
          \frac{\langle E_{\mathrm{total}}\rangle_{x,\beta'}}{N}
          -(1-x)e_A-xe_B\right]d\beta'.

The endmember energies e_A and e_B are predicted for this same supercell and
divided by N. Every result is **per binary substitutional site**, including
the mean energy; N does not include spectator atoms. The finite-cell entropy
uses the binomial coefficient, approaching
x*ln(x) + (1-x)*ln(1-x) for beta*delta_f in the thermodynamic limit.
Pure compositions return zero mixing free energy without proposing swaps.
Only configurational contributions represented by the CE are included.

The user must provide ``energy_factor`` converting the model prediction to
**total supercell energy**, and ``boltzmann_constant`` in compatible units.
For example, for a model in eV per atom and T in kelvin, supply
``energy_factor=float(len(scell))`` and k_B in eV/K. The atom count in this
conversion can differ from N when spectators are present. Endmember and
mixed configurations must use the same normalization.

**Fixed along the integration path** does not exclude a CE fitted to effective
configuration free energies at a chosen physical temperature. For example, a
CE representing configuration free energies at 300 K is held fixed while the
auxiliary sampling beta varies from zero to 1/(k_B*300 K). The result at that
endpoint is the mixing free energy for the supplied 300 K model. Intermediate
sampling temperatures are integration parameters; the model is not updated to
represent the material at those physical temperatures.

The entire ``temperatures`` list or range uses the same CE, including the
endmember and beta=0 evaluations. If the physical model depends on temperature,
repeat the complete integration separately for each target temperature using
its corresponding CE held fixed. The helper does not select or refit models
as the sampling temperature changes. Following a changing model along the path
would require temperature-derivative contributions to the integrand and a
consistent reference treatment, which this implementation does not provide.

Temperatures must be finite, positive and distinct. Either input ordering is
accepted; sampling and results are ordered from highest T to lowest T,
equivalently increasing beta. Compositions must correspond to integer NB for
the chosen supercell and are not silently rounded. Include sufficiently high
temperatures and refine the grid, especially across transitions: the code uses
composite trapezoidal quadrature in beta and does not estimate quadrature bias.
For context, see the `icet canonical free-energy discussion
<https://icet.materialsmodeling.org/advanced_topics/free_energy_canonical_ensemble.html>`_.

The beta=0 integration node is always included. Its mean energy is estimated
from ``n_reference_samples`` independent uniform fixed-composition decorations,
evaluated with the actual CE. This finite-cell reference is not obtained by
substituting independent-site random correlations into a nonlinear model, or
by extrapolating positive-temperature samples. At each positive temperature,
``n_mc_eq`` proposals are discarded before ``n_mc_steps`` production proposals.
Energy observations occur every ``sample_interval`` production proposals and
include rejected moves. The same chain continues across temperatures without
target-rate adaptation or rebuilding valid nonlinear correlation caches.

Python example (assuming ``model`` predicts eV/atom and ``scell`` has a binary
site count divisible by four; the same model is used at all sampling nodes)::

    from ase.units import kB
    from clusterx.thermodynamics.monte_carlo import compute_free_energy_of_mixing

    result = compute_free_energy_of_mixing(
        model, scell, compositions=[0.0, 0.25, 0.5, 0.75, 1.0],
        temperatures=[4000.0, 2000.0, 1000.0, 750.0, 500.0, 300.0],
        energy_factor=float(len(scell)), boltzmann_constant=kB,
        n_mc_eq=10_000, n_mc_steps=100_000, sample_interval=10,
        n_reference_samples=2000, n_blocks=20, random_seed=42,
        predict_swap=True, reduce_structure=True, nonlinear=True,
        max_steps_per_batch=20_000, filename="mixing-free-energy.json",
    )

If ``model`` describes effective configuration free energies at 300 K, the
300 K result in this example is the physical target; the higher-temperature
nodes provide the auxiliary integration path for that same fixed model.

A complete named-step input, executed as ``cell --input_file mixing.toml``,
is shown below. It assumes a **one-atom binary parent cell**, hence 64 atoms
for this shape, and a saved model in eV/atom. Change the factor and shape for
your actual model; a parent cell containing spectators needs a different factor::

    do = ["mixing"]

    [mixing]
    command = "compute_free_energy_of_mixing"
    model_filepath = "model.pickle"
    plat_filepath = "plat.json"
    sc_shape = [4, 4, 4]
    compositions = [0.0, 0.25, 0.5, 0.75, 1.0]
    temperatures = [4000.0, 2000.0, 1000.0, 750.0, 500.0, 300.0]
    energy_factor = 64.0
    boltzmann_constant = 8.617330337217213e-5
    n_mc_eq = 10000
    n_mc_steps = 100000
    sample_interval = 10
    n_reference_samples = 2000
    n_blocks = 20
    random_seed = 42
    predict_swap = true
    reduce_structure = true
    nonlinear = true
    max_steps_per_batch = 20000
    output_filepath = "mixing-free-energy.json"
    overwrite = false

Instead of ``temperatures``, supply all three of ``temperature_start``,
``temperature_stop`` and ``n_temperatures`` for a linearly spaced temperature
range. These alternatives are mutually exclusive. Direct CLI invocation uses
JSON lists, for example::

    cell compute_free_energy_of_mixing -mofp model.pickle -plfp plat.json \
        -scsh '[4,4,4]' -x '[0.25,0.5,0.75]' -temps '[4000,2000,1000,500]' \
        -ef 64 -k_B 8.617330337217213e-5 -n_eq 10000 -n_steps 100000 \
        -sample_every 10 -n_ref 2000 -blocks 20 -rs 42 \
        -pred_swap true -reduce true -nonlinear true -out mixing-free-energy.json

The returned dictionary and JSON file have ``settings``, ``endmembers``,
``references`` (the beta=0 nodes), and ``results``. Each result includes
``composition``, ``n_b``, ``temperature``, ``beta``, ``mean_energy_per_site``,
``mean_mixing_energy_per_site``, ``free_energy_of_mixing_per_site``, and standard
errors in the corresponding ``*_stderr`` fields. Species, conversion inputs,
site counts, random seeds, proposal counts and quadrature are recorded. An
existing destination requires explicit ``overwrite=true``. Input model files
are not rewritten while warming geometry caches.
Fields named ``mean_energy_per_site`` refer to the scaled quantity predicted
by the supplied CE. For an effective configuration free-energy CE, they are
mean effective free energies, rather than thermodynamic internal energies.

Production standard errors use ``n_blocks`` consecutive block means; there
must be at least two observations per block. Blocks must be long enough to
resolve autocorrelation. The beta=0 error uses independent-sample statistics.
Free-energy errors propagate these node errors with the quadrature weights,
neglecting correlations between temperatures; they do not include inadequate
equilibration, quadrature, finite-size or model errors. Check convergence in
all these controls, particularly near phase transitions.

Energy statistics keep only online moments and block averages. Accepted-move
histories are not serialized, and temporary proposal arrays respect
``max_steps_per_batch``. In the lower-level API, the optional
``metropolis(..., energy_observer=callback, observation_interval=10)`` calls
``callback(step, total_energy)`` after the requested proposal intervals,
including rejections. Energy includes the absolute model intercept and the
user's factor even for reduced MC. Steps are local to that call, no initial
observation is made, and the callback must not mutate the chain.
