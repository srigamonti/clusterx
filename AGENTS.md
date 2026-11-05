# Working on CELL / clusterx

This file applies to the whole repository. Read any more specific `AGENTS.md` files
in directories you change, and inspect the working-tree diff before starting.

See [WISHLIST.md](WISHLIST.md) for possible improvements and open design questions.
Entries are proposals, not active tasks; work on them when included in the current
task scope.

## Project context

CELL is the scientific Python package distributed as `clusterX` and imported as
`clusterx`. It implements cluster expansion (CE), including advanced model
construction, nonlinear CE, derivative-structure generation, and thermodynamics.
It supports complex substitutional materials with multiple components and
sublattices, including systems with partial periodicity.

Development began in 2015 and first commit was on February 2018. 
Most of the code was written by hand before LLM assistance became common. 
Treat it as established scientific software with
accumulated domain knowledge and implicit conventions. The aim of agent-assisted
work is to improve correctness, clarity, maintainability, performance, and usability
while preserving the scientific meaning of existing workflows.

The main scientific references are:

- Rigamonti et al., *CELL: a Python package for cluster expansion with a focus on
  complex alloys*, npj Computational Materials **10**, 195 (2024).
  [Article](https://www.nature.com/articles/s41524-024-01363-x),
  DOI: `10.1038/s41524-024-01363-x`.
- Stroth, Draxl, and Rigamonti, *Cluster expansion toward nonlinear modeling and
  classification*, Physical Review Research **7**, 033091 (2025).
  [Article PDF](https://journals.aps.org/prresearch/pdf/10.1103/68lv-86k6),
  DOI: `10.1103/68lv-86k6`.

Use these papers for the methodology and the implementation and tests for current
behavior. If they appear inconsistent, explain the discrepancy before changing
scientific assumptions. Project documentation is linked from `README.rst`.

## Core domain model

The declared inheritance chain is:

```text
ase.Atoms -> ParentLattice -> SuperCell -> Structure
```

- `ParentLattice` augments the lattice geometry with the species allowed at each
  crystal site, site types/sublattices, and spectator sites that cannot substitute.
- `SuperCell` repeats a parent lattice using an integer transformation matrix and
  retains its allowed-species and site-type information.
- `Structure` represents a particular decoration of a supercell: one allowed
  species is selected at each site. `sigmas` encode indices into each site's
  ordered list of allowed species; they are distinct from atomic numbers.

The implementation mixes inheritance with composition. In particular, `Structure`
stores a `scell` object and delegates missing attributes through `__getattr__`;
parent lattices also retain internal ASE objects. Inspect initialization,
delegation, copying, mutation, and serialization before relying on inherited
`Atoms` behavior. Updating one representation may leave another or a cache stale.

## Repository map

| Area | Main locations and responsibilities |
| --- | --- |
| Lattice and configurations | `clusterx/parent_lattice.py`, `super_cell.py`, `structure.py` |
| Structure collections | `clusterx/structures_set.py`: structures, properties, and persistence |
| Cluster construction | `clusterx/clusters/cluster.py`, `clusterx/clusters/clusters_pool.py`: `Cluster`, `ClustersPool`, `ClusterOrbit` |
| Symmetry and geometry | `clusterx/symmetry.py`, `clusterx/utils.py` |
| Correlations and models | `clusterx/correlations.py`, `clusterx/model.py`, `clusterx/estimators/` |
| Selection and training sets | `clusterx/clusters_selector.py`, `clusterx/structure_selector.py` |
| Derivative structures | `clusterx/derivative_structures.py`: `DerivativeStructures`, HNF-based supercells, configuration generation, symmetry reduction, degeneracies, masks, and properties |
| Thermodynamics | `clusterx/thermodynamics/`: Monte Carlo, MonteCarloLite, Wang-Landau, and density-of-states analysis |
| CLI | `clusterx/cli/`: command implementations, configuration handling, model-building pipelines, and reports |
| NOMAD / NOMAD OASIS | `clusterx/dbinterfaces/` and corresponding CLI commands |
| Other interfaces | `clusterx/io/`, `clusterx/calculators/`, and visualization modules |
| Tests and tooling | `clusterx/test/`, `setup.py`, `ruff.toml`, `.gitlab-ci.yml` |

`DSGenerator` remains a compatibility alias for `DerivativeStructures`. Preserve
it for existing Python imports and pickle files saved under the former name.
Artifact reports use the `print_info_*` commands. `print_info_derivative_structures`
(also `print_info_dss`) reports derivative structures; preserve the older
`summarize_derivative_structures` and `summarize_dsgenerator` commands and modules
as compatibility aliases. Likewise preserve `print_plat_info` and
`print_cpool_info` as aliases of `print_info_plat` and `print_info_cpool`.

## Scientific conventions to preserve

- Preserve site ordering, sublattice tags, and the order of allowed species.
  Reordering these can change the meaning of `sigmas` and existing saved data.
  Vacancies are normally represented by atomic number `0` (ASE symbol `X`);
  respect explicit vacancy-number options in file interfaces.
- Handle spectator sites and multiple sublattices explicitly. Do not infer
  crystallographic equivalence solely from a shared site-type tag.
- Respect periodic boundary conditions in each direction, including 1D and 2D
  systems. Supercells can use non-diagonal integer matrices. In `SuperCell`, cell
  vectors are rows and the transformation convention is `S = P V`.
- Preserve symmetry tolerances, cluster equivalence, orbit multiplicities, basis
  conventions, and correlation normalization unless the task changes them
  deliberately. Cluster order must stay aligned with correlation columns and
  model coefficients, including after selection or pruning.
- Check coefficient conventions at each API boundary. In particular, the
  `Model(ecis=...)` interface expects coefficients already multiplied by cluster
  multiplicities. Avoid applying multiplicities twice.
- Nonlinear CE augments the correlation features, for example with polynomial
  products. It can remain linear in its fitted coefficients. Distinguish this
  feature transformation from the choice of site-function basis. Preserve the
  fitted pipeline's transformations, scaling, selection, feature order, and
  intercept when predicting, pruning, or saving/loading models.
- For derivative structures, verify completeness, symmetry uniqueness,
  composition constraints, and degeneracy counts on small systems. Compare
  against exact enumeration or combinatorial counts where practical.
- For thermodynamics, preserve ensemble constraints, sampling probabilities,
  energy units, and normalization per atom, parent cell, or supercell as
  appropriate. Validate optimized local energy changes against full model
  evaluations, including nonlinear models when supported by the affected path.

## How to make changes

- Read the relevant implementation, its callers, and nearby tests before editing.
  Preserve user changes. For review-only requests, report findings without edits.
- Prefer focused, incremental improvements. Keep mechanical cleanup separate
  from changes to algorithms or scientific behavior. Legacy code or a stale
  comment alone does not justify a broad rewrite.
- Preserve public Python imports, CLI options, and persisted data formats unless
  changing them is part of the task. Account for ASE databases, JSON, NPZ, and
  pickle files used by the affected workflow. Moving classes can break old
  pickles even when the Python API otherwise looks equivalent.
- When scientific intent is ambiguous, state the assumption or ask the
  maintainer. Routine implementation choices can proceed within the task scope.
  Never silently change equations, units, normalization, or expected scientific
  results to make tests pass.
- For performance work, measure a representative case and verify numerical
  equivalence. Consider combinatorial scaling and memory use in cluster
  generation, enumeration, correlation evaluation, and sampling. Define cache
  invalidation when adding or changing cached state.
- Follow surrounding style and `ruff.toml`: Python 3.10-compatible syntax,
  four-space indentation, double quotes, and a 120-character line length.
  Keep existing license notices. Avoid unrelated reformatting and dependency
  changes; explain any new dependency that is necessary.

## CLI and NOMAD development

These interfaces are actively growing. Inspect their current code and tests
instead of assuming their APIs are fixed or fully described by the papers.

- The console entry point is `cell=clusterx.cli.main:main`. CLI commands use
  `plac`, module-level `commands` lists, and dynamic discovery in
  `clusterx/cli/commands.py`. Check that discovery logic before adding helper
  modules. Use existing `plac.annotations` patterns for argument conversion.
- Exercise both direct command invocation and TOML configuration when changing
  shared dispatch or configuration behavior. Preserve support for repeated
  commands and custom command modules where relevant.
- TOML named steps use unique single-bracket table names and a `command` selector
  to support interleaving. Preserve legacy command tables/arrays and `do` ordering;
  legacy array entries execute as a group.
- Prefer shared Python logic with thin CLI wrappers. Inspect helpers in
  `config_utils.py`, `report_utils.py`, and existing model-building commands
  before introducing another implementation.
- NOMAD and NOMAD OASIS can use different base URLs and authentication settings.
  Keep endpoints configurable and preserve structure/property metadata and units
  during conversion. Mock HTTP and authentication in normal tests; they should
  work without live services or real credentials. Keep passwords, tokens, and
  local credential files out of source control and diagnostic output.

## Environment and validation

`setup.py` declares Python >=3.10 and the dependencies. CI currently uses Python
3.12, an editable install, and `cell test`. Reuse an appropriate existing virtual
environment; activate it so CLI subprocess tests find the matching `cell`.
For a new environment, install the package with `python -m pip install -e .`.

Useful commands from the repository root, with that environment active:

```bash
python -m pytest -q clusterx/test/test_parent_lattice.py
cell test -l
cell test -n test_parent_lattice
cell test
```

Select tests for the code changed and its affected consumers. For shared lattice,
symmetry, correlation, or model changes, include dependent cluster-pool,
derivative-structure, and thermodynamics checks as appropriate. Use the full CI
command for broad changes when feasible; describe any checks not run and why.
If Ruff is available, run `ruff check` on changed Python files and distinguish
existing violations from new ones.

Package imports and CLI startup must not read or write the legacy
`cellconfig.json` file. Several legacy tests write files with fixed names in the
working directory, and `cell test` writes `testlog.xml` and `testlog.html`. Run artifact-producing checks
in a disposable working directory with absolute test paths and the editable
installation available. For new file-producing tests, use `tmp_path` and change
the working directory with `monkeypatch.chdir` when needed.

For scientific or behavioral changes, add a regression check that can detect the
original problem. Prefer small, analytically checkable systems, explicit numerical
tolerances, and controlled random seeds. Do not replace physical validation with
tests that merely repeat the implementation. Documentation-only edits need no
scientific test run.

At handoff, state what changed, the relevant scientific or compatibility effects,
the checks actually run, and any unresolved limitations.
