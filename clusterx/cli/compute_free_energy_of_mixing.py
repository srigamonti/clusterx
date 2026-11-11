# Copyright (c) 2015-2026, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.
"""Canonical Monte Carlo thermodynamic integration with explicit physical units."""

import json

import numpy as np
import plac

from clusterx.cli.config_utils import cmd_message
from clusterx.cli.metropolis import _boolean
from clusterx.model import Model
from clusterx.parent_lattice import ParentLattice
from clusterx.super_cell import SuperCell
from clusterx.thermodynamics.monte_carlo import (
    _mc_count,
    _positive_scalar,
    compute_free_energy_of_mixing as _compute_free_energy_of_mixing,
)

commands = ["compute_free_energy_of_mixing"]


@plac.annotations(
    model_filepath=("Saved CE model", "option", "mofp", str),
    plat_filepath=("Saved parent lattice", "option", "plfp", str),
    sc_shape=("Supercell shape as JSON integer/vector/matrix", "option", "scsh", json.loads),
    compositions=("B fractions as a JSON number or list", "option", "x", json.loads),
    temperatures=("Positive distinct temperatures as a JSON number or list", "option", "temps", json.loads),
    temperature_start=("First endpoint of a linear temperature range", "option", "t_start", float),
    temperature_stop=("Last endpoint of a linear temperature range", "option", "t_stop", float),
    n_temperatures=("Number of points in the temperature range", "option", "nt", int),
    energy_factor=("Required multiplier converting predictions to TOTAL supercell energy", "option", "ef", float),
    boltzmann_constant=("Required k_B consistent with scaled energy and temperature units", "option", "k_B", float),
    n_mc_steps=("Production proposals per composition/temperature", "option", "n_steps", int),
    n_mc_eq=("Discarded equilibration proposals at each temperature", "option", "n_eq", int),
    sample_interval=("Production energy observation interval in proposals", "option", "sample_every", int),
    n_reference_samples=("Independent fixed-composition beta=0 decorations", "option", "n_ref", int),
    n_blocks=("Blocks for approximate production standard errors", "option", "blocks", int),
    random_seed=("Seed for independent reference and MC random streams", "option", "rs", int),
    predict_swap=("Use local energy/correlation changes", "option", "pred_swap", _boolean),
    reduce_structure=("Use reduced local geometry", "option", "reduce", _boolean),
    nonlinear=("Support nonlinear fitted transformations", "option", "nonlinear", _boolean),
    energy_key=("Energy property of a multi-output model", "option", "energy_key", str),
    max_steps_per_batch=("Maximum temporary MC temperature-array size", "option", "batch", int),
    output_filepath=("JSON result destination", "option", "out", str),
    overwrite=("Explicitly replace an existing result file", "option", "overwrite", _boolean),
)
def compute_free_energy_of_mixing(
    model_filepath: str = "model.pickle",
    plat_filepath: str = "plat.json",
    sc_shape=1,
    compositions=None,
    temperatures=None,
    temperature_start: float | None = None,
    temperature_stop: float | None = None,
    n_temperatures: int | None = None,
    energy_factor: float | None = None,
    boltzmann_constant: float | None = None,
    n_mc_steps: int = 10_000,
    n_mc_eq: int = 1_000,
    sample_interval: int = 1,
    n_reference_samples: int = 1_000,
    n_blocks: int = 10,
    random_seed: int | None = None,
    predict_swap: bool = True,
    reduce_structure: bool = False,
    nonlinear: bool = True,
    energy_key: str = "energy",
    max_steps_per_batch: int = 100_000,
    output_filepath: str = "free_energy_of_mixing.json",
    overwrite: bool = False,
):
    r"""Compute binary configurational mixing free energy with canonical Monte Carlo.

    Use one saved cluster-expansion (CE) model, held fixed along the integration
    path, and one binary substitutional site type, optionally with fixed spectators.
    Species A and B are the first and second entries in that site's ordered
    allowed-species list (sigmas 0 and 1). Each composition is sampled at fixed
    NB = x*N, where N counts binary substitutional sites, excluding spectators.
    Only configurational contributions represented by the CE are included.

    Method and normalization
    ------------------------
    With beta = 1/(boltzmann_constant*T), integrate the mean mixing energy:

        beta*delta_f_mix = -log(binomial(N, NB))/N
                          + integral_0^beta [<E_total>/N - (1-x)*e_A - x*e_B] d beta'

    The endmember energies e_A and e_B are evaluated for this same supercell
    and divided by N. Results are per binary substitutional site, in the
    user-chosen energy unit. Pure compositions give zero mixing free energy.
    The finite-cell binomial entropy is used, rather than its infinite-size
    approximation. Free energy includes this entropy and the energy integral;
    it is not simply the sampled mean energy.

    Composite trapezoidal quadrature includes beta=0 explicitly. Its mean
    energy is estimated from independent uniform fixed-composition decorations
    evaluated with the actual model, also for nonlinear CE. It is not inferred
    from the highest simulated temperature. Refine the temperature grid,
    including the interval from beta=0, to check integration convergence.

    Fixed CE and physical temperature
    ---------------------------------
    The CE may represent configuration energies or effective configuration free
    energies at a chosen physical temperature. Only its parameters along the
    integration path must remain fixed. For example, a CE fitted to effective
    configuration free energies at 300 K is held fixed while the auxiliary
    sampling beta varies from zero to 1/(k_B*300 K). The endpoint gives the
    mixing free energy for that 300 K model. Intermediate sampling temperatures
    are integration parameters; the model still represents the 300 K system.

    Every entry in temperatures, or in a temperature range, uses the same CE,
    including the endmember and beta=0 predictions. For a family of models that
    depends on physical temperature, repeat the complete integration separately
    for each target temperature with its corresponding CE held fixed. The
    command does not update, select or refit models as sampling temperature
    changes. A path with changing model parameters would require additional
    temperature-derivative terms and a consistent reference treatment, which
    are not implemented here.

    Required inputs and units
    -------------------------
    model_filepath, plat_filepath, sc_shape:
        Saved model (default model.pickle), parent lattice (default plat.json),
        and supercell shape (default 1). Direct CLI shapes use JSON, e.g.
        -scsh '[4,4,4]'; TOML accepts an integer, vector or integer matrix.
    compositions:
        Required B fraction or list of distinct fractions in [0, 1], e.g.
        -x '[0.25,0.5,0.75]'. Each x*N must be an integer; compositions are not
        silently rounded to fit the cell.
    temperatures:
        Required positive distinct sampling temperature or list, unless a range is
        supplied. Any input ordering is accepted. Sampling and output proceed
        from highest to lowest T, equivalently increasing beta. Alternatively
        provide ALL of temperature_start, temperature_stop and n_temperatures
        (at least 2) for a linear temperature grid. Do not combine the two forms.
        All nodes use the fixed supplied CE; see the physical-temperature
        interpretation above when it models effective configuration free energy.
    energy_factor, boltzmann_constant:
        Both must be supplied as finite positive numbers. energy_factor
        multiplies model predictions to give TOTAL SUPERCELL energy in the
        same energy unit as boltzmann_constant*T. The user is responsible for
        both normalization and units; CELL infers neither.

        For model energies in eV/atom, T in K and k_B in eV/K, use the TOTAL
        atom count as energy_factor, including spectators. For meV/atom with
        k_B in eV/K, use that count divided by 1000. Already-total energies in
        the matching energy unit require an explicit factor 1. This conversion
        count can differ from the number N of binary sites used in the output.
        One factor must be valid for mixed and pure compositions alike.

    Sampling controls
    -----------------
    n_mc_eq (default 1000) discards equilibration proposals at EACH temperature.
    n_mc_steps (default 10000) is the number of subsequent production proposals
    per composition/temperature, not a sweep count. sample_interval (default 1)
    observes energy every specified number of production proposals, including
    rejected moves. Chains continue across temperatures with their cached
    nonlinear state; no target-rate temperature adaptation is used.

    n_reference_samples (default 1000, minimum 2) controls the independent beta=0
    samples for each mixed composition, separately from MC equilibration.
    n_blocks (default 10, minimum 2) controls consecutive production blocks for
    approximate standard errors. floor(n_mc_steps/sample_interval) must be at
    least 2*n_blocks. Choose blocks long enough to resolve autocorrelation.
    random_seed fixes the reference and MC random streams for reproducibility.

    predict_swap=true uses local energy/correlation changes. nonlinear=true
    supports nonlinear fitted transformations and retains the affine model's
    direct energy-difference path. reduce_structure=true uses local reduced
    geometry and requires diagonal supercells. max_steps_per_batch (default
    100000) bounds temporary proposal-temperature arrays; it does not change
    the chain or observation cadence. Online statistics and block means avoid
    storing full accepted-move histories. Direct CLI booleans take explicit
    true/false values. energy_key selects energy from a multi-output model.

    Examples
    --------
    These examples assume a ONE-ATOM binary parent cell and a saved CE model
    in eV/atom: a 4x4x4 supercell has 64 atoms, so energy_factor=64. T is in K
    and the supplied k_B is in eV/K. Change the factor for your actual model,
    normalization, energy unit and cell size. Proposal counts illustrate the
    syntax; their adequacy must be established by convergence checks.
    If the CE describes configuration free energies at 300 K, the 300 K result
    is the physical target and the higher-temperature nodes are auxiliary
    integration points for that same fixed model.

    Direct CLI invocation::

        cell compute_free_energy_of_mixing -mofp model.pickle -plfp plat.json \
            -scsh '[4,4,4]' -x '[0.0,0.25,0.5,0.75,1.0]' \
            -temps '[4000,2000,1000,750,500,300]' \
            -ef 64 -k_B 8.617330337217213e-5 -n_eq 10000 -n_steps 100000 \
            -sample_every 10 -n_ref 2000 -blocks 20 -rs 42 \
            -pred_swap true -reduce true -nonlinear true -batch 20000 \
            -out mixing-free-energy.json

    For a linear temperature range, replace -temps and its list with
    -t_start 4000 -t_stop 300 -nt 30. The quadrature still operates in beta;
    evenly spaced temperatures do not give evenly spaced beta nodes.

    Named TOML step, saved as mixing.toml and executed with
    ``cell --input_file mixing.toml``::

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

    Output and uncertainty
    ----------------------
    output_filepath defaults to free_energy_of_mixing.json. Existing results
    require explicit overwrite=true (CLI: -overwrite true). Input model files
    are not rewritten while warming geometry caches.

    The JSON and returned dictionary contain settings, endmembers, references
    (beta=0 nodes), and results. Each result row includes composition, n_b,
    temperature, beta, mean_energy_per_site, mean_mixing_energy_per_site,
    free_energy_of_mixing_per_site, and the corresponding *_stderr estimates
    for mean energy and free energy. Settings record species, atom/site counts,
    conversion factors, sampling controls and seeds.
    E_total and fields named mean_energy_per_site denote the scaled quantity
    predicted by the supplied CE. For an effective configuration free-energy
    model, these are mean effective free energies, not internal energies.

    Production errors use block means; reference errors use independent-sample
    statistics. Free-energy errors propagate node errors with quadrature weights
    and neglect cross-temperature covariance. They exclude quadrature bias,
    inadequate equilibration, finite-size and model errors. Check convergence
    in the beta grid, reference sample count, equilibration, production length,
    block length and supercell size, particularly near phase transitions.
    """
    cmd_message("head")
    energy_factor = _positive_scalar(energy_factor, "energy_factor")
    boltzmann_constant = _positive_scalar(boltzmann_constant, "boltzmann_constant")
    range_supplied = any(value is not None for value in (temperature_start, temperature_stop, n_temperatures))
    if temperatures is not None and range_supplied:
        raise ValueError("Specify temperatures or a temperature range, not both")
    if range_supplied:
        temperature_start = _positive_scalar(temperature_start, "temperature_start")
        temperature_stop = _positive_scalar(temperature_stop, "temperature_stop")
        n_temperatures = _mc_count(n_temperatures, "n_temperatures", minimum=2)
        if temperature_start == temperature_stop:
            raise ValueError("Temperature range endpoints must differ")
        temperatures = np.linspace(temperature_start, temperature_stop, n_temperatures)
    elif temperatures is None:
        raise ValueError("Provide temperatures or temperature_start, temperature_stop, and n_temperatures")
    model = Model(filepath=model_filepath)
    plat = ParentLattice(filepath=plat_filepath)
    scell = SuperCell(plat, sc_shape)
    result = _compute_free_energy_of_mixing(
        model, scell, compositions, temperatures, energy_factor=energy_factor, boltzmann_constant=boltzmann_constant,
        n_mc_steps=n_mc_steps, n_mc_eq=n_mc_eq, sample_interval=sample_interval,
        n_reference_samples=n_reference_samples, n_blocks=n_blocks, random_seed=random_seed,
        predict_swap=predict_swap, reduce_structure=reduce_structure, nonlinear=nonlinear, energy_key=energy_key,
        max_steps_per_batch=max_steps_per_batch, filename=output_filepath, overwrite=overwrite,
    )
    print(f"Wrote {len(result['results'])} free-energy estimates to {output_filepath}")
    return result
