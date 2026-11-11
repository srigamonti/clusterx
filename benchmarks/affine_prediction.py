"""Benchmark warmed affine MC prediction without modifying model artifacts.

Run from an installed checkout, for example:
python benchmarks/affine_prediction.py model.pickle --energy-factor 960 \
    --temperature 600 --boltzmann-constant 8.617330337217213e-5

The example assumes eV/atom predictions and a 960-atom supercell. Supply the
conversion factor, temperature, and k_B appropriate to the actual model units.
"""

import argparse
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import sklearn
from threadpoolctl import threadpool_info, threadpool_limits

from clusterx.model import Model
from clusterx.structure import Structure
from clusterx.super_cell import SuperCell
from clusterx.thermodynamics.monte_carlo import MonteCarlo


def run_case(filepath, steps, warmup, fast, seed, energy_factor, temperature, boltzmann_constant):
    model = Model(filepath=filepath)
    model.corrc.pickle_file = None  # Keep warmed orbit caches in memory; never write back to study artifacts.
    scell = SuperCell(model.get_parent_lattice(), [4, 4, 3])
    tags = scell.get_substitutional_tags()
    if len(tags) != 1 or len(scell.get_sublattice_types()[tags[0]]) != 2:
        raise ValueError("This benchmark expects a single binary substitutional sublattice.")
    active = np.asarray(scell.get_substitutional_sites())
    if len(active) % 2:
        raise ValueError("Half filling requires an even number of substitutional sites.")
    if model._get_affine_parameters(len(model.corrc.get_cpool())) is None:
        raise ValueError("The model is not eligible for affine prediction.")
    if not fast:
        pipeline = model.estimator
        offset = pipeline.predict(np.zeros((1, len(model.corrc.get_cpool()))))[0]
        model._get_affine_parameters = lambda width: None

        def reference_flip(structure, index, old_sigma, new_sigma, reduce_structure=False):
            delta = model.predict_correlations_diff(structure, index, old_sigma, new_sigma, reduce_structure)
            return pipeline.predict(delta.reshape(1, -1))[0] - offset

        model.predict_flip = reference_flip
    mc = MonteCarlo(model, scell, nsubs={int(tags[0]): [len(active) // 2]},
                    predict_swap=True, reduce_structure=True, seed=seed)
    options = dict(temperature=temperature, boltzmann_constant=boltzmann_constant,
                   energy_factor=energy_factor, serialize=False)
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        mc.metropolis(no_of_sampling_steps=warmup, **options)
        start = perf_counter()
        mc.metropolis(no_of_sampling_steps=steps, continue_chain=True, **options)
        elapsed = perf_counter() - start
    np.testing.assert_array_equal(np.bincount(mc.structure.sigmas[active], minlength=2), [len(active) // 2] * 2)
    conditions = dict(atoms=len(scell), substitutional_sites=len(active), concentration=0.5,
                      temperature=temperature, boltzmann_constant=boltzmann_constant,
                      energy_factor=energy_factor, predict_swap=True, reduce_structure=True)
    return (elapsed, mc.structure.sigmas.copy(), deepcopy(mc._rng_np.bit_generator.state),
            mc._rng_std.getstate(), conditions)


def numerical_checks(filepath, reference_path):
    model = Model(filepath=filepath)
    model.corrc.pickle_file = None
    width = len(model.corrc.get_cpool())
    rng = np.random.default_rng(503)
    x = rng.uniform(-1, 1, size=(1000, width))
    expected = model.estimator.predict(x)
    actual = np.array([model.predict_on_correlations(row) for row in x])
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
    result = {"max_fast_pipeline_error": float(np.max(np.abs(expected - actual)))}
    if reference_path is None:
        return result
    reference = Model(filepath=reference_path)
    reference.corrc.pickle_file = None
    scell = SuperCell(model.get_parent_lattice(), [4, 4, 3])
    active = np.asarray(scell.get_substitutional_sites())
    tag = int(scell.get_substitutional_tags()[0])
    structure = scell.gen_random_structure(nsubs={tag: [len(active) // 2]}, rng=rng)
    model.init_reduced_model(structure)
    structure.init_reduced_structure(model.scell_reduced)
    errors, prediction_errors = [], []
    for _ in range(32):
        i = rng.choice(active[structure.sigmas[active] == 0])
        j = rng.choice(active[structure.sigmas[active] == 1])
        changed_sigmas = structure.sigmas.copy()
        changed_sigmas[i], changed_sigmas[j] = changed_sigmas[j], changed_sigmas[i]
        changed = Structure(scell, sigmas=changed_sigmas)
        before = reference.estimator.predict(reference.corrc.get_cluster_correlations(structure).reshape(1, -1))[0]
        after = reference.estimator.predict(reference.corrc.get_cluster_correlations(changed).reshape(1, -1))[0]
        errors.append(abs(model.predict_swap(structure, i, j, reduce_structure=True) - (after - before)))
        prediction_errors.append(abs(model.predict(changed) - after))
    result.update(max_local_full_difference_error=float(max(errors)),
                  max_full_pruned_prediction_error=float(max(prediction_errors)))
    np.testing.assert_allclose(errors, 0, atol=1e-12)
    np.testing.assert_allclose(prediction_errors, 0, atol=1e-12)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=str)
    parser.add_argument("--reference-model", type=str)
    parser.add_argument("--energy-factor", type=float, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--boltzmann-constant", type=float, required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        checks = numerical_checks(args.model, args.reference_model)
        baseline, fast = [], []
        for _ in range(args.repeats):
            options = (args.energy_factor, args.temperature, args.boltzmann_constant)
            reference_run = run_case(args.model, args.steps, args.warmup, False, args.seed, *options)
            fast_run = run_case(args.model, args.steps, args.warmup, True, args.seed, *options)
            np.testing.assert_array_equal(reference_run[1], fast_run[1])
            np.testing.assert_equal(reference_run[2], fast_run[2])
            assert reference_run[3] == fast_run[3]
            baseline.append(reference_run[0])
            fast.append(fast_run[0])
        report = dict(model=args.model, shape=[4, 4, 3], steps=args.steps, warmup=args.warmup,
                      seed=args.seed, numerical_libraries=threadpool_info(), baseline_seconds=baseline,
                      fast_seconds=fast, speedup=float(np.median(baseline) / np.median(fast)),
                      baseline_median_seconds=float(np.median(baseline)), fast_median_seconds=float(np.median(fast)),
                      versions=dict(python=sys.version.split()[0], numpy=np.__version__, sklearn=sklearn.__version__),
                      identical_occupations_and_rng=True, **checks, **fast_run[4])
    output = json.dumps(report, indent=2)
    print(output)
    if args.output:
        args.output.write_text(output + "\n")


if __name__ == "__main__":
    main()
