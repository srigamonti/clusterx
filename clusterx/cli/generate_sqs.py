import plac
import numpy as np

from clusterx.cli.config_utils import cmd_message
from clusterx.clusters.clusters_pool import ClustersPool
from clusterx.correlations import CorrelationsCalculator
from clusterx.parent_lattice import ParentLattice
from clusterx.super_cell import SuperCell
from clusterx.model import Model

commands = ["generate_sqs"]


def objective_function(
    correlations_delta: np.array,
    correlations_tolerance: float,
    radius_weight: float,
    cpool: ClustersPool,
) -> float:
    """Equation 3 in A. van de Walle, et al. (2013)
    http://dx.doi.org/10.1016/j.calphad.2013.06.006"""
    radii = cpool.get_all_radii()
    below_tolerance = np.abs(correlations_delta) < correlations_tolerance
    if np.any(below_tolerance):
        max_L = radii[below_tolerance].max()
    else:
        max_L = 0.0
    return -radius_weight * max_L + np.sum(np.abs(correlations_delta))


def run_annealing(
    parent_lattice: ParentLattice,
    ccalc: CorrelationsCalculator,
    shape,
    nsubs,
    rng: np.random.Generator = np.random.default_rng(),
):
    cpool = ccalc.get_cpool()
    model = Model(ccalc, ecis=np.ones(len(cpool)))
    scell = SuperCell(parent_lattice, p=shape)
    structure = scell.gen_random_structure(nsubs=nsubs, rng=rng, track_indices=True)
    print(structure.sigmas)

    tolerance = 1e-3
    radius_weight = 0.1

    correlations_target = np.zeros(len(cpool))
    correlations = ccalc.get_cluster_correlations(structure)
    correlations_delta = correlations_target - correlations
    cost_old = objective_function(correlations_delta, tolerance, radius_weight, cpool)

    beta = 400
    accept = True

    for i in range(1000):
        print(i, cost_old, accept)
        i, j, site_type = structure.get_random_swap(None)
        sigmas = structure.sigmas
        sigma_i = sigmas[i]
        sigma_j = sigmas[j]
        correlations_diff = model.predict_correlations_diff(
            structure, i, sigmas[i], sigmas[j]
        )
        sigmas[i] = sigma_j
        correlations_diff += model.predict_correlations_diff(
            structure, j, sigmas[i], sigmas[j]
        )
        sigmas[i] = sigma_i
        correlations += correlations_diff
        correlations_delta = correlations_target - correlations
        cost_new = objective_function(
            correlations_delta, tolerance, radius_weight, cpool
        )

        cost_delta = cost_new - cost_old

        if cost_delta < 0 or rng.random() < np.exp(-cost_delta * beta):
            sigmas[i], sigmas[j] = sigmas[j], sigmas[i]
            cost_old = cost_new
            accept = True
        else:
            correlations -= correlations_diff
            accept = False

    correlations = ccalc.get_cluster_correlations(structure)
    correlations_delta = correlations_target - correlations
    cost_new = objective_function(correlations_delta, tolerance, radius_weight, cpool)
    print("Final correlations: ", correlations)
    print("Final cost: ", cost_new)
    return structure


def generate_sqs(
    plat_filepath: str = "plat.json",
    cpool_filepath: str = "cpool.json",
    ccalc_filepath: str = "ccalc.pickle",
):
    cmd_message("head")
