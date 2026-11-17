# Copyright (c) 2015-2024, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

import copy
import os
from pprint import pformat
from typing import Optional

import numpy as np
import plac
from sklearn.pipeline import Pipeline

from clusterx.cli.build_nonlinear_model import fit_nonlinear_pipeline
from clusterx.cli.config_utils import (
    cmd_message,
    convert_to_float_list,
    convert_to_int_list,
)
from clusterx.cli.report_utils import MarkdownReport
from clusterx.clusters.clusters_pool import ClustersPool
from clusterx.correlations import CorrelationsCalculator
from clusterx.estimators.column_expander import ColumnExpander
from clusterx.model import Model
from clusterx.parent_lattice import ParentLattice
from clusterx.structures_set import StructuresSet
from clusterx.super_cell import SuperCell

commands = ["build_model_from_sset"]


def _output_paths(prefix: str, prune: bool) -> dict:
    """Output file names.

    Names are ``<prefix>_<stage>_<stage name>_<kind>.<ext>``; the numeric stage
    makes an alphabetical listing show the log first, then all files of the
    full (before pruning) model, then all files of the pruned model.
    """
    paths = {"log": f"{prefix}_0_log.md"}
    stages = [("1", "full")] + ([("2", "pruned")] if prune else [])
    for num, name in stages:
        base = f"{prefix}_{num}_{name}"
        paths[name] = {
            "cpool": f"{base}_cpool.json",
            "ccalc": f"{base}_ccalc.pickle",
            "comat": f"{base}_comat.npz",
            "model": f"{base}_model.pickle",
        }
    return paths


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in ("false", "0", "no", "off", "")
    return bool(value)


@plac.annotations(
    property_name=(
        "Property to be modelled. Must be present in the StructuresSet object.",
        "positional",
        None,
        str,
    ),
    sset_filepath=(
        "Path to a serialized StructuresSet object (training data).",
        "option",
        "ssf",
        str,
    ),
    npoints=("List of numbers of points of the clusters to build.", "option", None, str),
    radii=("List of cluster radii, one per entry of npoints.", "option", None, str),
    plat_filepath=(
        "Path to a serialized ParentLattice. Default: the one of the structures set.",
        "option",
        "plf",
        str,
    ),
    psc=("Super cell definition where to find clusters.", "option", "p", list),
    method=("Method to use to find clusters.", "option", "me", int),
    vacancy_atomic_number=(
        "Atomic number of vacancy sites when serializing the clusters pool.",
        "option",
        None,
        int,
    ),
    basis_name=(
        "Cluster basis: indicator-binary, trigonometric, polynomial, or chebyshev.",
        "option",
        "b",
        str,
    ),
    regression_model=("Dictionary of estimator options.", "option", "rm", dict),
    selection_model=(
        "Dictionary of estimator options for feature selection model (e.g. LASSO).",
        "option",
        "sm",
        dict,
    ),
    selection_threshold=(
        "Threshold on feature importance (|coefficient|) of the selection model. "
        "The default 0.0 keeps all features.",
        "option",
        "thr",
        float,
    ),
    nonlinear_transformation=(
        "Dictionary of nonlinear transformation settings.",
        "option",
        "nt",
        dict,
    ),
    weights_filepath=(
        "Sample weights for fitting and evaluating the weighted MSE.",
        "option",
        None,
        str,
    ),
    standardize=("Standardize the input data.", "flag", "std", bool),
    test_size=(
        "Fraction of samples reserved as hold-out test set (no split if not given).",
        "option",
        "ts",
        float,
    ),
    random_state=("Random seed of the optional train/test split.", "option", "rs", int),
    shuffle=("Shuffle samples before the optional split.", "flag", "sh", bool),
    prune=(
        "Build a model on a pruned clusters pool containing only clusters used by the trained model.",
        "option",
        None,
        str,
    ),
    output_prefix=(
        "Prefix of all created files. Default: property_name.",
        "option",
        "op",
        str,
    ),
)
def build_model_from_sset(
    property_name: str,
    sset_filepath: str,
    npoints=None,  # List[int] or str
    radii=None,  # List[float] or str
    plat_filepath: Optional[str] = None,
    psc: Optional[list] = None,
    method: int = 0,
    vacancy_atomic_number: int = 0,
    basis_name: str = "trigonometric",
    regression_model: Optional[dict] = None,
    selection_model: Optional[dict] = None,
    selection_threshold: float = 0.0,
    selection_must_include: list[int] = [],
    nonlinear_transformation: Optional[dict] = None,
    weights_filepath: Optional[str] = None,
    standardize: bool = False,
    test_size: Optional[float] = None,
    random_state: int = 42,
    shuffle: bool = False,
    prune=True,
    output_prefix: Optional[str] = None,
):
    """
    Build a (nonlinear) cluster expansion model from a structures set in one step.

    Chains the steps of ``build_cpool``, ``build_ccalc``, ``compute_comat`` and
    ``build_nonlinear_model``. Unless ``prune`` is false, it then identifies the
    clusters whose correlations actually influence the trained model (taking
    the nonlinear transformation, selection model, ``selection_threshold`` and
    ``selection_must_include`` into account), builds a pruned clusters pool and
    correlations calculator with only those clusters, and serializes a second
    model that takes the pruned correlations as input. A `ColumnExpander` step
    at the start of its pipeline zero-pads the pruned correlations to the
    dimension expected by the trained steps. The prediction is exactly the same
    as that of the full model.

    Created files (``<prefix>`` is ``output_prefix`` or ``property_name``)::

        <prefix>_0_log.md                  Markdown log of the run
        <prefix>_1_full_{cpool.json, ccalc.pickle, comat.npz, model.pickle}
        <prefix>_2_pruned_{cpool.json, ccalc.pickle, comat.npz, model.pickle}

    Parameters
    ----------
    property_name : str
        Property to model. Must be present in the structures set.
    sset_filepath : str
        Serialized StructuresSet with the training data.
    npoints, radii : list or str
        Clusters to build: numbers of points and corresponding maximum radii.
    plat_filepath : str, optional
        Parent lattice file. Default: the parent lattice of the structures set.
    psc, method, vacancy_atomic_number
        As in ``build_cpool``.
    basis_name : str
        As in ``build_ccalc``.
    regression_model, selection_model, selection_must_include
        As in ``build_nonlinear_model``.
    nonlinear_transformation, weights_filepath, standardize, test_size
        As in ``build_nonlinear_model``.
    random_state, shuffle
        As in ``build_nonlinear_model``.
    selection_threshold : float, default 0.0
        As in ``build_nonlinear_model`` but with a default of 0.0, i.e. the
        selection model removes no feature (features with exactly zero
        importance are still dropped from the pruned pool if the final
        regressor gives them zero weight). Raise it to select features.
    prune : bool, default True
        Whether to build the pruned clusters pool, correlations calculator and model.
    output_prefix : str, optional
        Prefix of created files (may include a directory, which is created).
    """
    cmd_message("head")

    if npoints is None or radii is None:
        raise ValueError("Both npoints and radii must be provided.")
    if isinstance(npoints, str):
        npoints = convert_to_int_list(npoints)
    if isinstance(radii, str):
        radii = convert_to_float_list(radii)
    prune = _as_bool(prune)

    prefix = output_prefix if output_prefix else property_name
    if os.path.dirname(prefix):
        os.makedirs(os.path.dirname(prefix), exist_ok=True)
    paths = _output_paths(prefix, prune)

    report = MarkdownReport()
    try:
        _run(
            report,
            paths,
            prune,
            property_name=property_name,
            sset_filepath=sset_filepath,
            npoints=npoints,
            radii=radii,
            plat_filepath=plat_filepath,
            psc=psc,
            method=method,
            vacancy_atomic_number=vacancy_atomic_number,
            basis_name=basis_name,
            regression_model=regression_model,
            selection_model=selection_model,
            selection_threshold=selection_threshold,
            selection_must_include=selection_must_include,
            nonlinear_transformation=nonlinear_transformation,
            weights_filepath=weights_filepath,
            standardize=standardize,
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle,
        )
    finally:
        report.save(paths["log"])
        print(f"Info(build_model_from_sset): Log written to '{paths['log']}'")


def _run(report, paths, prune, **cfg):
    property_name = cfg["property_name"]

    report.title(f"build_model_from_sset: {property_name}")
    report.section("Configuration")
    report.code(pformat({**cfg, "prune": prune}, sort_dicts=False))

    # ------------------------------------------------------------------
    # Structures set, clusters pool and correlations calculator
    # ------------------------------------------------------------------
    sset = StructuresSet.from_file(cfg["sset_filepath"])
    if cfg["plat_filepath"] is not None:
        plat = ParentLattice(filepath=cfg["plat_filepath"])
    else:
        plat = sset.get_parent_lattice()

    report.section("Clusters pool")
    if cfg["psc"] is not None:
        scell = SuperCell(plat, p=cfg["psc"])
        cpool = ClustersPool(
            plat,
            npoints=cfg["npoints"],
            radii=cfg["radii"],
            super_cell=scell,
            method=cfg["method"],
        )
    else:
        cpool = ClustersPool(
            plat, npoints=cfg["npoints"], radii=cfg["radii"], method=cfg["method"]
        )
    report.capture(cpool.print_info)
    cpool.serialize(
        filepath=paths["full"]["cpool"],
        vacancy_atomic_number=cfg["vacancy_atomic_number"],
    )

    ccalc = CorrelationsCalculator(
        basis_name=cfg["basis_name"], parent_lattice=plat, clusters_pool=cpool
    )

    report.section("Correlations matrix")
    comat = ccalc.get_correlation_matrix(sset)
    pvals = np.asarray(sset.get_property_values(property_name=property_name))
    np.savez(paths["full"]["comat"], comat=comat, **{f"property_{property_name}": pvals})
    # serialize after computing correlations to keep the supercell orbits for reuse
    ccalc.serialize(filepath=paths["full"]["ccalc"])
    report.bullets(
        [
            f"Structures: {comat.shape[0]}",
            f"Clusters (columns): {comat.shape[1]}",
            f"Basis: {cfg['basis_name']}",
        ]
    )

    weights = None
    if cfg["weights_filepath"] is not None:
        weights = np.load(cfg["weights_filepath"])["weights"]
        if len(weights) != len(pvals):
            raise ValueError(
                f"weights has length {len(weights)}, but pvals has length {len(pvals)}."
            )

    # ------------------------------------------------------------------
    # Full model
    # ------------------------------------------------------------------
    report.section("Full model", 1)
    fit = fit_nonlinear_pipeline(
        comat,
        pvals,
        weights=weights,
        regression_model=cfg["regression_model"],
        selection_model=cfg["selection_model"],
        selection_threshold=cfg["selection_threshold"],
        selection_must_include=cfg["selection_must_include"],
        nonlinear_transformation=cfg["nonlinear_transformation"],
        standardize=cfg["standardize"],
        test_size=cfg["test_size"],
        random_state=cfg["random_state"],
        shuffle=cfg["shuffle"],
        report=report,
    )
    pipeline = fit["pipeline"]
    full_model = Model(corrc=ccalc, property_name=property_name, estimator=pipeline)
    full_model.serialize(paths["full"]["model"])

    if not prune:
        _report_files(report, paths, ["full"])
        return

    # ------------------------------------------------------------------
    # Pruned model
    # ------------------------------------------------------------------
    report.section("Pruned model", 1)
    kept = np.asarray(fit["used_columns"], dtype=int)  # sorted ascending
    if len(kept) == 0:
        raise RuntimeError(
            "The trained model uses no clusters at all; nothing to prune to."
        )

    subpool = cpool.get_subpool(cluster_indexes=kept.tolist())
    report.text(
        f"Pruned clusters pool: {len(subpool)} of {len(cpool)} clusters "
        f"(original indices: {kept.tolist()})."
    )
    report.capture(subpool.print_info)
    subpool.serialize(
        filepath=paths["pruned"]["cpool"],
        vacancy_atomic_number=cfg["vacancy_atomic_number"],
    )

    ccalc_pruned = CorrelationsCalculator(
        basis_name=cfg["basis_name"], parent_lattice=plat, clusters_pool=subpool
    )
    comat_pruned = ccalc_pruned.get_correlation_matrix(sset)
    np.savez(
        paths["pruned"]["comat"],
        comat=comat_pruned,
        **{f"property_{property_name}": pvals},
    )
    ccalc_pruned.serialize(filepath=paths["pruned"]["ccalc"])

    # The pruned pipeline starts with an adapter expanding the pruned
    # correlations to the dimension the trained steps were fitted with.
    pruned_pipeline = Pipeline(
        [
            ("expander", ColumnExpander(comat.shape[1], kept)),
            *copy.deepcopy(pipeline.steps),
        ]
    )
    pruned_model = Model(
        corrc=ccalc_pruned, property_name=property_name, estimator=pruned_pipeline
    )
    pruned_model.serialize(paths["pruned"]["model"])

    # Consistency checks between full and pruned models
    comat_diff = float(np.max(np.abs(comat_pruned - comat[:, kept])))
    preds_full = pipeline.predict(comat)
    preds_pruned = pruned_pipeline.predict(comat_pruned)
    pred_diff = float(np.max(np.abs(preds_full - preds_pruned)))
    structures = sset.get_structures()[:3]
    model_diff = max(
        abs(full_model.predict(s) - pruned_model.predict(s)) for s in structures
    )

    report.section("Consistency of pruned and full models")
    report.table(
        ["check", "max abs difference"],
        [
            ["pruned correlations vs. selected columns of full correlations", f"{comat_diff:.3e}"],
            ["predictions on all structures (pipelines)", f"{pred_diff:.3e}"],
            [f"Model.predict on first {len(structures)} structures", f"{model_diff:.3e}"],
        ],
    )
    tol = 1e-8
    if max(comat_diff, pred_diff, model_diff) > tol:
        report.text(
            f"**WARNING**: the pruned model deviates from the full model by more than {tol}."
        )
        print("WARNING: pruned model deviates from the full model. See the log.")

    _report_files(report, paths, ["full", "pruned"])


def _report_files(report, paths, stages):
    report.section("Created files")
    rows = [["log", paths["log"]]]
    for stage in stages:
        rows += [[f"{stage} {kind}", path] for kind, path in paths[stage].items()]
    report.table(["file", "path"], rows)
