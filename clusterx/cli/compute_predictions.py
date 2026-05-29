# Copyright (c) CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.
import os
import pickle
from typing import Optional

import numpy as np
import plac

from clusterx.cli.config_utils import cmd_message
from clusterx.derivative_structures import DSGenerator
from clusterx.model import Model
from clusterx.parent_lattice import ParentLattice
from clusterx.structures_set import StructuresSet

commands = ["compute_predictions"]


@plac.pos(
    "target_filepath",
    help="Path to a serialized StructuresSet or DSGenerator object.",
)
@plac.pos(
    "property_name",
    help="Name of the property where predictions are stored.",
)
@plac.pos(
    "model_filepath",
    help="Path to a serialized CE model.",
)
@plac.opt(
    "plat_filepath",
    abbrev="plf",
    help="Optional path to a serialized ParentLattice object used for validation.",
    type=str,
)
@plac.opt(
    "store_ensemble_predictions",
    abbrev="sep",
    help="Whether to store ensemble predictions. Only applicable if the model is an ensemble. Default: False.",
    type=bool,
)
def compute_predictions(
    target_filepath: str,
    property_name: str,
    model_filepath: str,
    plat_filepath: Optional[str] = None,
    store_ensemble_predictions: bool = False,
):
    """Compute CE-model predictions and store them in the serialized target."""
    cmd_message("head")

    target_kind, target = _load_target(target_filepath)
    model = _load_model(model_filepath)

    model_property_name = getattr(model, "property_name", None)
    if property_name != model_property_name:
        print(
            "Info(compute_predictions): storing predictions as "
            f"'{property_name}', while the CE model property_name is "
            f"'{model_property_name}'."
        )

    if plat_filepath is not None:
        _validate_parent_lattice(plat_filepath, target_kind, target, model)

    if target_kind == "dsgenerator":
        target.compute_properties(
            property_name=property_name, cemodel=model, store_ensemble_predictions=store_ensemble_predictions
        )
        target.serialize(filepath=target_filepath)
    elif target_kind == "structures_set":
        values = [_prediction_to_jsonable(model.predict(structure)) for structure in target]
        target.set_property_values(
            property_name=property_name,
            property_vals=values,
            update_db=False,
        )
        target.serialize(filepath=target_filepath, overwrite=True)
    else:
        raise ValueError(f"Unknown target kind: {target_kind}")

    print(
        "Info(compute_predictions): stored predictions in "
        f"'{property_name}' for {_target_count(target_kind, target)} "
        "structures/configurations."
    )


def _load_target(filepath: str):
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Target file not found: {filepath}")

    pickle_error = None
    try:
        with open(filepath, "rb") as f:
            obj = pickle.load(f)
    except Exception as err:
        pickle_error = err
    else:
        if isinstance(obj, DSGenerator):
            return "dsgenerator", obj

    try:
        return "structures_set", StructuresSet(filepath=filepath)
    except Exception as err:
        raise ValueError(
            f"Could not load '{filepath}' as a serialized DSGenerator or "
            f"StructuresSet. Pickle error: {pickle_error}. "
            f"StructuresSet error: {err}"
        ) from err


def _load_model(filepath: str):
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")

    model = Model(filepath=filepath)
    if hasattr(model, "corrc") and hasattr(model, "property_name"):
        return model

    with open(filepath, "rb") as f:
        model = pickle.load(f)
    if not isinstance(model, Model):
        raise TypeError(f"Expected serialized Model, got {type(model).__name__}.")
    return model


def _validate_parent_lattice(plat_filepath: str, target_kind: str, target, model):
    plat = ParentLattice(filepath=plat_filepath)

    if target_kind == "dsgenerator":
        target_plat = target.plat
    else:
        target_plat = target.get_parent_lattice()

    if target_plat != plat:
        raise ValueError(f"Parent lattice from target file does not match '{plat_filepath}'.")

    if model.get_parent_lattice() != plat:
        raise ValueError(f"Parent lattice from CE model does not match '{plat_filepath}'.")


def _target_count(target_kind: str, target):
    if target_kind == "dsgenerator":
        return len(target.configurations)
    return len(target)


def _prediction_to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value
