# Copyright (c) 2015-2019, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

"""Tests for clusterx.model module."""

import pytest
from ase.build import bulk
import numpy as np
from sklearn.linear_model import LinearRegression

from clusterx.parent_lattice import ParentLattice
from clusterx.super_cell import SuperCell
from clusterx.model import Model
from clusterx.clusters.clusters_pool import ClustersPool
from clusterx.correlations import CorrelationsCalculator
from clusterx.structures_set import StructuresSet
from clusterx.calculators.emt import EMT2


@pytest.fixture
def plat():
    pri = bulk("H", crystalstructure="sc", a=1.0)
    sub = bulk("He", crystalstructure="sc", a=1.0)
    return ParentLattice(atoms=pri, substitutions=[sub], pbc=(1, 1, 1))


@pytest.fixture
def cpool(plat):
    return ClustersPool(plat, npoints=[1, 2], radii=[0, 1.1])


@pytest.fixture
def corrc(plat, cpool):
    return CorrelationsCalculator("trigonometric", plat, cpool)


@pytest.fixture
def scell(plat):
    p = [3, 3, 3]
    return SuperCell(plat, p)


@pytest.fixture
def structure(plat, scell):
    return scell.gen_random_structure(nsubs=int(np.prod(scell.get_index()) / 2))


@pytest.fixture
def structures_set_empty(plat):
    return StructuresSet(parent_lattice=plat)


@pytest.fixture
def structures_set(structures_set_empty, scell):
    np.random.seed(10)
    n_structures = 10
    for i in range(n_structures):
        structures_set_empty.add_structure(scell.gen_random_structure())
    strset.set_calculator(EMT2())
    energies = strset.compute_property_values()
    return structures_set_empty


@pytest.fixture
def model(plat, cpool, corrc):
    """Init with ECIS, default."""
    mult = cpool.get_multiplicities()
    model = Model(corrc, "energy", ecis=mult)
    return model


@pytest.fixture
def model_estimator(plat, cpool, corrc):
    """Init with estimator."""
    mult = cpool.get_multiplicities()
    estimator = LinearRegression()
    estimator.coef_ = mult
    estimator.intercept_ = 0.0
    model = Model(corrc, "energy", estimator=estimator)
    return model


def test_raise_init(plat, cpool, corrc):
    with pytest.raises(ValueError):
        Model(corrc, "energy", ecis=None, estimator=None)


def test_serialize_load(model, structure):
    filepath = "model_test.pickle"
    model.serialize(filepath=filepath)
    model_loaded = Model(filepath=filepath)
    prediction = model.predict(structure)
    prediction_loaded = model_loaded.predict(structure)
    np.testing.assert_allclose(prediction_loaded, prediction)


def test_ecis_estimator_init(model, model_estimator, structure):
    prediction = model.predict(structure)
    prediction_estimator = model_estimator.predict(structure)
    np.testing.assert_allclose(prediction_estimator, prediction)
