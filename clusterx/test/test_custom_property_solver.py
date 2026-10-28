import pytest

from clusterx.cli.generate_derivative_structures import _load_property_solver


def test_loads_module_function_solver(tmp_path):
    module_path = tmp_path / "custom_solvers.py"
    module_path.write_text(
        "def compute_energy(structure, shape, concentration, row, scale=1):\n    return scale * concentration\n"
    )

    solver, kwargs = _load_property_solver(
        {
            "filename": str(module_path),
            "function": "compute_energy",
            "kwargs": {"scale": 2},
        }
    )

    assert solver(None, None, 0.25, row=None, **kwargs) == 0.5


def test_loads_selected_method_from_solver_class(tmp_path):
    module_path = tmp_path / "custom_solvers.py"
    module_path.write_text(
        "class MySolvers:\n"
        "    def compute_property_1(self, structure, shape, concentration, row):\n"
        "        return 1\n"
        "    def compute_property_2(self, structure, shape, concentration, row):\n"
        "        return 2\n"
    )

    solver, _ = _load_property_solver(
        {
            "filename": str(module_path),
            "classname": "MySolvers",
            "method": "compute_property_2",
        }
    )

    assert solver(None, None, 0.25, row=None) == 2


def test_class_solver_defaults_to_compute_property(tmp_path):
    module_path = tmp_path / "custom_solvers.py"
    module_path.write_text(
        "class MySolver:\n"
        "    def compute_property(self, structure, shape, concentration, row):\n"
        "        return concentration\n"
    )

    solver, _ = _load_property_solver({"filename": str(module_path), "classname": "MySolver"})

    assert solver(None, None, 0.25, row=None) == 0.25


@pytest.mark.parametrize(
    "solver_config",
    [
        {"filename": "solver.py"},
        {"filename": "solver.py", "function": "run", "classname": "Solver"},
        {"filename": "solver.py", "function": "run", "kwargs": []},
    ],
)
def test_rejects_invalid_solver_configuration(solver_config):
    with pytest.raises((TypeError, ValueError)):
        _load_property_solver(solver_config)
