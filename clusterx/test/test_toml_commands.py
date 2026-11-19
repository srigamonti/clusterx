"""Ordered named steps and compatibility with existing TOML workflows."""

from copy import deepcopy
import subprocess
import sys

import pytest
import toml

from clusterx.cli import main as cli_main
from clusterx.cli.config_utils import get_toml_commands


NAMED_STEPS = """
[export_before]
command = "convert"
value = "before"

[plot]
value = "middle"

[export_after]
command = "convert"
value = "after"
"""


def test_named_steps_interleave_commands():
    calls = get_toml_commands(toml.loads(NAMED_STEPS), ["convert", "plot"])
    assert [(label, command, params["value"]) for label, command, params, _ in calls] == [
        ("export_before", "convert", "before"),
        ("plot", "plot", "middle"),
        ("export_after", "convert", "after"),
    ]


def test_single_bracket_names_are_unique_standard_toml():
    with pytest.raises(toml.TomlDecodeError):
        toml.loads('[convert]\nvalue="before"\n[plot]\nvalue="middle"\n[convert]\nvalue="after"\n')


def test_legacy_arrays_keep_grouped_execution():
    source = '[[convert]]\nvalue="before"\n[plot]\nvalue="middle"\n[[convert]]\nvalue="after"\n'
    calls = get_toml_commands(toml.loads(source), ["convert", "plot"])
    assert [(command, params["value"]) for _, command, params, _ in calls] == [
        ("convert", "before"), ("convert", "after"), ("plot", "middle")
    ]


def test_do_orders_and_repeats_named_steps_without_mutating_config():
    config = toml.loads(NAMED_STEPS)
    config["do"] = ["export_after", "plot", "export_before", "export_after"]
    config["plot"]["ignore"] = True
    config["export_after"]["options"] = {"values": [1, 2]}
    original = deepcopy(config)
    calls = get_toml_commands(config, ["convert", "plot"])
    assert [label for label, _, _, _ in calls] == config["do"]
    assert calls[1][3] is True
    calls[0][2]["options"]["values"].append(3)
    assert calls[3][2]["options"]["values"] == [1, 2]
    assert config == original


def test_nested_options_and_arrays_are_command_arguments():
    config = toml.loads("""
[make_plot]
command = "plot"
[make_plot.options]
linewidth = 2
[[make_plot.series]]
ycolumn = "full"
[[make_plot.series]]
ycolumn = "pruned"
""")
    calls = get_toml_commands(config, ["plot"])
    assert calls == [("make_plot", "plot", {"options": {"linewidth": 2},
                                            "series": [{"ycolumn": "full"}, {"ycolumn": "pruned"}]}, False)]


def test_registered_command_keeps_native_command_argument():
    assert get_toml_commands({"run": {"command": "native option"}}, ["run"]) == [
        ("run", "run", {"command": "native option"}, False)
    ]


def test_root_metadata_is_not_dispatched():
    config = {"custom_dir": "/tmp", "custom_modules": [], "run": {}}
    assert get_toml_commands(config, ["run"]) == [("run", "run", {}, False)]
    config["do"] = []
    assert get_toml_commands(config, ["run"]) == []


@pytest.mark.parametrize("config,message", [
    ({"unknown": {}}, "named step needs"),
    ({"step": {"command": 3}}, "string 'command'"),
    ({"step": {"command": "unknown"}}, "Unknown command 'unknown'"),
    ({"step": {"command": "run", "ignore": "true"}}, "must be true or false"),
    ({"do": "run", "run": {}}, "must be a list"),
    ({"do": [3], "run": {}}, "must be a list"),
    ({"do": ["missing"]}, "No command table"),
    ({"do": ["run"], "run": 4}, "must be a table"),
])
def test_invalid_workflow(config, message):
    with pytest.raises(ValueError, match=message):
        get_toml_commands(config, ["run"])


@pytest.fixture
def dispatch(monkeypatch, tmp_path):
    executed = []

    def convert(value):
        executed.append(("convert", value))

    def plot(value):
        executed.append(("plot", value))

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli_main.cmds, "commands", ["convert", "plot"])
    monkeypatch.setattr(cli_main.cmds, "convert", convert, raising=False)
    monkeypatch.setattr(cli_main.cmds, "plot", plot, raising=False)

    def run(source, default_file=False):
        path = tmp_path / ("cellinput.toml" if default_file else "workflow.toml")
        path.write_text(source)
        argv = ["cell"] if default_file else ["cell", "--input_file", str(path)]
        monkeypatch.setattr(sys, "argv", argv)
        cli_main.main()
        return executed

    return run, executed


@pytest.mark.parametrize("default_file", [False, True])
def test_main_executes_in_named_step_order(dispatch, default_file):
    run, _ = dispatch
    assert run(NAMED_STEPS, default_file=default_file) == [
        ("convert", "before"), ("plot", "middle"), ("convert", "after")
    ]


def test_main_do_and_ignore(dispatch):
    run, _ = dispatch
    config = toml.loads(NAMED_STEPS)
    config["do"] = ["export_after", "plot", "export_before"]
    config["plot"]["ignore"] = True
    del config["plot"]["value"]  # An ignored step need not supply required arguments.
    assert run(toml.dumps(config)) == [("convert", "after"), ("convert", "before")]


@pytest.mark.parametrize("bad_step", [
    {"command": "missing", "value": "bad"},
    {"command": "convert", "unknown_option": "bad"},
    {"command": "convert"},
])
def test_main_checks_whole_workflow_before_execution(dispatch, bad_step):
    run, executed = dispatch
    config = {"first": {"command": "convert", "value": "before"}, "invalid": bad_step}
    with pytest.raises(SystemExit) as exc:
        run(toml.dumps(config))
    assert exc.value.code == 1
    assert executed == []


def test_subprocess_named_steps_with_custom_commands(tmp_path):
    module = tmp_path / "workflow_commands.py"
    module.write_text('''from pathlib import Path
commands = ["record", "mark"]
def record(value):
    with Path("execution.txt").open("a") as handle:
        handle.write("record:" + value + "\\n")
def mark(value):
    with Path("execution.txt").open("a") as handle:
        handle.write("mark:" + value + "\\n")
''')
    source = NAMED_STEPS.replace('"convert"', '"record"').replace("[plot]", "[mark]")
    source = f'custom_dir = "{tmp_path.as_posix()}"\ncustom_modules = ["workflow_commands.py"]\n' + source
    path = tmp_path / "workflow.toml"
    path.write_text(source)
    result = subprocess.run(["cell", "--input_file", str(path)], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "execution.txt").read_text().splitlines() == [
        "record:before", "mark:middle", "record:after"
    ]
