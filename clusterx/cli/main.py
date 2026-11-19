# Copyright (c) 2015-2026, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

import os
import sys
from inspect import signature
from pprint import pformat

import plac

from clusterx.cli import commands as cmds
from clusterx.cli.commands import import_custom_modules
from clusterx.cli.config_utils import get_toml_commands, read_toml_config


def main():
    """
    CELL Command Line Interface

    Usage:
        Parameters can be passed as command-line flags, for example::

            cell command -opt1=val1 -opt2=val2 --flag1

        Alternatively, you can use an input file in TOML format::

            cell --input_file cellinput.toml

        If no arguments are provided, CELL will automatically look for a file named
        `cellinput.toml` in the current working directory and use it if found.

    TOML Input File:
        Using a TOML file has the advantage of supporting multiple commands in a single run,
        which is more powerful and flexible than the command-line interface alone.

        Example of a TOML file with multiple commands::

            [command1]
            opt1 = "val1"
            opt2 = "val2"
            flag1 = true

            [command2]
            opt1 = "val1"
            opt2 = "val2"
            flag1 = true

        To repeat commands with other operations in between, use unique step
        names and specify the command in each table. This is standard TOML and
        needs only single brackets. Steps run in the order they appear::

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

        A top-level ``do = ["export_before", "plot", "export_after"]`` can
        explicitly select and order steps. Put it before any table header.

        Legacy command tables and arrays of tables remain supported. Array
        entries execute as a group, so use named steps for interleaving::

            [[command1]]
            opt1 = "val1A"
            opt2 = "val2A"

            [[command1]]
            opt1 = "val1B"
            opt2 = "val2B"

        You can avoid execution of a command by adding the key `ignore` with the
        value `true`, e.g., below commandA and the second call to commandB will
        be ignored::

            [commandA]
            ignore = true
            opt1 = val1
            opt2 = val2

            [[commandB]]
            opt1 = val1
            opt2 = val2

            [[commandB]]
            ignore = true
            opt1 = val1
            opt2 = val2


    Custom Commands:
        You can extend CELL by placing a file named `custom_cell_commands.toml` in your
        working directory. This file should contain the following attributes::

            custom_dir = "path/to/my/custom/commands/folder/"      # Directory containing custom commands
            custom_modules = ["custom_command1.py", "custom_command2.py"]  # Python modules to load from that directory

        Each listed module must be located in the specified `custom_dir`.

        Instead of using a separate `custom_cell_commands.toml` file,
        you can also include the `custom_dir` and `custom_modules` attributes
        directly in your main input TOML file (e.g., `cellinput.toml`) under the top-level section.
        This way, you can manage both your custom commands and your main command configurations in a single file.

    Notes to develpoers:
        - We use plac, which is a simple and intuitive library for creating command-line interfaces in Python.
        - To avoid problems between plac and python type hints, we specify the type of each argument in
            the plac.annotations decorator. For using annotations (see https://plac.readthedocs.io/en/latest/#more-features)
            consider that annotatio is a 6-tuple of the form (help, kind, abbrev, type, choices, metavar).

    """

    # Check for cellinput.toml in the current working directory
    if len(sys.argv) == 3 and sys.argv[1] == "--input_file":
        toml_file_path = os.path.join(os.getcwd(), sys.argv[2])
    else:
        toml_file_path = os.path.join(os.getcwd(), "cellinput.toml")

    custom_dir = os.getcwd()  # Default to CWD

    if os.path.isfile("custom_cell_commands.toml"):
        config_dict = read_toml_config("custom_cell_commands.toml")
        custom_dir = config_dict.get("custom_dir", os.getcwd())
        custom_module_filenames = config_dict.get("custom_modules", [])
        import_custom_modules(custom_module_filenames, custom_dir)
        # Add the custom directory to sys.path to import custom modules
        sys.path.append(custom_dir)

    no_toml_file = True
    if os.path.isfile(toml_file_path):
        config_dict = read_toml_config(toml_file_path)
        custom_dir = config_dict.get("custom_dir", os.getcwd())
        custom_module_filenames = config_dict.get("custom_modules", [])
        import_custom_modules(custom_module_filenames, custom_dir)
        # Add the custom directory to sys.path to import custom modules
        sys.path.append(custom_dir)
        no_toml_file = False

    print_help = len(sys.argv) == 2 and (sys.argv[1] == "-h" or sys.argv[1] == "--help")

    if print_help:
        print(main.__doc__)

    config_dict = {}
    if no_toml_file or (len(sys.argv) > 1 and sys.argv[1] != "--input_file"):
        plac.call(cmds, sys.argv[1:])
    else:
        config_dict = read_toml_config(toml_file_path)

        try:
            calls = get_toml_commands(config_dict, cmds.commands)
            # Check arguments before running commands that may write files.
            for label, command, params, ignored in calls:
                if ignored:
                    continue
                try:
                    sig = signature(getattr(cmds, command))
                except (ValueError, TypeError):
                    continue  # Some extension callables do not expose a signature.
                try:
                    sig.bind(**params)
                except TypeError as exc:
                    raise ValueError(f"Invalid arguments for step '{label}' ({command}): {exc}") from exc
        except ValueError as exc:
            print(f"Invalid TOML command configuration: {exc}", file=sys.stderr)
            sys.exit(1)

        for label, command, params, ignored in calls:
            description = command if label == command else f"{command} (step {label})"
            if ignored:
                print(f"Skipping {description} because 'ignore' is True")
            else:
                print(f"\nExecuting {description} with params:\n {pformat(params)}\n")
                getattr(cmds, command)(**params)


if __name__ == "__main__":
    main()
