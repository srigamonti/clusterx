# Copyright (c) 2015-2019, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

import sys
import os
import plac
from clusterx.cli.config_utils import read_toml_config
from clusterx.cli import commands as cmds


def main():
    """
    CELL command line interface

    This subpackage collects modules which define the commands
    available in cell. Uses plac, based on argparse, for
    command-line-argument parsing. Plac infers command options
    from function declaration, making easy for developers to add
    new commands, by just defining a function.
    """

    # Check for cellinput.toml in the current working directory
    toml_file_path = os.path.join(os.getcwd(), "cellinput.toml")
    custom_dir = os.getcwd()  # Default to CWD

    no_toml_file = True
    if os.path.isfile(toml_file_path):
        config_dict = read_toml_config(toml_file_path)
        custom_dir = config_dict.get("custom_dir", os.getcwd())

        # Add the custom directory to sys.path to import custom modules
        sys.path.append(custom_dir)
        no_toml_file = False

    print_help = len(sys.argv) == 2 and (sys.argv[1] == "-h" or sys.argv[1] == "--help")

    if print_help:
        print(main.__doc__)

    config_dict = {}
    commands = []
    if no_toml_file or len(sys.argv) > 1:
        plac.call(cmds, sys.argv[1:])
    else:
        config_dict = read_toml_config(toml_file_path)
        commands = [
            key for key, value in config_dict.items() if isinstance(value, dict)
        ]
        print("commands in cellinput.toml file are", commands)
        available_commands = cmds.commands  # List of available command names
        print("available commands are", available_commands)

        for command in commands:
            if command in available_commands:
                func = getattr(cmds, command)
                arg_dict_from_toml = config_dict[str(command)]
                print("Argument dict from TOML file: ", arg_dict_from_toml)
                func(**arg_dict_from_toml)
            else:
                print(f"Unknown command: {command}")
                sys.exit(1)


if __name__ == "__main__":
    main()
