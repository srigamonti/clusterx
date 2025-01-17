# Copyright (c) 2015-2019, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

import sys
import os
import plac
from clusterx.cli.config_utils import read_toml_config
from clusterx.cli import commands as cmds
from clusterx.cli.commands import import_custom_modules


def main():
    """
    CELL command line interface

    Parameters can be passed with an input file in TOML format, e.g.

    cell --input_file [cellinput].toml

    """

    # Check for cellinput.toml in the current working directory
    if len(sys.argv) == 3 and sys.argv[1] == "--input_file":
        toml_file_path = os.path.join(os.getcwd(), sys.argv[2])
    else:
        toml_file_path = os.path.join(os.getcwd(), "cellinput.toml")

    custom_dir = os.getcwd()  # Default to CWD

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

        commands_ = [
            key
            for key, value in config_dict.items()
            if isinstance(value, dict)
            or (
                isinstance(value, list)
                and all(isinstance(item, dict) for item in value)
            )
        ]
        commands = config_dict.get("do", commands_)
        print("commands in cellinput.toml file are", commands)
        available_commands = cmds.commands  # List of available command names
        print("available commands are", available_commands)

        for command in commands:
            if command in available_commands:
                func = getattr(cmds, command)
                arg_dict_from_toml = config_dict[str(command)]
                print("Argument dict from TOML file: ", arg_dict_from_toml)

                # Check if arg_dict_from_toml is a list of dictionaries
                if isinstance(arg_dict_from_toml, list) and all(
                    isinstance(item, dict) for item in arg_dict_from_toml
                ):
                    # Execute func for each dictionary in the list
                    for params in arg_dict_from_toml:
                        print(f"Executing {command} with params: {params}")
                        func(**params)
                elif isinstance(arg_dict_from_toml, dict):
                    # Execute func with the single dictionary of parameters
                    print(f"Executing {command} with params: {arg_dict_from_toml}")
                    func(**arg_dict_from_toml)
                else:
                    print(
                        f"Invalid format for arguments in command '{command}': {arg_dict_from_toml}"
                    )
                    sys.exit(1)
            else:
                print(f"Unknown command: {command}")
                sys.exit(1)


if __name__ == "__main__":
    main()
