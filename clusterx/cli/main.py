# Copyright (c) 2015-2019, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

import plac
import inspect
import sys
from clusterx.cli.config_utils import read_toml_config, generate_argv

def main():
    """
    CELL command line interface

    This subpackage collects modules which define the commands 
    available in cell. Uses plac, based on argparse, for 
    command-line-argument parsing. Plac infers command options
    from function declaration, making easy for developers to add
    new commands, by just defining a function.
    """
    from clusterx.cli import commands as cmds

    no_toml_file = not any(arg in sys.argv for arg in {"--toml_file", "-tf"})
    print_help = len(sys.argv) == 2 and ( sys.argv[1] == "-h" or sys.argv[1] == "--help" )

    if print_help:
        print(main.__doc__)

    config_dict = {}
    commands = []
    if no_toml_file:
        print(sys.argv[1:])
        plac.call(cmds, sys.argv[1:])
    else:
        try:
            toml_file_index = sys.argv.index('--toml_file') + 1
        except ValueError:
            pass

        try:
            toml_file_index = sys.argv.index('-tf') + 1
        except ValueError:
            raise ValueError("Neither --toml_file nor -tf in cli")

        toml_file = sys.argv[toml_file_index]
        config_dict = read_toml_config(toml_file)
        commands = list(config_dict.keys())
            
        available_commands = cmds.commands  # List of available command names in my_module

        for command in commands:
            if command in available_commands:
                func = getattr(cmds, command)
                argv_from_toml = generate_argv(func, config_dict[str(command)])
                plac.call(func, argv_from_toml)
            else:
                print(f"Unknown command: {command}")
                sys.exit(1)


    
if __name__ == '__main__':
    main()


