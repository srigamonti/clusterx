# Copyright (c) 2015-2019, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

import plac

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
    for out in plac.call(cmds): print(out)


