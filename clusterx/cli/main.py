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


