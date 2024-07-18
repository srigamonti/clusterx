# Copyright (c) 2015-2019, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

import sys
import os
import toml
from os.path import dirname, basename, isfile
import glob

##########################################
# This module performs dynamical creation of functions from the modules contained
# in clusterx/cli/, i.e. it performs dynamical imports like:
#
# from clusterx.cli.plot_clusters import plot_clusters1
# 
# as needed by plac
##########################################

# Get list of modules

mods = glob.glob(dirname(__file__)+"/*.py")

modules = [ basename(f)[:-3] for f in mods if isfile(f) and not f.endswith('__init__.py') and not f.endswith('main.py') and not f.endswith('commands.py') and not f.endswith('config_utils.py')]

# Check for cellinput.toml in the current working directory
toml_file_path = os.path.join(os.getcwd(), 'cellinput.toml')
custom_dir = os.getcwd()  # Default to CWD
custom_modules = []

if os.path.isfile(toml_file_path):
    config_dict = toml.load(toml_file_path)
    custom_modules = config_dict.get('custom_modules', [])
    custom_dir = config_dict.get('custom_dir', os.getcwd())
    sys.path.append(custom_dir)

# Dynamically retrieve functions from modules in clusterx.cli, 
# and re-create them in the current module
commands = [] # This array is needed by plac
for module in modules:
    for command in getattr(__import__('clusterx.cli.'+module,fromlist=[module]),'commands'):
        commands.append(command)
        setattr(sys.modules[__name__], command, getattr(__import__('clusterx.cli.'+module,fromlist=[module]),command))


# Add custom modules
for module in custom_modules:
    module_name = module[:-3]  # Strip .py extension
    custom_commands = getattr(__import__(module_name, fromlist=[module_name]), 'commands', [])
    for command in custom_commands:
        commands.append(command)
        setattr(sys.modules[__name__], command, getattr(__import__(module_name, fromlist=[module_name]), command))

##########################################

def __init__():
    """
    CELL command line interface
    """
    print(__init__.__doc__)

def __missing__(name):
    return ('Command %r does not exist' % name,)

def __exit__(etype, exc, tb):
    "Will be called automatically at the end of the intepreter loop"
    if etype in (None, GeneratorExit): # success
        print('ok')
