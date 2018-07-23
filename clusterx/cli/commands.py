import sys
import plac

##########################################
# This performs dynamical function creations from the modules contained in this clusterx/cli/,
# i.e. it performs dynamically imports like:
# from clusterx.cli.plot_clusters import plot_clusters1, as needed by plac
#First we get the list of modules
from os.path import dirname, basename, isfile
import glob
mods = glob.glob(dirname(__file__)+"/*.py")
#print('mods',mods)

modules = [ basename(f)[:-3] for f in mods if isfile(f) and not f.endswith('__init__.py') and not f.endswith('main.py') and not f.endswith('commands.py')]
#print('modules',modules)


# Finally the functions are included here
commands = [] # This array is needed by plac
for module in modules:
    #print('#########################')
    #print("module in for: ", #print('#########################')                                                   odule)
    #print("commands: ",getattr(__import__('clusterx.cli.'+module,fromlist=[module]),'commands'))
    for command in getattr(__import__('clusterx.cli.'+module,fromlist=[module]),'commands'):
        commands.append(command)
        setattr(sys.modules[__name__], command, getattr(__import__('clusterx.cli.'+module,fromlist=[module]),command))
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

if __name__ == '__main__':
    main = __import__(__name__) # the module imports itself!
    for out in plac.call(main): print(out)
