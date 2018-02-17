from os.path import dirname, basename, isfile
import glob
fnames = glob.glob(dirname(__file__)+"/*.py")
tests = [ basename(f)[:-3] for f in fnames if isfile(f) and not f.endswith('__init__.py') and not f.endswith('main.py')]

