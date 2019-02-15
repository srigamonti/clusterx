# Copyright (c) 2015-2019, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

from os.path import dirname, basename, isfile
import glob
fnames = glob.glob(dirname(__file__)+"/*.py")
tests = [ basename(f)[:-3] for f in fnames if isfile(f) and not f.endswith('__init__.py') and not f.endswith('main.py')]

