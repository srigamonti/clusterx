# Copyright (c) 2015-2019, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

import plac
import pytest
import os
import clusterx.test.__init__ as tm
from clusterx.test.__init__ import tests as tlist

commands = ['test']

@plac.annotations(
    testslist=('Print available tests', 'flag','l'),
    name=("Make one test","option","n")
)
def test(testslist=False, name=""):
    "Test CELL"
    if testslist:
        print(tlist)
        return()
    else:
        if name is not "":
            path = os.path.join(os.path.dirname(tm.__file__),name+".py")
            print(path)
        else:
            path = os.path.dirname(tm.__file__)

        exit_code = pytest.main([path,"-v","--cache-clear","--capture=no","--disable-warnings","--junit-xml=testlog.xml","--html=testlog.html"])
        if not exit_code == pytest.ExitCode.OK:
            raise AssertionError(f"Tests have failed! pytest exit code: {exit_code}")
        return ()
