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
def test(testslist=False, name=None):
    "Test CELL"
    if testslist:
        print(tlist)
        return()
    else:
        if name is not None:
            path = os.path.join(os.path.dirname(tm.__file__),name,".py")
        else:
            path = os.path.dirname(tm.__file__)
            
        pytest.main([os.path.dirname(tm.__file__),"-v","--cache-clear","--capture=no","--junit-xml=testlog.xml"])
        return ()


