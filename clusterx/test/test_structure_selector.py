from clusterx.structure_selector import StructureSelector

def test_structure_selector():

    ss = StructureSelector()

    _str = ss.say_hello()

    assert(isinstance(_str,str))
