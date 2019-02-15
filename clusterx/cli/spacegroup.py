# Copyright (c) 2015-2019, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

import plac
from ase.spacegroup import Spacegroup

commands = ["print_sg_ops"]

@plac.annotations(
    spacegroup_number=('Spacegroup number', 'option','sgn',int),
    cluster_number=('Plot one cluster','option','n',int,[2,3,4],None))
def print_sg_ops(spacegroup_number, cluster_number="3"):
    "Print to screen the symmetry operations of a given spacegroup"
    sg = Spacegroup(spacegroup_number)
    print(sg)
        
    sites, kinds = sg.equivalent_sites([(0,0.235,0.3),(0.356,1,2.1)])
    print(sites)
    print(kinds)
    return ()

@plac.annotations(
    plot_all=('Plot all clusters2', 'flag','a'),
    cluster_number=('Plot one cluster','option','n',int,[2,3,4],None))
def plot_clusters2(plot_all, cluster_number="3"):
    "Plot clusters2"
    if plot_all:
        plt.plot([1,2,3,4])
        plt.ylabel('some numbers')
        plt.show()
    else:
        a=[]
        for i in range(cluster_number):
            a.append(i)
        plt.plot(a)
        plt.ylabel('some other numbers')
        plt.show()
        
    return ()



