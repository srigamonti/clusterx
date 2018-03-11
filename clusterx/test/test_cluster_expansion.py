"""Draft of what a cluster expansion would be with CELL

    cual = bulk('CuAl','zincblende',a=6.1)
    agal = bulk('AgAl','zincblende',a=6.1)
    sral = bulk('SrAl','zincblende',a=6.1)
    cuc = bulk('CuC','zincblende',a=6.1)

    plat = ParentLattice(atoms=cual,substitutions=[agal,sral,cuc])

    scell = SuperCell(plat, np.diag([2,2,2]))
    
    ss = StructuresSet(plat, name="t1", calculator = EMT2())
    
    for i in range(10):
        ss.add_structure(scell.gen_random(nsubs={0:[1],1:[3,2]}))

    scell = SuperCell(plat, np.diag([3,3,3]))
    for i in range(10):
        ss.add_structure(scell.gen_random(nsubs={0:[4],1:[5,7]}))

    ss.calculate_property("energy")

    cs = ClustersSet(plat, name="small", npoints=[2,3,4], radii=[6,6,5])

    write_correlations(structures_set=ss, clusters_set=cs)




"""
