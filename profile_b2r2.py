#!/usr/bin/env python
# coding: utf-8

import numpy as np
from src import b2r2
import argparse as ap
import qml 
from glob import glob

if __name__ == "__main__":

    parser = ap.ArgumentParser()
    parser.add_argument("version") # l, a, n
    args = parser.parse_args()

    reactants = sorted(glob("data/GDB7-20-TS/xyz/reactant_*.xyz"))
    products = sorted(glob("data/GDB7-20-TS/xyz/product_*.xyz"))
    reactants = reactants[:1000]
    products = products[:1000]
    mols_reactants = [qml.Compound(x) for x in reactants]
    ncharges_reactants = [[x.nuclear_charges] for x in mols_reactants]
    coords_reactants = [[x.coordinates] for x in mols_reactants]
    mols_products = [qml.Compound(x) for x in products]
    ncharges_products = [[x.nuclear_charges] for x in mols_products]
    coords_products = [[x.coordinates] for x in mols_products]
    unique_ncharges = np.unique(np.concatenate([x[0] for x in
                                            ncharges_reactants]))

    if args.version == 'l':
        b2r2_l = b2r2.get_b2r2_l(ncharges_reactants, ncharges_products, 
                                coords_reactants, coords_products,
                                elements=unique_ncharges, Rcut=3)
    elif args.version == 'a':
        b2r2_l = b2r2.get_b2r2_a(ncharges_reactants, ncharges_products, 
                                coords_reactants, coords_products,
                                elements=unique_ncharges, Rcut=3)
    elif args.version == 'n':
        b2r2_l = b2r2.get_b2r2_n(ncharges_reactants, ncharges_products, 
                                coords_reactants, coords_products,
                                elements=unique_ncharges, Rcut=3)
    elif args.version == 'cheap':
        b2r2_l = b2r2.get_b2r2_l(ncharges_reactants, ncharges_products, 
                                coords_reactants, coords_products,
                                elements=unique_ncharges, Rcut=3, variation='cheap')


