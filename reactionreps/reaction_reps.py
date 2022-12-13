import itertools
from ast import literal_eval

import numpy as np
import pandas as pd
import qml
from dscribe.descriptors import SOAP
from scipy import stats

from reactionreps.b2r2 import get_b2r2_a_molecular, get_b2r2_l_molecular, get_b2r2_n_molecular
from reactionreps.utils import xyz_to_atomsobj
from reactionreps.slatm import get_slatm

pt = {"H": 1, "C": 6, "N": 7, "O": 8, "S": 16, "Cl": 17, "F": 9}


class B2R2:
    """Reps available in B2R2 series"""

    def __init__(self):
        self.unique_ncharges = []
        self.barriers = []
        self.energies = []
        self.mols_reactants = [[]]
        self.mols_products = [[]]
        return

    def get_sn2_data(self):
        reactions = pd.read_csv("data/SN2-20/reactions.csv", index_col=0)
        reactions["reactant"] = reactions["reactant"].apply(literal_eval)
        reactions["product"] = reactions["product"].apply(literal_eval)
        self.energies = reactions["rxn_nrj"].to_numpy()
        all_r_files = [
            x for reactions in reactions["reactant"].to_list() for x in reactions
        ]
        all_p_files = [
            x for reactions in reactions["product"].to_list() for x in reactions
        ]
        all_files = list(set(all_r_files)) + list(set(all_p_files))
        all_mols = [qml.Compound(x) for x in all_files]
        self.ncharges = [mol.nuclear_charges for mol in all_mols]
        self.unique_ncharges = np.unique(np.concatenate(self.ncharges))

        self.mols_reactants = [
            [qml.Compound(x) for x in reactants]
            for reactants in reactions["reactant"].to_list()
        ]
        self.mols_products = [
            [qml.Compound(x) for x in products]
            for products in reactions["product"].to_list()
        ]

        return

    def get_gdb7_rxn_data(self):
        reactions = pd.read_csv("data/GDB7-20-TS/dataset.csv")
        self.barriers = reactions["ea kcal/mol"].to_numpy()
        all_r_files = [
            "data/GDB7-20-TS/xyz/" + x for x in reactions["reactant"].to_list()
        ]
        all_p_files = [
            "data/GDB7-20-TS/xyz/" + x for x in reactions["product"].to_list()
        ]
        ncharges = [x[0].nuclear_charges for x in self.mols_reactants]
        self.unique_ncharges = np.unique(np.concatenate(self.ncharges))

        self.mols_reactants = [[qml.Compound(x)] for x in all_r_files]
        self.mols_products = [[qml.Compound(x)] for x in all_p_files]
        return

    def get_proparg_data(self):
        data = pd.read_csv("data/Proparg-21-TS/data.csv", index_col=0)
        reactants_files = [
            "data/Proparg-21-TS/data_react_xyz/"
            + data.mol.values[i]
            + data.enan.values[i]
            + ".xyz"
            for i in range(len(data))
        ]
        products_files = [
            "data/Proparg-21-TS/data_prod_xyz/"
            + data.mol.values[i]
            + data.enan.values[i]
            + ".xyz"
            for i in range(len(data))
        ]
        all_mols = [qml.Compound(x) for x in reactants_files + products_files]
        self.barriers = data.dErxn.to_numpy()
        self.ncharges = [mol.nuclear_charges for mol in all_mols]
        self.unique_ncharges = np.unique(np.concatenate(self.ncharges))

        self.mols_reactants = [[qml.Compound(x)] for x in reactants_files]
        self.mols_products = [[qml.Compound(x)] for x in products_files]

        return

    def get_hydroform_data(self):
        co_df = pd.read_csv("data/Hydroform-22-TS/Co_clean.csv")
        names = co_df["name"].to_list()
        labels = [name[3:] for name in names]
        co_reactants = [
            "data/Hydroform-22-TS/geometries/co/r/" + label + "_reactant.xyz"
            for label in labels
        ]
        co_products = [
            "data/Hydroform-22-TS/geometries/co/p/" + label + "_product.xyz"
            for label in labels
        ]
        self.co_barriers = co_df["f_barr"].to_numpy()
        self.mols_reactants_co = [[qml.Compound(x)] for x in co_reactants]
        self.mols_products_co = [[qml.Compound(x)] for x in co_products]

        ir_df = pd.read_csv("data/Hydroform-22-TS/Ir_clean.csv")
        names = ir_df["name"].to_list()
        labels = [name[3:] for name in names]
        ir_reactants = [
            "data/Hydroform-22-TS/geometries/ir/r/" + label + "_reactant.xyz"
            for label in labels
        ]
        ir_products = [
            "data/Hydroform-22_TS/geometries/ir/p/" + label + "_product.xyz"
            for label in labels
        ]
        self.ir_barriers = ir_df["f_barr"].to_numpy()
        self.mols_reactants_ir = [[qml.Compound(x)] for x in ir_reactants]
        self.mols_products_ir = [[qml.Compound(x)] for x in ir_products]

        rh_df = pd.read_csv("data/Hydroform-22-TS/Rh_clean.csv")
        names = rh_df["name"].to_list()
        labels = [name[3:] for name in names]
        rh_reactants = [
            "data/Hydroform-22-TS/geometries/rh/r/" + label + "_reactant.xyz"
            for label in labels
        ]
        rh_products = [
            "data/Hydroform-22-TS/geometries/rh/p/" + label + "_product.xyz"
            for label in labels
        ]
        self.rh_barriers = rh_df["f_barr"].to_numpy()
        self.mols_reactants_rh = [[qml.Compound(x)] for x in rh_reactants]
        self.mols_products_rh = [[qml.Compound(x)] for x in rh_products]

        self.barriers = np.concatenate(
            (self.co_barriers, self.ir_barriers, self.rh_barriers), axis=0
        )

        all_reactants = co_reactants + ir_reactants + rh_reactants
        all_products = co_products + ir_products + rh_products
        list_reactants = [qml.Compound(x) for x in all_reactants]

        self.mols_reactants = [[qml.Compound(x)] for x in all_reactants]
        self.mols_products = [[qml.Compound(x)] for x in all_products]

        ncharges = [mol.nuclear_charges for mol in list_reactants]
        self.unique_ncharges = np.unique(np.concatenate(self.ncharges))

        return

    def get_b2r2_l(self, Rcut=3.5, gridspace=0.03):
        b2r2_reactants = [
            [
                get_b2r2_l_molecular(
                    x.nuclear_charges,
                    x.coordinates,
                    Rcut=Rcut,
                    gridspace=gridspace,
                    elements=self.unique_ncharges,
                )
                for x in reactants
            ]
            for reactants in self.mols_reactants
        ]
        # first index is reactants
        b2r2_reactants_sum = np.array([sum(x) for x in b2r2_reactants])

        b2r2_products = [
            [
                get_b2r2_l_molecular(
                    x.nuclear_charges,
                    x.coordinates,
                    Rcut=Rcut,
                    gridspace=gridspace,
                    elements=self.unique_ncharges,
                )
                for x in products
            ]
            for products in self.mols_products
        ]
        b2r2_products_sum = np.array([sum(x) for x in b2r2_products])

        b2r2_diff = b2r2_products_sum - b2r2_reactants_sum

        return b2r2_diff

    def get_b2r2_a(self, Rcut=3.5, gridspace=0.03):
        elements = self.unique_ncharges
        b2r2_reactants = [
            [
                get_b2r2_a_molecular(
                    x.nuclear_charges,
                    x.coordinates,
                    Rcut=Rcut,
                    gridspace=gridspace,
                    elements=elements,
                )
                for x in reactants
            ]
            for reactants in self.mols_reactants
        ]
        # first index is reactants
        b2r2_reactants_sum = np.array([sum(x) for x in b2r2_reactants])

        b2r2_products = [
            [
                get_b2r2_a_molecular(
                    x.nuclear_charges,
                    x.coordinates,
                    Rcut=Rcut,
                    gridspace=gridspace,
                    elements=elements,
                )
                for x in products
            ]
            for products in self.mols_products
        ]
        b2r2_products_sum = np.array([sum(x) for x in b2r2_products])

        b2r2_diff = b2r2_products_sum - b2r2_reactants_sum

        return b2r2_diff

    def get_b2r2_n(self, Rcut=3.5):
        b2r2_reactants = [
            [
                get_b2r2_n_molecular(
                    x.nuclear_charges,
                    x.coordinates,
                    Rcut=Rcut,
                    elements=self.unique_ncharges,
                )
                for x in reactants
            ]
            for reactants in self.mols_reactants
        ]
        # first index is reactants
        b2r2_reactants_sum = np.array([sum(x) for x in b2r2_reactants])

        b2r2_products = [
            [
                get_b2r2_n_molecular(
                    x.nuclear_charges,
                    x.coordinates,
                    Rcut=Rcut,
                    elements=self.unique_ncharges,
                )
                for x in products
            ]
            for products in self.mols_products
        ]
        b2r2_products_sum = np.array([sum(x) for x in b2r2_products])

        return np.concatenate((b2r2_reactants_sum, b2r2_products_sum), axis=1)


class QML:
    """Reps available in the qml package"""

    def __init__(self):
        self.unique_ncharges = []
        self.max_natoms = 0
        self.atomtype_dict = {"H": 0, "C": 0, "N": 0, "O": 0, "S": 0, "Cl": 0, "F": 0}
        self.mols_reactants = [[]]
        self.mols_products = [[]]
        self.energies = []
        self.barriers = []
        return

    def get_sn2_data(self):
        reactions = pd.read_csv("data/SN2-20/reactions.csv", index_col=0)
        reactions["reactant"] = reactions["reactant"].apply(literal_eval)
        reactions["product"] = reactions["product"].apply(literal_eval)
        self.energies = reactions["rxn_nrj"].to_numpy()
        all_r_files = [
            x for reactions in reactions["reactant"].to_list() for x in reactions
        ]
        all_p_files = [
            x for reactions in reactions["product"].to_list() for x in reactions
        ]
        all_files = list(set(all_r_files)) + list(set(all_p_files))
        all_mols = [qml.Compound(x) for x in all_files]
        self.ncharges = [mol.nuclear_charges for mol in all_mols]
        self.unique_ncharges = np.unique(np.concatenate(self.ncharges))
        self.max_natoms = max([len(mol.nuclear_charges) for mol in all_mols])

        # atomtype dict for BoB
        for elem in pt.keys():
            counts = []
            for ncharge_list in self.ncharges:
                count = np.count_nonzero(ncharge_list == pt[elem])
                counts.append(count)

                # keep max count
                if counts:
                    self.atomtype_dict[elem] = max(counts)
                else:
                    self.atomtype_dict[elem] = 1

        self.mols_reactants = [
            [qml.Compound(x) for x in reactants]
            for reactants in reactions["reactant"].to_list()
        ]
        self.mols_products = [
            [qml.Compound(x) for x in products]
            for products in reactions["product"].to_list()
        ]

        return

    def get_gdb7_rxn_data(self):
        reactions = pd.read_csv("data/GDB7-20-TS/dataset.csv")
        self.barriers = reactions["ea kcal/mol"].to_numpy()
        all_r_files = [
            "data/GDB7-20-TS/xyz/" + x for x in reactions["reactant"].to_list()
        ]
        all_p_files = [
            "data/GDB7-20-TS/xyz/" + x for x in reactions["product"].to_list()
        ]
        self.mols_reactants = [[qml.Compound(x)] for x in all_r_files]
        self.mols_products = [[qml.Compound(x)] for x in all_p_files]
        self.ncharges = [x[0].nuclear_charges for x in self.mols_reactants]
        self.unique_ncharges = np.unique(np.concatenate(self.ncharges))
        self.max_natoms = max([len(x) for x in self.ncharges])

        for elem in pt.keys():
            counts = []
            for ncharge_list in self.ncharges:
                count = np.count_nonzero(ncharge_list == pt[elem])
                counts.append(count)

                # keep max count
                if counts:
                    self.atomtype_dict[elem] = max(counts)
                else:
                    self.atomtype_dict[elem] = 1

        return

    def get_proparg_data(self):
        data = pd.read_csv("data/Proparg-21-TS/data.csv", index_col=0)
        reactants_files = [
            "data/Proparg-21-TS/data_react_xyz/"
            + data.mol.values[i]
            + data.enan.values[i]
            + ".xyz"
            for i in range(len(data))
        ]
        products_files = [
            "data/Proparg-21-TS/data_prod_xyz/"
            + data.mol.values[i]
            + data.enan.values[i]
            + ".xyz"
            for i in range(len(data))
        ]
        all_mols = [qml.Compound(x) for x in reactants_files + products_files]
        self.barriers = data.dErxn.to_numpy()
        self.ncharges = [mol.nuclear_charges for mol in all_mols]
        self.unique_ncharges = np.unique(np.concatenate(self.ncharges))
        self.max_natoms = max([len(x) for x in self.ncharges])

        # atomtype dict for BoB
        for elem in pt.keys():
            counts = []
            for ncharge_list in self.ncharges:
                count = np.count_nonzero(ncharge_list == pt[elem])
                counts.append(count)

                # keep max count
                if counts:
                    self.atomtype_dict[elem] = max(counts)
                else:
                    self.atomtype_dict[elem] = 1

        self.mols_reactants = [[qml.Compound(x)] for x in reactants_files]
        self.mols_products = [[qml.Compound(x)] for x in products_files]

        return

    def get_hydroform_data(self):
        co_df = pd.read_csv("data/Hydroform-22-TS/Co_clean.csv")
        names = co_df["name"].to_list()
        labels = [name[3:] for name in names]
        co_reactants = [
            "data/Hydroform-22-TS/geometries/co/r/" + label + "_reactant.xyz"
            for label in labels
        ]
        co_products = [
            "data/Hydroform-22-TS/geometries/co/p/" + label + "_product.xyz"
            for label in labels
        ]
        self.co_barriers = co_df["f_barr"].to_numpy()
        self.mols_reactants_co = [[qml.Compound(x)] for x in co_reactants]
        self.mols_products_co = [[qml.Compound(x)] for x in co_products]

        ir_df = pd.read_csv("data/Hydroform-22-TS/Ir_clean.csv")
        names = ir_df["name"].to_list()
        labels = [name[3:] for name in names]
        ir_reactants = [
            "data/Hydroform-22-TS/geometries/ir/r/" + label + "_reactant.xyz"
            for label in labels
        ]
        ir_products = [
            "data/Hydroform-22-TS/geometries/ir/p/" + label + "_product.xyz"
            for label in labels
        ]
        self.ir_barriers = ir_df["f_barr"].to_numpy()
        self.mols_reactants_ir = [[qml.Compound(x)] for x in ir_reactants]
        self.mols_products_ir = [[qml.Compound(x)] for x in ir_products]

        rh_df = pd.read_csv("data/Hydroform-22-TS/Rh_clean.csv")
        names = rh_df["name"].to_list()
        labels = [name[3:] for name in names]
        rh_reactants = [
            "data/Hydroform-22-TS/geometries/rh/r/" + label + "_reactant.xyz"
            for label in labels
        ]
        rh_products = [
            "data/Hydroform-22-TS/geometries/rh/p/" + label + "_product.xyz"
            for label in labels
        ]
        self.rh_barriers = rh_df["f_barr"].to_numpy()
        self.mols_reactants_rh = [[qml.Compound(x)] for x in rh_reactants]
        self.mols_products_rh = [[qml.Compound(x)] for x in rh_products]

        self.barriers = np.concatenate(
            (self.co_barriers, self.ir_barriers, self.rh_barriers), axis=0
        )

        all_reactants = co_reactants + ir_reactants + rh_reactants
        all_products = co_products + ir_products + rh_products
        list_reactants = [qml.Compound(x) for x in all_reactants]

        self.mols_reactants = [[qml.Compound(x)] for x in all_reactants]
        self.mols_products = [[qml.Compound(x)] for x in all_products]

        self.ncharges = [mol.nuclear_charges for mol in list_reactants]
        self.unique_ncharges = np.unique(np.concatenate(ncharges))
        self.max_natoms = max([len(x) for x in ncharges])

        # atomtype dict for BoB
        for elem in pt.keys():
            counts = []
            for ncharge_list in self.ncharges:
                count = np.count_nonzero(ncharge_list == pt[elem])
                counts.append(count)

                # keep max count
                if counts:
                    self.atomtype_dict[elem] = max(counts)
                else:
                    self.atomtype_dict[elem] = 1

        return

    def get_CM(self):
        cm_reactants = [
            np.array(
                [
                    qml.representations.generate_coulomb_matrix(
                        x.nuclear_charges, x.coordinates, size=self.max_natoms
                    )
                    for x in reactants
                ]
            )
            for reactants in self.mols_reactants
        ]
        cm_reactants = np.array([np.concatenate(x, axis=0) for x in cm_reactants])
        cm_products = [
            np.array(
                [
                    qml.representations.generate_coulomb_matrix(
                        x.nuclear_charges, x.coordinates, size=self.max_natoms
                    )
                    for x in products
                ]
            )
            for products in self.mols_products
        ]
        cm_products = np.array([np.concatenate(x, axis=0) for x in cm_products])

        cm_r_p = np.concatenate((cm_reactants, cm_products), axis=1)
        return cm_reactants, cm_products, cm_r_p

    def get_BoB(self):
        bob_reactants = [
            np.array(
                [
                    qml.representations.generate_bob(
                        x.nuclear_charges,
                        x.coordinates,
                        self.unique_ncharges,
                        size=self.max_natoms,
                        asize=self.atomtype_dict,
                    )
                    for x in reactants
                ]
            )
            for reactants in self.mols_reactants
        ]
        bob_reactants = np.array([np.concatenate(x, axis=0) for x in bob_reactants])
        bob_products = [
            np.array(
                [
                    qml.representations.generate_bob(
                        x.nuclear_charges,
                        x.coordinates,
                        self.unique_ncharges,
                        size=self.max_natoms,
                        asize=self.atomtype_dict,
                    )
                    for x in products
                ]
            )
            for products in self.mols_products
        ]
        bob_products = np.array([np.concatenate(x, axis=0) for x in bob_products])

        bob_r_p = np.concatenate((bob_reactants, bob_products), axis=1)
        return bob_reactants, bob_products, bob_r_p

    def get_FCHL19(self):
        fchl_reactants = [
            np.array(
                [
                    qml.representations.generate_fchl_acsf(
                        x.nuclear_charges,
                        x.coordinates,
                        elements=self.unique_ncharges,
                        gradients=False,
                        pad=self.max_natoms,
                    )
                    for x in reactants
                ]
            )
            for reactants in self.mols_reactants
        ]
        fchl_reactants_sum = np.array([sum(sum(x)) for x in fchl_reactants])

        fchl_products = [
            np.array(
                [
                    qml.representations.generate_fchl_acsf(
                        x.nuclear_charges,
                        x.coordinates,
                        elements=self.unique_ncharges,
                        gradients=False,
                        pad=self.max_natoms,
                    )
                    for x in products
                ]
            )
            for products in self.mols_products
        ]
        fchl_products = np.array([sum(sum(x)) for x in fchl_products])

        fchl_diff = fchl_products - fchl_reactants_sum

        return fchl_reactants_sum, fchl_products, fchl_diff

    def get_SLATM(self):
        mbtypes = qml.representations.get_slatm_mbtypes(self.ncharges)

        slatm_reactants = [
            np.array(
                [
                    qml.representations.generate_slatm(
                        x.coordinates, x.nuclear_charges, mbtypes, local=False
                    )
                    for x in reactants
                ]
            )
            for reactants in self.mols_reactants
        ]

        slatm_reactants_sum = np.array([sum(x) for x in slatm_reactants])
        slatm_products = [
            np.array(
                [
                    qml.representations.generate_slatm(
                        x.coordinates, x.nuclear_charges, mbtypes, local=False
                    )
                    for x in products
                ]
            )
            for products in self.mols_products
        ]
        slatm_products = np.array([sum(x) for x in slatm_products])
        slatm_diff = slatm_products - slatm_reactants_sum

        return slatm_reactants_sum, slatm_products, slatm_diff

    def get_SLATM_twobody(self):
        slatm_reactants = [
            np.array(
                [
                    get_slatm(
                        x.nuclear_charges,
                        x.coordinates,
                        elements=self.unique_ncharges,
                    )
                    for x in reactants
                ]
            )
            for reactants in self.mols_reactants
        ]

        slatm_reactants_sum = np.array([sum(x) for x in slatm_reactants])
        slatm_products = [
            np.array(
                [
                    get_slatm(
                        x.nuclear_charges,
                        x.coordinates,
                        elements=self.unique_ncharges,
                    )
                    for x in products
                ]
            )
            for products in self.mols_products
        ]
        slatm_products = np.array([sum(x) for x in slatm_products])
        slatm_diff = slatm_products - slatm_reactants_sum

        return slatm_reactants_sum, slatm_products, slatm_diff


class DScribe:
    """For SOAP via DScribe"""

    def __init__(self):
        self.unique_ncharges = []
        self.atoms_reactants = [[]]
        self.atoms_products = [[]]
        self.energies = []
        self.barriers = []

        return

    def get_sn2_data(self):
        reactions = pd.read_csv("data/SN2-20/reactions.csv", index_col=0)
        reactions["reactant"] = reactions["reactant"].apply(literal_eval)
        reactions["product"] = reactions["product"].apply(literal_eval)
        self.energies = reactions["rxn_nrj"].to_numpy()

        all_r_files = [
            x for reactions in reactions["reactant"].to_list() for x in reactions
        ]
        all_p_files = [
            x for reactions in reactions["product"].to_list() for x in reactions
        ]
        all_files = list(set(all_r_files)) + list(set(all_p_files))
        all_atoms = [xyz_to_atomsobj(x) for x in all_files]
        ncharges = [a.numbers for a in all_atoms]
        self.unique_ncharges = np.unique(np.concatenate(ncharges))

        self.atoms_reactants = [
            [xyz_to_atomsobj(x) for x in reactants]
            for reactants in reactions["reactant"].to_list()
        ]
        self.atoms_products = [
            [xyz_to_atomsobj(x) for x in products]
            for products in reactions["product"].to_list()
        ]
        return

    def get_gdb7_rxn_data(self):
        reactions = pd.read_csv("data/GDB7-20-TS/dataset.csv")
        self.barriers = reactions["ea kcal/mol"].to_numpy()
        all_r_files = [
            "data/GDB7-20-TS/xyz/" + x for x in reactions["reactant"].to_list()
        ]
        all_p_files = [
            "data/GDB7-20-TS/xyz/" + x for x in reactions["product"].to_list()
        ]
        self.atoms_reactants = [[xyz_to_atomsobj(x)] for x in all_r_files]
        self.atoms_products = [[xyz_to_atomsobj(x)] for x in all_p_files]
        ncharges = [x[0].numbers for x in self.atoms_reactants]
        self.unique_ncharges = np.unique(np.concatenate(ncharges))

        return

    def get_proparg_data(self):
        data = pd.read_csv("data/Proparg-21-TS/data.csv", index_col=0)
        reactants_files = [
            "data/Proparg-21-TS/data_react_xyz/"
            + data.mol.values[i]
            + data.enan.values[i]
            + ".xyz"
            for i in range(len(data))
        ]
        products_files = [
            "data/Proparg-21-TS/data_prod_xyz/"
            + data.mol.values[i]
            + data.enan.values[i]
            + ".xyz"
            for i in range(len(data))
        ]

        self.barriers = data.dErxn.to_numpy()
        all_files = reactants_files + products_files
        all_atoms = [xyz_to_atomsobj(x) for x in all_files]
        ncharges = [a.numbers for a in all_atoms]
        self.unique_ncharges = np.unique(np.concatenate(ncharges))

        self.atoms_reactants = [[xyz_to_atomsobj(x)] for x in reactants_files]
        self.atoms_products = [[xyz_to_atomsobj(x)] for x in products_files]
        return

    def get_hydroform_data(self):
        co_df = pd.read_csv("data/Hydroform-22-TS/Co_clean.csv")
        names = co_df["name"].to_list()
        labels = [name[3:] for name in names]
        co_reactants = [
            "data/Hydroform-22-TS/geometries/co/r/" + label + "_reactant.xyz"
            for label in labels
        ]
        co_products = [
            "data/Hydroform-22-TS/geometries/co/p/" + label + "_product.xyz"
            for label in labels
        ]
        self.co_barriers = co_df["f_barr"].to_numpy()
        self.atoms_reactants_co = [[xyz_to_atomsobj(x)] for x in co_reactants]
        self.atoms_products_co = [[xyz_to_atomsobj(x)] for x in co_products]

        ir_df = pd.read_csv("data/Hydroform-22-TS/Ir_clean.csv")
        names = ir_df["name"].to_list()
        labels = [name[3:] for name in names]
        ir_reactants = [
            "data/Hydroform-22-TS/geometries/ir/r/" + label + "_reactant.xyz"
            for label in labels
        ]
        ir_products = [
            "data/Hydroform-22-TS/geometries/ir/p/" + label + "_product.xyz"
            for label in labels
        ]
        self.ir_barriers = ir_df["f_barr"].to_numpy()
        self.atoms_reactants_ir = [[xyz_to_atomsobj(x)] for x in ir_reactants]
        self.atoms_products_ir = [[xyz_to_atomsobj(x)] for x in ir_products]

        rh_df = pd.read_csv("data/Hydroform-22-TS/Rh_clean.csv")
        names = rh_df["name"].to_list()
        labels = [name[3:] for name in names]
        rh_reactants = [
            "data/Hydroform-22-TS/geometries/rh/r/" + label + "_reactant.xyz"
            for label in labels
        ]
        rh_products = [
            "data/Hydroform-22-TS/geometries/rh/p/" + label + "_product.xyz"
            for label in labels
        ]
        self.rh_barriers = rh_df["f_barr"].to_numpy()
        self.atoms_reactants_rh = [[xyz_to_atomsobj(x)] for x in rh_reactants]
        self.atoms_products_rh = [[xyz_to_atomsobj(x)] for x in rh_products]

        all_reactants = co_reactants + ir_reactants + rh_reactants
        all_products = co_products + ir_products + rh_products
        list_reactants = [xyz_to_atomsobj(x) for x in all_reactants]
        self.atoms_reactants = [[xyz_to_atomsobj(x)] for x in all_reactants]
        self.atoms_products = [[xyz_to_atomsobj(x)] for x in all_products]

        self.barriers = np.concatenate(
            (self.co_barriers, self.ir_barriers, self.rh_barriers)
        )
        ncharges = [a.numbers for a in list_reactants]
        self.unique_ncharges = np.unique(np.concatenate(ncharges))
        self.max_natoms = max([len(x) for x in ncharges])

        return

    def get_SOAP(self):
        soap = SOAP(
            species=self.unique_ncharges,
            rcut=5.0,
            nmax=8,
            lmax=8,
            sigma=0.2,
            periodic=False,
            crossover=True,
            sparse=False,
        )
        soap_reactants = [
            np.array([soap.create(x) for x in reactants])
            for reactants in self.atoms_reactants
        ]
        soap_reactants_global = [
            np.array([np.sum(x, axis=0) for x in reactants])
            for reactants in soap_reactants
        ]
        soap_reactants_sum = np.array([sum(x) for x in soap_reactants_global])

        soap_products = [
            np.array([soap.create(x) for x in products])
            for products in self.atoms_products
        ]
        soap_products_global = [
            np.array([np.sum(x, axis=0) for x in products])
            for products in soap_products
        ]
        soap_products_global = np.array([sum(x) for x in soap_products_global])
        soap_diff = soap_products_global - soap_reactants_sum

        return soap_reactants_sum, soap_products_global, soap_diff
