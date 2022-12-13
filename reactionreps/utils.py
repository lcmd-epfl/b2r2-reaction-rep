import ase


def xyz_to_atomsobj(xyz):
    """Convert xyz file to ase atoms object.

    Args:
        xyz (str): path to xyz file

    Returns:
        atomsobj (ase atoms object): ase atoms object
    """
    with open(xyz, "r") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    nat = int(lines[0])

    symbols = []
    positions = []
    for line in lines[2 : 2 + nat]:
        symbol, x, y, z = line.split()
        symbols.append(symbol)
        positions.append([float(x), float(y), float(z)])

    atomsobj = ase.Atoms(symbols=symbols, positions=positions)
    return atomsobj
