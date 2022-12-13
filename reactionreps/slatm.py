import itertools

import numpy as np


def gaussian(x, sigma=0.05):
    coeff = 1 / (sigma * np.sqrt(2 * np.pi))

    return coeff * np.exp(-(x**2) / (2 * sigma**2))


def get_twobody(
    ncharge_a, ncharge_b, coords_a, coords_b, sigma=0.05, rcut=4.8, gridspace=0.03
):
    r0 = 0.1
    nx = int((rcut - r0) / gridspace) + 1
    grid = np.linspace(r0, rcut, nx)

    distance = np.linalg.norm(coords_b - coords_a)
    if distance < rcut:
        potential = (
            0.5
            * gaussian(grid - distance, sigma=sigma)
            * (1 / grid**6)
            * ncharge_b
            * ncharge_a
            * gridspace
        )
    else:
        potential = np.zeros(len(grid))

    return potential


def get_bags(unique_ncharges):
    combs = list(itertools.combinations(unique_ncharges, r=2))
    combs = [list(x) for x in combs]
    # add self interaction
    self_combs = [[x, x] for x in unique_ncharges]
    return combs + self_combs


def get_slatm(
    ncharges, coords, sigma=0.05, rcut=4.8, gridspace=0.03, elements=[1, 6, 7, 8, 9, 17]
):
    for ncharge in ncharges:
        if ncharge not in elements:
            print("warning!", ncharge, "not included in rep")

    bags = get_bags(elements)
    size = len(np.arange(0.1, rcut, gridspace))
    twobodyrep = np.zeros((len(bags), size))

    for k, bag in enumerate(bags):
        for i, ncharge_a in enumerate(ncharges):
            for j in range(i):
                ncharge_b = ncharges[j]
                # check whether to use it
                bag_candidate = [ncharge_a, ncharge_b]
                inv_bag_candidate = [ncharge_b, ncharge_a]
                if bag == bag_candidate or bag == inv_bag_candidate:
                    potential = get_twobody(
                        ncharge_a,
                        ncharge_b,
                        coords[i],
                        coords[j],
                        sigma=sigma,
                        rcut=rcut,
                        gridspace=gridspace,
                    )
                    twobodyrep[k] += potential

    # add one and two body
    twobodyrep = np.concatenate(twobodyrep)

    onebodyrep = np.zeros(len(elements))
    for i, elem in enumerate(elements):
        onebodyrep[i] += elem * (ncharges == elem).sum()

    rep = np.concatenate((onebodyrep, twobodyrep))
    assert len(rep) == len(elements) + size * len(bags)
    return rep
