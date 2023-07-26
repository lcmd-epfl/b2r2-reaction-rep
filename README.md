# B2R2 Reaction Representation

This repository provides the means to:
- Generate the B2R2 reaction representations 
- Generate reaction representations from molecular representations (SLATM, FCHL, etc.)
- Access 4 reaction datasets (SN2-20, GDB7-20-TS, Proparg-21-TS and Hydroform-22-TS)

An example generating learning curves for all reaction reps for the SN2 dataset is provided in `paper_sn2_example.ipynb`. 

An example using a dataset specified by a path is provided in `own_dataset_example.ipynb`.

## Handling experimental data 
While the reaction rep is formulated in terms of reactants and products, reagents may also be specified by adding them to the list of reactants.

Currently, the expected input is .xyz files (cheap estimate of 3D structure) of reactants (+reagents) and products. Soon, a module will be provided to perform the 2D->3D conversion for you, but for now, please look into the tools available through `rdkit` or `obabel`.

## How to install
A `requirements.txt` file is provided which includes all required packages to run all examples.
Note that the order in the requirements file matters, so installation can be done like:
```
xargs -L 1 pip install < requirements.txt
```

## Citation
If using the B2R2 reaction representation, consider citing the paper with the bibtex provided:
```
@article{vangerwen2022physics,
  title={Physics-based representations for machine learning properties of chemical reactions},
  author={van Gerwen, Puck and Fabrizio, Alberto and Wodrich, Matthew D and Corminboeuf, Clemence},
  journal={Machine Learning: Science and Technology},
  volume={3},
  number={4},
  pages={045005},
  year={2022},
  publisher={IOP Publishing},
  doi={10.1088/2632-2153/ac8f1a}
}

```

## Todo
- 2D -> 3D support
- Usage for experimental data including conditions
- Speed-up (fortran or numba)
