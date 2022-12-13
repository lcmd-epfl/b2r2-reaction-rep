# B2R2 Reaction Representation

This repo accompanies the paper "Physics-based representations for machine learning of chemical reactions".

This repository provides the means to:
- Generate the B2R2 reaction representations (in `src/reaction_reps`)
- Generate reaction representations from molecular representations (SLATM, FCHL, etci.)
- Access 4 reaction datasets (SN2-20, GDB7-20-TS, Proparg-21-TS and Hydroform-22-TS)
- For the latter (Hydroform-22-TS), this is the only place to find the dataset at the moment

An example generating learning curves for all reaction reps for the SN2 dataset is provided in `paper_sn2_example.ipynb`. 


# How to install
A `requirements.txt` file is provided which includes all required packages to run all examples.
Note that the order in the requirements file matters, so installation can be done like:
```
cat requirements.txt | xargs pip install
```

# Citation
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

# Todo

- Usage for experimental data including conditions
- Speed-up (fortran or numba)
