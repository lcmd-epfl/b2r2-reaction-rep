# B2R2 Reaction Representation

This repo accompanies the paper "Physics-based representations for machine learning of chemical reactions".

This repository provides the means to:
- Generate the B2R2 reaction representations (in `src/reaction_reps`)
- Generate reaction representations from molecular representations (SLATM, FCHL, etci.)
- Access 4 reaction datasets (SN2-20, GDB7-20-TS, Proparg-21-TS and Hydroform-22-TS)

An example generating learning curves for all reaction reps for the SN2 dataset is provided in `paper_sn2_example.ipynb`. 


# How to use

A `conda` environment is provided (`b2r2.yaml`) which can be created by running:

```conda env create --file b2r2.yaml```

The resulting b2r2 environment has all the necessary dependencies and can be used to access all functionalities. To set the environment up in `jupyter`, run:

```
conda install ipykernel
python -m ipykernel install --user --name=b2r2
```

which will then let you select the b2r2 kernel to run. This is the recommended way to proceed.

Otherwise, dependencies are:
- `pandas`
- `numpy`
- `scipy`
- `ase`
- `dscribe`
- `qml` (version  `0.4.0.12` or newer from the develop branch )

The latter is required to access the `fchl` module of `qml`. Follow installation instructions [here](http://www.qmlcode.org/installation.html).
We have successfully tested our code with several modern versions of all other dependencies.


# Todo

- Usage for experimental data including conditions
- Speed-up (fortran or numba)
