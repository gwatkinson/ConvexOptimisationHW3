# Convex Optimisation HW3

This is the repo for the HW3 of the Convex Optimisation course of the MVA master.

It contains a theoritical part and a practical one. Both will be in the final notebook.


## Reproducing the results

### Environment

You can recreate the exact same conda environment from the lock files:

```
conda create -n <YOURENV> --file conda-<OS>.lock

conda activate <YOURENV>
```

The main packages are numpy, cvxpy and plotting libraries.

If the lock file doesn't work, you can install the package with 

```
conda create -n <YOURENV> --file environment.yml
```
But the version might be different.

### Script
To reproduce the results presented in the final notebook, you can run the `final.py` file:

```
    python final.py
```

It also contains the two requested functions:

```python
    from final import centering_step, barr_method
```

You can also look at the `final.ipynb` notebook, that is the notebook form of the previous file. The results can thus be run interactively.

Finally, `report.ipynb` is the same as `final.ipynb` but reordered, so it won't run by default (the cell ordering is wrong).

### Plots

If you want to go further, you can look at the `test.ipynb` notebook that uses the package developed in `src`. This is the notebook that produced the plots shown in the report.

The plot are generated in the first section, but other experiments are run in this notebook.

## Package

The package developed is in the `src` folder.
