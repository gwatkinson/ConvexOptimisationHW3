# Convex Optimisation HW3

This is the repo for the HW3 of the Convex Optimisation course of the MVA master.

It contains a theoritical part and a practical one. Both will be in the final notebook.


## Reproducing the results

### Environment

You can recreate the exact same conda environment from the lock files in the [`lock_files`](https://github.com/gwatkinson/HW3/tree/main/lock_files) folder :

```
conda create -n <YOURENV> --file lock_files/conda-<OS>.lock

conda activate <YOURENV>
```

The main packages are numpy, cvxpy and plotting libraries.

If the lock file doesn't work, you can install the package with 

```
conda create -n <YOURENV> --file lock_files/environment.yml
```
But the version might be different.

### Script
To reproduce the results presented in the final notebook, you can run the [`final.py`](https://github.com/gwatkinson/HW3/blob/main/final.py) file:

```
    python final.py
```

It also contains the two requested functions:

```python
    from final import centering_step, barr_method
```

You can also look at the [`final.ipynb`](https://github.com/gwatkinson/HW3/blob/main/final.ipynb) notebook, that is the notebook form of the previous file. The results can thus be run interactively.

Finally, [`report.ipynb`](https://github.com/gwatkinson/HW3/blob/main/report.ipynb) is the same as [`final.ipynb`](https://github.com/gwatkinson/HW3/blob/main/final.ipynb) but reordered, so it won't run by default (the cell ordering is wrong).

### Plots

If you want to go further, you can look at the [`test.ipynb`](https://github.com/gwatkinson/HW3/blob/main/test.ipynb) notebook that uses the package developed in [`src`](https://github.com/gwatkinson/HW3/tree/main/src). This is the notebook that produced the plots shown in the report.

The plot are generated in the first section, but other experiments are run in this notebook.

## Package

The package developed is in the [`src`](https://github.com/gwatkinson/HW3/tree/main/src) folder.

## Report 

You can find the pdf report that I sent in [`pdfs/FinalReport.pdf`](https://github.com/gwatkinson/HW3/blob/main/pdfs/FinalReport.pdf).
