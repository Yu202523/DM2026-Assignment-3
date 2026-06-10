# NYCU Data Mining Assignment 3

This repository contains the final Kaggle notebook used for our Assignment 3 solution on sequential accelerometer activity classification.

## Final notebook

The final reproducible notebook is:

- `assignment3_kaggle_notebook_v74_final_submit.ipynb`

This notebook implements our final `v7.4` solution, which combines:
- a tabular tree-based ensemble (LightGBM + XGBoost),
- a CNN+BiGRU sequence model,
- and a LightGBM meta-stacking model for final prediction fusion.

## Recommended environment

This notebook is designed to run on **Kaggle Notebook**.

Recommended settings:
- GPU enabled
- Python 3
- Kaggle default package environment

Main dependencies used in the notebook:
- `numpy`
- `pandas`
- `scikit-learn`
- `lightgbm`
- `xgboost`
- `torch`
- `scipy`
- `tqdm`

## Dataset setup

Please attach the assignment dataset in Kaggle so that the notebook can discover the following structure:

```text
sample_submission.csv
train/train/User_xxx/*.csv
test/test/User_xxx/*.csv
