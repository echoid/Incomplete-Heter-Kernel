# Incomplete-Heter-Kernel

This repository provides the implementation of **Heterogeneous Incomplete Probability Mass Kernel (HI-PMK)**, a novel kernel-based method designed to handle incomplete data across heterogeneous data types and various missing mechanisms.

## Objective

The primary objective of this repository is to:
1. Demonstrate the performance of models, particularly `HI-PMK`, on datasets with incomplete data.
2. Evaluate `HI-PMK` for classification and clustering tasks under different missing mechanisms:
   - **MCAR**: Missing Completely At Random
   - **MAR**: Missing At Random
   - **MNAR**: Missing Not At Random

## Features

- **Model**: `HI-PMK`
  - Handles heterogeneous data (numerical and categorical).
  - Does not require imputation.
  - Effectively addresses diverse missing data mechanisms.

- **Tasks**:
  - **Classification**: Validate the utility of `HI-PMK` for supervised learning.
  - **Clustering**: Test the robustness of `HI-PMK` in unsupervised learning.


The implementation provided in this repository is **untuned** and primarily aimed at reproducibility. It serves as a foundation for further research and experimentation:
- Hyperparameters are not optimized.
- Results may vary depending on the environment or parameter adjustments.



## Running Classification Tasks


To run classification tasks, use the following command:

Example: Run on the Australian dataset with a 10% missing rate under MCAR:

```python main.py --datasets australian --missing_types mcar --missing_rates 0.1```


## Running Clustering Tasks


To run clustering tasks on specific datasets:

For mammo:
```
python main.py --datasets mammo
```

For kidney:

```
python main.py --datasets kidney
```

