# Incomplete-Heter-Kernel

The primary objective of this repository is to demonstrate the performance of models on datasets with incomplete data.

The models (`PMK` and `PMK_KPCA`) are specifically designed to handle heterogeneous data types and various missing data mechanisms (MCAR, MAR, MNAR). 


## Models

`MPK`: Our proposed method kernel-based model designed to handle incomplete data.

`MPK_KPCA`: A combination of MPK with Kernel Principal Component Analysis (KPCA) to handle complex data structures.


## Running Classification Tasks

To run classification tasks on the heart dataset using a missing rate of 10%:


```python main.py --datasets heart --models PMK --missing_types mcar --missing_rates 0.1```


## Running Clustering Tasks


For kidney:

```
python main.py --datasets kidney
```
For mammo:
```
python main.py --datasets mammo
```
