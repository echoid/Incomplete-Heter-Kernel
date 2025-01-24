# Incomplete-Heter-Kernel

The primary objective of this repository is to demonstrate the performance of models on datasets with incomplete data.

The model (`PMK`) is specifically designed to handle heterogeneous data types and various missing data mechanisms (MCAR, MAR, MNAR). 


## Model

`MPK`: Our proposed method kernel-based model designed to handle incomplete data.


## Running Classification Tasks

To run classification tasks on the heart dataset using a missing rate of 5%:


```python main.py --datasets australian --missing_types mcar --missing_rates 0.1```


## Running Clustering Tasks


For kidney:

```
python main.py --datasets kidney
```
For mammo:
```
python main.py --datasets mammo
```
