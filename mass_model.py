import numpy as np
from pmk import PMK_Kernel



def run_impk(train_X, test_X, data_stats):

    param_value = None  # use default: log2(num of inst) + 1
    impk_krn = PMK_Kernel(param_value, data_stats)
    impk_krn.set_nbins(param_value)
    train, test = impk_krn.build_model(train_X, test_X)  # this does the pre-processing step
    print("- Sim: Train")
    sim_train = impk_krn.transform(train)
    print("- Sim: Train/Test")
    sim_test = impk_krn.transform(train,test)

    return sim_train, sim_test.T