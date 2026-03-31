import matplotlib.pyplot as plt
import pandas as pd
import h5py
import numpy as np

with h5py.File('./hdf5_data.h5', 'r') as hdf:
    #for each subgroup in Pre-Processed Data:
    #   for each file in the subgroup:
    #       extract features
    Group = hdf.get('Pre-Processed Data')
    ls = list(Group.keys())
    print(ls)
    for sg in ls:
        files = list(Group.get(sg).keys())
        print(files)
        for file in files:
            #extract data here
            print("hi")


