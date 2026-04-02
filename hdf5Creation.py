#import statements for pandas and hdf5 libraries
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import h5py
import glob

from featureExtraction import segment_dataframe


#Create a new hdf5 file with the name hdf5_data.h5 in write mode
with h5py.File('./hdf5_data.h5', 'w') as hdf:
    # Create HDF5 Group and Subgroups for Raw Data
    G11 = hdf.create_group('/Raw Data/Brett')
    G12 = hdf.create_group('/Raw Data/Logan')
    G13 = hdf.create_group('/Raw Data/Vince')

    # create datasets with all of Brett's raw data under subgroup Brett
    for i in range(1, 6):
        G11.create_dataset('walking' + str(i), data=pd.read_csv(f'Raw_Data/Brett_RawData/Walk-' + str(i) + '_RawData.csv'))
        G11.create_dataset('jumping' + str(i), data=pd.read_csv(f'Raw_Data/Brett_RawData/Jump-' + str(i) + '_RawData.csv'))

    # create datasets with all of Logan's raw data under subgroup Logan
    # Sorting files into alphhabetical order to ensure consistency with labels for data with preprocessing later
    lFiles = sorted(glob.glob('Raw_Data/Logan_data/*/*.csv'))
    for i in range(1, 6):
        G12.create_dataset('walking' + str(i), data=pd.read_csv(lFiles[i + 4]))
        G12.create_dataset('jumping' + str(i), data=pd.read_csv(lFiles[i - 1]))

    # create datasets with all of Vince's raw data under subgroup Vince
    vFiles = sorted(glob.glob('Raw_Data/Vince_Data/*/*.csv'))
    for i in range(1, 6):
        G13.create_dataset('walking' + str(i), data=pd.read_csv(vFiles[i + 4]))
        G13.create_dataset('jumping' + str(i), data=pd.read_csv(vFiles[i - 1]))

    # Create HDF5 group and subgroups for preprocessed data
    G21 = hdf.create_group('/Pre-Processed Data/Brett')
    G22 = hdf.create_group('/Pre-Processed Data/Logan')
    G23 = hdf.create_group('/Pre-Processed Data/Vince')

    # Adds all of the preprocessed data to the HDF5 file in their respecting subgroups
    bFiles = sorted(glob.glob('Pre-Processed_Data/Brett/*.csv'))
    lFiles = sorted(glob.glob('Pre-Processed_Data/Logan/*.csv'))
    vFiles = sorted(glob.glob('Pre-Processed_Data/Vince/*.csv'))
    for i in range(1, 6):
        G21.create_dataset('walk' + str(i), data=pd.read_csv(bFiles[i + 4]))
        G21.create_dataset('jump' + str(i), data=pd.read_csv(bFiles[i - 1]))
        G22.create_dataset('walk' + str(i), data=pd.read_csv(lFiles[i + 4]))
        G22.create_dataset('jump' + str(i), data=pd.read_csv(lFiles[i - 1]))
        G23.create_dataset('walk' + str(i), data=pd.read_csv(vFiles[i + 4]))
        G23.create_dataset('jump' + str(i), data=pd.read_csv(vFiles[i - 1]))

    # Create HDF5 groups/subgroups for train/test splits
    G31_walk = hdf.create_group('/Segmented data/Train/walking')
    G31_jump = hdf.create_group('/Segmented data/Train/jumping')
    G32_walk = hdf.create_group('/Segmented data/Test/walking')
    G32_jump = hdf.create_group('/Segmented data/Test/jumping')

    all_windows = []  # each entry: (numpy_array, label_string)

    # segment signals into approximately 5-second windows
    for i in range(10):
        if i <= 4:
            label = 'jumping'
        else:
            label = 'walking'

        #segment a csv file from each group member
        bWindows = segment_dataframe(pd.read_csv(bFiles[i]))
        lWindows = segment_dataframe(pd.read_csv(lFiles[i]))
        vWindows = segment_dataframe(pd.read_csv(vFiles[i]))
        windows = bWindows + lWindows + vWindows
        for w in windows:
            all_windows.append((w.values, label))

    # Separate indices, shuffle, then split 90 / 10
    indices = list(range(len(all_windows)))
    train_idx, test_idx = train_test_split(indices, test_size=0.1, shuffle=True, random_state=42)

    # Counters so each dataset gets a unique name within its group
    counters = {
        ('Train', 'walking'): 0,
        ('Train', 'jumping'): 0,
        ('Test', 'walking'): 0,
        ('Test', 'jumping'): 0,
    }
    group_map = {
        ('Train', 'walking'): G31_walk,
        ('Train', 'jumping'): G31_jump,
        ('Test', 'walking'): G32_walk,
        ('Test', 'jumping'): G32_jump,
    }


    def save_windows(idx_list, split_name):
        for idx in idx_list:
            arr, label = all_windows[idx]
            key = (split_name, label)
            name = f'window_{counters[key]}'
            group_map[key].create_dataset(name, data=arr)
            counters[key] += 1


    save_windows(train_idx, 'Train')
    save_windows(test_idx, 'Test')
