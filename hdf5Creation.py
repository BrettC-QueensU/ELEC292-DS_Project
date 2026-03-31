#import statements for pandas and hdf5 libraries
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import h5py
import glob

#Create a new hdf5 file with the name hdf5_data.h5 in write mode
with h5py.File('./hdf5_data.h5', 'w') as hdf:
    #Create first group and subgroup with Brett's data
    G11= hdf.create_group('/Raw Data/Brett')
    #create datasets with all of Brett's raw data under subgroup Brett
    for i in range(1, 6):
        G11.create_dataset('walking' + str(i), data=pd.read_csv(f'Raw_Data/Brett_RawData/Walk-' + str(i) + '_RawData.csv'))
        G11.create_dataset('jumping' + str(i), data=pd.read_csv(f'Raw_Data/Brett_RawData/Jump-' + str(i) + '_RawData.csv'))

    # Create second subgroup with Logan's data
    G12 = hdf.create_group('/Raw Data/Logan')
    # create datasets with all of Logan's raw data under subgroup Logan
    # Sorting files into alphhabetical order to ensure consistency with labels for data with preprocessing later
    lFiles = sorted(glob.glob('Raw_Data/Logan_data/*/*.csv'))
    for i in range(1, 6):
        G12.create_dataset('walking' + str(i), data=pd.read_csv(lFiles[i + 4]))
        G12.create_dataset('jumping' + str(i), data=pd.read_csv(lFiles[i - 1]))

    # create third subgroup under Raw Data with Vince's data
    G13 = hdf.create_group('/Raw Data/Vince')
    # create datasets with all of Vince's raw data under subgroup Vince
    vFiles = sorted(glob.glob('Raw_Data/Vince_Data/*/*.csv'))
    for i in range(1, 6):
        G13.create_dataset('walking' + str(i), data=pd.read_csv(vFiles[i + 4]))
        G13.create_dataset('jumping' + str(i), data=pd.read_csv(vFiles[i - 1]))

    # Create group pre-processed data and 3 subgroups for each member's data
    G21 = hdf.create_group('/Pre-Processed Data/Brett')
    for i in range(1, 6):
        G21.create_dataset('walk' + str(i), data=pd.read_csv('Pre-Processed_Data/Brett/Walk-' + str(i) + '_PreProcessed.csv'))
        G21.create_dataset('jump' + str(i), data=pd.read_csv('Pre-Processed_Data/Brett/Jump-' + str(i) + '_PreProcessed.csv'))

    G22 = hdf.create_group('/Pre-Processed Data/Logan')
    lFiles = sorted(glob.glob('Pre-Processed_Data/Logan/*.csv'))
    for i in range(1, 6):
        G22.create_dataset('walk' + str(i), data=pd.read_csv(lFiles[i + 4]))
        G22.create_dataset('jump' + str(i), data=pd.read_csv(lFiles[i - 1]))

    G23 = hdf.create_group('/Pre-Processed Data/Vince')
    vFiles = sorted(glob.glob('Pre-Processed_Data/Vince/*.csv'))
    for i in range(1, 6):
        G23.create_dataset('walk' + str(i), data=pd.read_csv(vFiles[i + 4]))
        G23.create_dataset('jump' + str(i), data=pd.read_csv(vFiles[i - 1]))


    #lastly, create group segmented data with subgroups train and test
    #iterate through the pre processed data and resample each file into a
    #dataframe with 5 second intervals.
    Group = hdf.get('Pre-Processed Data')
    ls = list(Group.keys())
    print(ls)
    all_data = [] #stores all data from resampling
    for sg in ls:
        files = list(Group.get(sg).keys())
        print(files)
        for file in files:
            data = Group.get(sg).get(file)
            data = np.array(data)
            df = pd.DataFrame(data[:, 0], columns=['value'])
            df.index = pd.to_datetime(data[:, 0], unit='s')
            windowed = df['value'].resample('5s').mean()
            all_data.append(windowed)
    combined = pd.concat(all_data).dropna()

    train_df, test_df = train_test_split(combined, test_size=0.1, shuffle=True, random_state=42)

    G31 = hdf.create_group('/Segmented Data/Train')
    G31.create_dataset('Train', data = train_df)
    G32 = hdf.create_group('/Segmented Data/Test')
    G32.create_dataset('Test', data = test_df)