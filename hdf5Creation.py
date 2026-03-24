#import statements for pandas and hdf5 libraries
import numpy as np
import pandas as pd
import h5py
#Create a new hdf5 file with the name hdf5_data.h5 in write mode
with h5py.File('./hdf5_data.h5', 'w') as hdf:
    #Create first group and subgroup with Brett's data
    G11= hdf.create_group('/Raw Data/Brett')
    #create datasets with all of Brett's raw data under subgroup Brett
    G11.create_dataset('walking1', data=pd.read_csv('Raw_Data/Brett_RawData/Walking/Walk-1_RawData.csv'))
    G11.create_dataset('walking2', data=pd.read_csv('Raw_Data/Brett_RawData/Walking/Walk-2_RawData.csv'))
    G11.create_dataset('walking3', data=pd.read_csv('Raw_Data/Brett_RawData/Walking/Walk-3_RawData.csv'))
    G11.create_dataset('walking4', data=pd.read_csv('Raw_Data/Brett_RawData/Walking/Walk-4_RawData.csv'))
    G11.create_dataset('walking5', data=pd.read_csv('Raw_Data/Brett_RawData/Walking/Walk-5_RawData.csv'))
    G11.create_dataset('jumping1', data=pd.read_csv("Raw_Data/Brett_RawData/Jumping/Jump-1_RawData.csv"))
    G11.create_dataset('jumping2', data=pd.read_csv("Raw_Data/Brett_RawData/Jumping/Jump-2_RawData.csv"))
    G11.create_dataset('jumping3', data=pd.read_csv("Raw_Data/Brett_RawData/Jumping/Jump-3_RawData.csv"))
    G11.create_dataset('jumping4', data=pd.read_csv("Raw_Data/Brett_RawData/Jumping/Jump-4_RawData.csv"))
    G11.create_dataset('jumping5', data=pd.read_csv("Raw_Data/Brett_RawData/Jumping/Jump-5_RawData.csv"))
    # Create second subgroup with Logan's data
    G12 = hdf.create_group('/Raw Data/Logan')
    # create datasets with all of Logan's raw data under subgroup Logan
    G12.create_dataset('walking1', data=pd.read_csv("Raw_Data/Logan_data/Walking/Walking_backpocket/Raw Data.csv"))
    G12.create_dataset('walking2', data=pd.read_csv("Raw_Data/Logan_data/Walking/Walking_frontpocket/Raw Data.csv"))
    G12.create_dataset('walking3', data=pd.read_csv("Raw_Data/Logan_data/Walking/Walking_inhand-faceup/Raw Data.csv"))
    G12.create_dataset('walking4', data=pd.read_csv("Raw_Data/Logan_data/Walking/Walking_inhand-side/Raw Data.csv"))
    G12.create_dataset('walking5', data=pd.read_csv("Raw_Data/Logan_data/Walking/Walking_sweaterpocket/Raw Data.csv"))
    G12.create_dataset('jumping1', data=pd.read_csv("Raw_Data/Logan_data/Jumping/Jumping_backpocket/Raw Data.csv"))
    G12.create_dataset('jumping2', data=pd.read_csv("Raw_Data/Logan_data/Jumping/Jumping_frontpocket/Raw Data.csv"))
    G12.create_dataset('jumping3', data=pd.read_csv("Raw_Data/Logan_data/Jumping/Jumping_inhand-faceup/Raw Data.csv"))
    G12.create_dataset('jumping4', data=pd.read_csv("Raw_Data/Logan_data/Jumping/Jumping_inhand-side/Raw Data.csv"))
    G12.create_dataset('jumping5', data=pd.read_csv("Raw_Data/Logan_data/Jumping/Jumping_sweaterpocket/Raw Data.csv"))
    # create third subgroup under Raw Data with Vince's data
    G13 = hdf.create_group('/Raw Data/Vince')
    # create datasets with all of Vince's raw data under subgroup Vince
    G13.create_dataset('walking1', data=pd.read_csv("Raw_Data/Vince_Data/Walk1_Hand/Raw Data.csv"))
    G13.create_dataset('walking2', data=pd.read_csv("Raw_Data/Vince_Data/Walk2_FrontPoc/Raw Data.csv"))
    G13.create_dataset('walking3', data=pd.read_csv("Raw_Data/Vince_Data/Walk3_BackPoc/Raw Data.csv"))
    G13.create_dataset('walking4', data=pd.read_csv("Raw_Data/Vince_Data/Walk4_Jacket/Raw Data.csv"))
    G13.create_dataset('walking5', data=pd.read_csv("Raw_Data/Vince_Data/Walk5_Hand/Raw Data.csv"))
    G13.create_dataset('jumping1', data=pd.read_csv("Raw_Data/Vince_Data/Jump1_Hand/Raw Data.csv"))
    G13.create_dataset('jumping2', data=pd.read_csv("Raw_Data/Vince_Data/Jump2_FrontPoc/Raw Data.csv"))
    G13.create_dataset('jumping3', data=pd.read_csv("Raw_Data/Vince_Data/Jump3_BackPocket/Raw Data.csv"))
    G13.create_dataset('jumping4', data=pd.read_csv("Raw_Data/Vince_Data/Jump4_Holding/Raw Data.csv"))
    G13.create_dataset('jumping5', data=pd.read_csv("Raw_Data/Vince_Data/Jump5_Hand/Raw Data.csv"))
    #Create group pre-processed data and 3 subgroups for each member's data
    G21 = hdf.create_group('/Pre-Processed Data/Brett')
    G22 = hdf.create_group('/Pre-Processed Data/Logan')
    G23 = hdf.create_group('/Pre-Processed Data/Vince')
    #lastly, create group segmented data with subgroups train and test
    G31 = hdf.create_group('/Segmented Data/Train')
    G32 = hdf.create_group('/Segmented Data/Test')