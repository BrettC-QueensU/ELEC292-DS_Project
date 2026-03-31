#import statements for pandas and hdf5 libraries
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import h5py
#Create a new hdf5 file with the name hdf5_data.h5 in write mode
with h5py.File('./hdf5_data.h5', 'w') as hdf:
    #Create first group and subgroup with Brett's data
    G11= hdf.create_group('/Raw Data/Brett')
    #create datasets with all of Brett's raw data under subgroup Brett
    G11.create_dataset('walking1', data=pd.read_csv('Raw_Data/Brett_RawData/Walk-1_RawData.csv'))
    G11.create_dataset('walking2', data=pd.read_csv('Raw_Data/Brett_RawData/Walk-2_RawData.csv'))
    G11.create_dataset('walking3', data=pd.read_csv('Raw_Data/Brett_RawData/Walk-3_RawData.csv'))
    G11.create_dataset('walking4', data=pd.read_csv('Raw_Data/Brett_RawData/Walk-4_RawData.csv'))
    G11.create_dataset('walking5', data=pd.read_csv('Raw_Data/Brett_RawData/Walk-5_RawData.csv'))
    G11.create_dataset('jumping1', data=pd.read_csv("Raw_Data/Brett_RawData/Jump-1_RawData.csv"))
    G11.create_dataset('jumping2', data=pd.read_csv("Raw_Data/Brett_RawData/Jump-2_RawData.csv"))
    G11.create_dataset('jumping3', data=pd.read_csv("Raw_Data/Brett_RawData/Jump-3_RawData.csv"))
    G11.create_dataset('jumping4', data=pd.read_csv("Raw_Data/Brett_RawData/Jump-4_RawData.csv"))
    G11.create_dataset('jumping5', data=pd.read_csv("Raw_Data/Brett_RawData/Jump-5_RawData.csv"))
    # Create second subgroup with Logan's data
    G12 = hdf.create_group('/Raw Data/Logan')
    # create datasets with all of Logan's raw data under subgroup Logan
    G12.create_dataset('walking1', data=pd.read_csv("Raw_Data/Logan_data/Walking_backpocket/Raw Data.csv"))
    G12.create_dataset('walking2', data=pd.read_csv("Raw_Data/Logan_data/Walking_frontpocket/Raw Data.csv"))
    G12.create_dataset('walking3', data=pd.read_csv("Raw_Data/Logan_data/Walking_inhand-faceup/Raw Data.csv"))
    G12.create_dataset('walking4', data=pd.read_csv("Raw_Data/Logan_data/Walking_inhand-side/Raw Data.csv"))
    G12.create_dataset('walking5', data=pd.read_csv("Raw_Data/Logan_data/Walking_sweaterpocket/Raw Data.csv"))
    G12.create_dataset('jumping1', data=pd.read_csv("Raw_Data/Logan_data/Jumping_backpocket/Raw Data.csv"))
    G12.create_dataset('jumping2', data=pd.read_csv("Raw_Data/Logan_data/Jumping_frontpocket/Raw Data.csv"))
    G12.create_dataset('jumping3', data=pd.read_csv("Raw_Data/Logan_data/Jumping_inhand-faceup/Raw Data.csv"))
    G12.create_dataset('jumping4', data=pd.read_csv("Raw_Data/Logan_data/Jumping_inhand-side/Raw Data.csv"))
    G12.create_dataset('jumping5', data=pd.read_csv("Raw_Data/Logan_data/Jumping_sweaterpocket/Raw Data.csv"))
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
    G21.create_dataset('walk1', data=pd.read_csv("Pre-Processed_Data/Brett/Walk-1_PreProcessed.csv"))
    G21.create_dataset('walk2', data=pd.read_csv("Pre-Processed_Data/Brett/Walk-2_PreProcessed.csv"))
    G21.create_dataset('walk3', data=pd.read_csv("Pre-Processed_Data/Brett/Walk-3_PreProcessed.csv"))
    G21.create_dataset('walk4', data=pd.read_csv("Pre-Processed_Data/Brett/Walk-4_PreProcessed.csv"))
    G21.create_dataset('walk5', data=pd.read_csv("Pre-Processed_Data/Brett/Walk-5_PreProcessed.csv"))
    G21.create_dataset('jump1', data=pd.read_csv("Pre-Processed_Data/Brett/Jump-1_PreProcessed.csv"))
    G21.create_dataset('jump2', data=pd.read_csv("Pre-Processed_Data/Brett/Jump-2_PreProcessed.csv"))
    G21.create_dataset('jump3', data=pd.read_csv("Pre-Processed_Data/Brett/Jump-3_PreProcessed.csv"))
    G21.create_dataset('jump4', data=pd.read_csv("Pre-Processed_Data/Brett/Jump-4_PreProcessed.csv"))
    G21.create_dataset('jump5', data=pd.read_csv("Pre-Processed_Data/Brett/Jump-5_PreProcessed.csv"))

    G22 = hdf.create_group('/Pre-Processed Data/Logan')
    G22.create_dataset('walk1', data=pd.read_csv("Pre-Processed_Data/Logan/Walking_backpocket_PreProcessed.csv"))
    G22.create_dataset('walk2', data=pd.read_csv("Pre-Processed_Data/Logan/Walking_frontpocket_PreProcessed.csv"))
    G22.create_dataset('walk3', data=pd.read_csv("Pre-Processed_Data/Logan/Walking_inhand-faceup_PreProcessed.csv"))
    G22.create_dataset('walk4', data=pd.read_csv("Pre-Processed_Data/Logan/Walking_inhand-side_PreProcessed.csv"))
    G22.create_dataset('walk5', data=pd.read_csv("Pre-Processed_Data/Logan/Walking_sweaterpocket_PreProcessed.csv"))
    G22.create_dataset('jump1', data=pd.read_csv("Pre-Processed_Data/Logan/Jumping_backpocket_PreProcessed.csv"))
    G22.create_dataset('jump2', data=pd.read_csv("Pre-Processed_Data/Logan/Jumping_frontpocket_PreProcessed.csv"))
    G22.create_dataset('jump3', data=pd.read_csv("Pre-Processed_Data/Logan/Jumping_inhand-faceup_PreProcessed.csv"))
    G22.create_dataset('jump4', data=pd.read_csv("Pre-Processed_Data/Logan/Jumping_inhand-side_PreProcessed.csv"))
    G22.create_dataset('jump5', data=pd.read_csv("Pre-Processed_Data/Logan/Jumping_sweaterpocket_PreProcessed.csv"))

    G23 = hdf.create_group('/Pre-Processed Data/Vince')
    G23.create_dataset('walk1', data=pd.read_csv("Pre-Processed_Data/Vince/Walk1_Hand_PreProcessed.csv"))
    G23.create_dataset('walk2', data=pd.read_csv("Pre-Processed_Data/Vince/Walk2_FrontPoc_PreProcessed.csv"))
    G23.create_dataset('walk3', data=pd.read_csv("Pre-Processed_Data/Vince/Walk3_BackPoc_PreProcessed.csv"))
    G23.create_dataset('walk4', data=pd.read_csv("Pre-Processed_Data/Vince/Walk4_Jacket_PreProcessed.csv"))
    G23.create_dataset('walk5', data=pd.read_csv("Pre-Processed_Data/Vince/Walk5_Hand_PreProcessed.csv"))
    G23.create_dataset('jump1', data=pd.read_csv("Pre-Processed_Data/Vince/Jump1_Hand_PreProcessed.csv"))
    G23.create_dataset('jump2', data=pd.read_csv("Pre-Processed_Data/Vince/Jump2_FrontPoc_PreProcessed.csv"))
    G23.create_dataset('jump3', data=pd.read_csv("Pre-Processed_Data/Vince/Jump3_BackPocket_PreProcessed.csv"))
    G23.create_dataset('jump4', data=pd.read_csv("Pre-Processed_Data/Vince/Jump4_Holding_PreProcessed.csv"))
    G23.create_dataset('jump5', data=pd.read_csv("Pre-Processed_Data/Vince/Jump5_Hand_PreProcessed.csv"))
    #lastly, create group segmented data with subgroups train and test
    #iterate through the pre processed data and resample each file into a
    #dataframe with 5 second intervals
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