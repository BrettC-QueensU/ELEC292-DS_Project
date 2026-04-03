import glob
import pandas as pd
import matplotlib.pyplot as plt

def preprocess_csv(df: pd.DataFrame):
    # import csv file as data frame, and interpolate to fill any missing values
    df = df.interpolate(method='linear')
    df_acc = df.drop(columns='Time (s)')

    # use a rolling mean on the data
    window_size = 21
    y_sma = df_acc.rolling(window_size).mean()

    # replacing the acceleration columns in the data frame with the ones in y_sma
    df['Linear Acceleration x (m/s^2)'] = y_sma['Linear Acceleration x (m/s^2)']
    df['Linear Acceleration y (m/s^2)'] = y_sma['Linear Acceleration y (m/s^2)']
    df['Linear Acceleration z (m/s^2)'] = y_sma['Linear Acceleration z (m/s^2)']
    df['Absolute acceleration (m/s^2)'] = y_sma['Absolute acceleration (m/s^2)']

    # delete rows with na at the start resulting from applying sma
    df_processed = df.dropna()

    # return the preprocessed data frame
    return df_processed



# Preprocessing Brett Data
files = glob.glob('Raw_Data/Brett_RawData/*.csv')
for filepath in files:
    filename = filepath.split('_RawData\\')[1].split("_RawData.csv")[0]
    df_processed_Brett = preprocess_csv(pd.read_csv(filepath))
    # generating a csv file with the preprocessed data
    df_processed_Brett.to_csv('Pre-Processed_Data/Brett/' + filename + '_PreProcessed.csv', index=False)

# Preprocessing Logan Data
files = glob.glob('Raw_Data/Logan_data/*/Raw Data.csv')
for filepath in files:
    filename = filepath.split('Logan_data\\')[1].split("\\Raw Data")[0]
    df_processed_Logan = preprocess_csv((pd.read_csv(filepath)))
    # generating a csv file with the preprocessed data
    df_processed_Logan.to_csv('Pre-Processed_Data/Logan/' + filename + '_PreProcessed.csv', index=False)

# Preprocessing Vince Data
files = glob.glob('Raw_Data/Vince_Data/*/Raw Data.csv')
for filepath in files:
    filename = filepath.split('Vince_Data\\')[1].split("\\Raw Data")[0]
    # generating a csv file with the preprocessed data
    df_processed_Vince = preprocess_csv((pd.read_csv(filepath)))
    df_processed_Vince.to_csv('Pre-Processed_Data/Vince/' + filename + '_PreProcessed.csv', index=False)


#Visualisation used to aid with trial and error of window size
#df = pd.read_csv('Pre-Processed_Data/Brett/Jump-1_PreProcessed.csv')

#time = df['Time (s)']
#ax = df['Linear Acceleration x (m/s^2)']
#ay = df['Linear Acceleration y (m/s^2)']
#az = df['Linear Acceleration z (m/s^2)']
#aa = df['Absolute acceleration (m/s^2)']

#fig, jump = plt.subplots(figsize=(12, 5))

#jump.plot(time, ax, label='X', color='tab:blue', linewidth=1.2)
#jump.plot(time, ay, label='Y', color='tab:orange', linewidth=1.2)
#jump.plot(time, az, label='Z', color='tab:green', linewidth=1.2)
#jump.plot(time, aa, label='Absolute', color='tab:purple', linewidth=1.2)
#jump.set_xlabel('Time (s)', fontsize=12)
#jump.set_ylabel('Linear Acceleration (m/s²)', fontsize=12)
#jump.set_title('Linear Acceleration vs Time (X, Y, Z, Absolute)', fontsize=14)
#jump.legend(title='Axis', fontsize=12)
#jump.grid(True, linestyle='--', alpha=0.5)
#jump.axhline(0, color='black', linewidth=0.8, linestyle='-')

#plt.tight_layout()
#plt.show()