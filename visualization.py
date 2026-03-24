import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

dataset = pd.read_csv('Raw_Data/Brett_RawData/Jumping/Jump-1_RawData.csv')
#dataset = pd.read_csv('Raw_Data/Logan_data/Jumping/Jumping_backpocket/Raw Data.csv')

time = dataset['Time (s)']
ax = dataset['Linear Acceleration x (m/s^2)']
ay = dataset['Linear Acceleration y (m/s^2)']
az = dataset['Linear Acceleration z (m/s^2)']

fig, jump = plt.subplots(figsize=(12, 5))

jump.plot(time, ax, label='X', color='tab:blue', linewidth=1.2)
jump.plot(time, ay, label='Y', color='tab:orange', linewidth=1.2)
jump.plot(time, az, label='Z', color='tab:green', linewidth=1.2)

jump.set_xlabel('Time (s)', fontsize=12)
jump.set_ylabel('Linear Acceleration (m/s²)', fontsize=12)
jump.set_title('Linear Acceleration vs Time (X, Y, Z)', fontsize=14)
jump.legend(title='Axis', fontsize=12)
jump.grid(True, linestyle='--', alpha=0.5)
jump.axhline(0, color='black', linewidth=0.8, linestyle='-')

plt.tight_layout()
plt.show()