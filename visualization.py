import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

files = {
    'Brett': 'Raw_Data/Brett_RawData/Jumping/Jump-4_RawData.csv',
    'Logan': 'Raw_Data/Logan_data/Jumping/Jumping_inhand-side/Raw Data.csv',
    'Vince': 'Raw_Data/Vince_Data/Jump1_Hand/Raw Data.csv',
}

axes_cols = {
    'X': 'Linear Acceleration x (m/s^2)',
    'Y': 'Linear Acceleration y (m/s^2)',
    'Z': 'Linear Acceleration z (m/s^2)',
}

colours = {'X': 'tab:blue', 'Y': 'tab:orange', 'Z': 'tab:green'}

fig, subplots = plt.subplots(3,1, figsize=(12, 9))

for ax, (label, path) in zip(subplots, files.items()):
    df = pd.read_csv(path)
    time = df['Time (s)']

    for axis_name, col in axes_cols.items():
        ax.plot(time, df[col], label=f'{axis_name}', color=colours[axis_name], linewidth=1.2)

    ax.set_title(label, fontsize=13, fontweight='bold')
    ax.set_ylabel('Linear Acceleration (m/s²)', fontsize=10)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='-')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(title='Axis', fontsize=9, loc='upper right')

subplots[-1].set_xlabel('Time (s)', fontsize=11)

fig.suptitle('Linear Acceleration vs Time (X, Y, Z)', fontsize=15, fontweight='bold', y=1.01)

plt.tight_layout()
plt.show()