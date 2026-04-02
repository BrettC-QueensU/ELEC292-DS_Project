import pandas as pd
import numpy as np

# UNSURE IF FEATURES MUST BE UNIQUE. e.x.:
# whether x-acc mean and y-acc mean counts as 1 feature (mean) or two features, (mean of x acc, mean of y acc)

# Extracts 10+ features from an inputted data-frame
# Treats the inputted df as a single window to extract data from
# features extracted: max, min, range, mean, median, variance, skewness, kurtosis, std, mode
def extract_features(input_df):
    acc_columns = [
        'Linear Acceleration x (m/s^2)',
        'Linear Acceleration y (m/s^2)',
        'Linear Acceleration z (m/s^2)',
        'Absolute acceleration (m/s^2)'
    ]

    x_labels = ['x: max', 'x: min', 'x: range', 'x: mean', 'x: median',
              'x: var', 'x: skew', 'x: kurt', 'x: std', 'x: mode']

    y_labels = [ 'y: max', 'y: min', 'y: range', 'y: mean', 'y: median',
              'y: var', 'y: skew', 'y: kurt', 'y: std', 'y: mode']

    z_labels = ['z: max', 'z: min', 'z: range', 'z: mean', 'z: median',
              'z: var', 'z: skew', 'z: kurt', 'z: std', 'z: mode']

    abs_labels = ['abs: max', 'abs: min', 'abs: range', 'abs mean', 'abs: median',
              'abs: var', 'abs: skew', 'abs: kurt', 'abs: std', 'abs: mode']

    labels = x_labels + y_labels + z_labels + abs_labels

    # input_df['column_name'].mean() is an example of how to get each feature for this part
    features = []

    for col in acc_columns:
        s = input_df.loc[col]

        features.append(s.max())
        features.append(s.min())
        features.append(s.max() - s.min())
        features.append(s.mean())
        features.append(s.median())
        features.append(s.var())
        features.append(s.skew())
        features.append(s.kurt())
        features.append(s.std())
        features.append(s.mode()[0])

    feature_df = pd.DataFrame([features], columns=labels)

    # Normalization is done in the logisticRegression.py file

    return feature_df

# helper function to split one pre-processed dataframe into approximately 5-second windows,
# If there are at least 4 seconds of data remaining after the final full 5-second window,
# Then saves one last window.
def segment_dataframe(df):
    times = df.iloc[:, 0].values
    sample_period = float(np.median(np.diff(times)))
    samples_per_window = int(round(5 / sample_period))
    min_samples = int(round(4 / sample_period))  # minimum samples for a 4-second window

    n_complete = len(df) // samples_per_window  # number of full windows
    windows = []
    for i in range(n_complete):
        start = i * samples_per_window
        end = start + samples_per_window
        windows.append(df.iloc[start:end].reset_index(drop=True))

    # Check if the remaining samples form at least a 4-second window
    remainder_start = n_complete * samples_per_window
    remainder = df.iloc[remainder_start:]
    if len(remainder) >= min_samples:
        windows.append(remainder.reset_index(drop=True))

    return windows
