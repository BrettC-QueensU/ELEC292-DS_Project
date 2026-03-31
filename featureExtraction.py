import numpy as np
import pandas as pd

# UNSURE IF FEATURES MUST BE UNIQUE. e.x.:
# whether x-acc mean and y-acc mean counts as 1 feature (mean) or two features, (mean of x acc, mean of y acc)

# Extracts 10+ features from an inputted data-frame and normalizes data.
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








# Testing function with a full-length csv file (rather than a trimmed one).


# calls extract_features() on each df in the list, and returns them as a combined df
#def extract_features_list(input_list):


# BELOW IS AI GENERATED CODE, NEEDS TO BE EDITED AND INTEGREATED INTO CODE FOR PROJECT

import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler




#def extract_features_and_normalize(input_df):
#    """
#    Extracts 10+ features from an inputted dataframe and normalizes them
#    before returning.
#    Treats the inputted df as a single window to extract features from.
#    Features extracted: max, min, range, mean, median, variance, skewness,
#                        kurtosis, std, mode
#    Returns a single-row normalized dataframe.
#    """
#    acc_columns = [
#        'Linear Acceleration x (m/s^2)',
#        'Linear Acceleration y (m/s^2)',
#        'Linear Acceleration z (m/s^2)',
#        'Absolute acceleration (m/s^2)'
#    ]
#
#    # short axis labels for column naming
#    axis_labels = ['x', 'y', 'z', 'abs']
#
#    features = {}
#
#    for col, label in zip(acc_columns, axis_labels):
#        s = input_df[col].values
#
#        features[f'max_{label}'] = np.max(s)
#        features[f'min_{label}'] = np.min(s)
#        features[f'range_{label}'] = np.max(s) - np.min(s)
#        features[f'mean_{label}'] = np.mean(s)
#        features[f'median_{label}'] = np.median(s)
#        features[f'variance_{label}'] = np.var(s)
#        features[f'skewness_{label}'] = skew(s)
#        features[f'kurtosis_{label}'] = kurtosis(s)
#        features[f'std_{label}'] = np.std(s)
#        features[f'mode_{label}'] = pd.Series(s).mode()[0]
#
#    feature_df = pd.DataFrame([features])
#
#    # normalize features using z-score standardization
#    scaler = StandardScaler()
#    feature_df_normalized = pd.DataFrame(
#        scaler.fit_transform(feature_df.T).T,
#        columns=feature_df.columns
#    )
#
#    return feature_df_normalized





