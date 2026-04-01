import pandas as pd
from sklearn.preprocessing import StandardScaler

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
        features.append(s.mode())

    feature_df = pd.DataFrame([features], columns=labels)

    # normalize features using z-score standardization
    scaler = StandardScaler()
    feature_df_normalized = pd.DataFrame(
        scaler.fit_transform(feature_df.T).T,
        columns=labels
    )

    return feature_df_normalized

