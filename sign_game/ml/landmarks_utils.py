import pandas as pd
import numpy as np


def normalize_handmarks_3dnp(hand_landmarks_np):
    """
    Normalizeds a handmarks numpy array per row by:
      1. Calculating the mean
            - X position mu_x
            - Y position mu_y
            - Z position mu_z
      2. Calculating the standard deviation of the:
            - X position std_x
            - Y position std_y
            - Z position std_z
      3. Normalizing as (x_i - mu_x) / std_x etc
    """
    image_mean = hand_landmarks_np.mean(axis=1)[:, np.newaxis, :]
    image_std = hand_landmarks_np.std(axis=1)[:, np.newaxis, :]
    return (hand_landmarks_np - image_mean) / image_std


def normalize_handmarks_per_image(df):
    """
    Normalize handmarks per image in order to centralize them against the
    mean x,y and z positions (hand centroid)
    """
    x = df.filter(regex="_(X|x)$")
    y = df.filter(regex="_(Y|y)$")
    z = df.filter(regex="_(Z|z)$")
    other = df.drop(columns=list(x.columns) + list(y.columns) +
                    list(z.columns))

    def normalize_axis(df_axis):
        mean = df_axis.mean(axis=1)
        std = df_axis.std(axis=1)
        mean_diff = df_axis.subtract(mean, axis='index')
        return mean_diff.divide(std, axis='index')

    x_norm = normalize_axis(x)
    y_norm = normalize_axis(y)
    z_norm = normalize_axis(z)
    return pd.concat([x_norm, y_norm, z_norm, other],
                     axis='columns')[df.columns]


if __name__ == "__main__":
    df = pd.DataFrame([{
        "L1_X": 1,
        "L1_Y": 1,
        "L1_Z": 1,
        "L2_X": 2,
        "L2_Y": 2,
        "L2_Z": 2,
        "L3_X": 3,
        "L3_Y": 3,
        "L3_Z": 3,
        "OTHER": 1
    }, {
        "L1_X": 1,
        "L1_Y": 1,
        "L1_Z": 1,
        "L2_X": 4,
        "L2_Y": 4,
        "L2_Z": 4,
        "L3_X": 100,
        "L3_Y": 100,
        "L3_Z": 100,
        "OTHER": 1
    }])
    print(df)
    print(normalize_handmarks_per_image(df))
