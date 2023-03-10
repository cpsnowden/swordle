import numpy as np
import pandas as pd


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
        return df_axis.subtract(mean, axis='index').divide(std, axis='index')

    x_norm = normalize_axis(x)
    y_norm = normalize_axis(y)
    z_norm = normalize_axis(z)
    return pd.concat([x_norm, y_norm, z_norm, other],
                     axis='columns')[df.columns]
