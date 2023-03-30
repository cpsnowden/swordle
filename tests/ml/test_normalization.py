import pandas as pd


def test_normalize_axis():
    from sign_game.ml.normalization import normalize_axis

    input = [
        {'x1': 1, 'x2': 2, 'x3': 3},
        {'x1': 10, 'x2': 20, 'x3': 30},
        {'x1': 10, 'x2': 20, 'x3': 40}
    ]

    input_df = pd.DataFrame.from_dict(input)

    normalized_df = normalize_axis(input_df)

    expected_output = [
        {'x1': -1.0, 'x2': 0.0, 'x3': 1.0},
        {'x1': -1.0, 'x2': 0.0, 'x3': 1.0},
        {
            'x1': -0.8728715609439694,
            'x2': -0.2182178902359923,
            'x3': 1.091089451179962
        }
    ]

    assert expected_output == normalized_df.to_dict(orient='records')


def test_frame_normalization():
    from sign_game.ml.normalization import create_frame_normalizer

    normalizer = create_frame_normalizer()

    input = [
        {'1_X': 1, '2_X': 2, '3_X': 3,
         '1_Y': 1, '2_Y': 2, '3_Y': 4,
         '1_Z': 1, '2_Z': 2, '3_Z': 5,
         'RANDOM_COL': 100
         },
    ]

    input_df = pd.DataFrame.from_dict(input)

    normalized_df = normalizer.fit_transform(input_df)

    expected_output = [
        {'1_X': -1.0, '2_X': 0.0, '3_X': 1.0,
         '1_Y': -0.8728715609439697, '2_Y': -0.2182178902359925, '3_Y': 1.0910894511799618,
         '1_Z': -0.8006407690254355, '2_Z': -0.32025630761017415, '3_Z': 1.12089707663561}
    ]

    assert expected_output == normalized_df.to_dict(orient='records')


def test_frame_normalization_passthrough():
    from sign_game.ml.normalization import create_frame_normalizer

    normalizer = create_frame_normalizer('passthrough')

    input = [
        {'1_X': 1, '2_X': 2, '3_X': 3,
         '1_Y': 1, '2_Y': 2, '3_Y': 4,
         '1_Z': 1, '2_Z': 2, '3_Z': 5,
         'RANDOM_COL': 100
         },
    ]

    input_df = pd.DataFrame.from_dict(input)

    normalized_df = normalizer.fit_transform(input_df)

    expected_output = [
        {'1_X': -1.0, '2_X': 0.0, '3_X': 1.0,
         '1_Y': -0.8728715609439697, '2_Y': -0.2182178902359925, '3_Y': 1.0910894511799618,
         '1_Z': -0.8006407690254355, '2_Z': -0.32025630761017415, '3_Z': 1.12089707663561,
         'RANDOM_COL': 100}
    ]

    assert expected_output == normalized_df.to_dict(orient='records')
