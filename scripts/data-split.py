import pandas as pd
import numpy as np

def csv_train_test_split(path, fraction):
    """
    Inputs:
    path (str) - path to a csv we want to split
    fraction (float) - fraction of the csv we want in testing set i.e. fraction 0.2

    """

    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(path)

    # Shuffle the rows of the DataFrame randomly
    df = df.sample(frac=1)

    # Calculate the number of rows for each file
    n_total = len(df)
    n_train = int(n_total * 0.8)
    n_test = n_total - n_train

    # Split the DataFrame into two separate DataFrames
    df_train = df[:n_train]
    df_test = df[n_train:]

    # Write the two DataFrames to separate CSV files
    df_test.to_csv('test.csv', index=False)
    print('test CSV completed!')
    df_train.to_csv('train.csv', index=False)
    print('train CSV completed!')

if __name__ == '__main__':
    csv_path = 'images_ds.csv'
    csv_train_test_split(csv_path, 0.2)
