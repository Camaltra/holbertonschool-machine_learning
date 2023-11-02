import pandas as pd


def from_file(filename, delimiter):
    return pd.read_csv(filename, delimiter=delimiter)