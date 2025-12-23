"""xgboost_comprehensive: A Flower / XGBoost app."""

import xgboost as xgb
import pandas as pd

fds = None  # Cache FederatedDataset

def load_data(
    partition_id,
    test_fraction,
):
    """Load partition data."""
    data = pd.read_csv("clients_data/client_" + str(partition_id) + ".csv", sep=';')

    data = data.dropna()

    # Fix: Drop only existing columns and handle Unnamed: 0
    cols_to_drop = ['Unnamed: 0', 'date', 'label']
    existing_cols = [col for col in cols_to_drop if col in data.columns]
    X = data.drop(columns=existing_cols)

    # Use label as target (it exists in your CSV)
    y = data["label"]

    # Rest of your code unchanged...
    X_train = X.iloc[:int((1-test_fraction) * len(X))]
    y_train = y.iloc[:int((1-test_fraction) * len(y))]
    X_test = X.iloc[int((1-test_fraction) * len(X)):]
    y_test = y.iloc[int((1-test_fraction) * len(y)):]

    num_train = len(X_train)
    num_test = len(X_test)

    train_dmatrix = xgb.DMatrix(X_train, label=y_train)
    valid_dmatrix = xgb.DMatrix(X_test, label=y_test)

    return train_dmatrix, valid_dmatrix, num_train, num_test



def replace_keys(input_dict, match="-", target="_"):
    """Recursively replace match string with target string in dictionary keys."""
    new_dict = {}
    for key, value in input_dict.items():
        new_key = key.replace(match, target)
        if isinstance(value, dict):
            new_dict[new_key] = replace_keys(value, match, target)
        else:
            new_dict[new_key] = value
    return new_dict

