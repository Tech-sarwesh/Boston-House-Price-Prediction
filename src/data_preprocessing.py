import pandas as pd

from sklearn.model_selection import train_test_split


def load_data(file_path):
    """
    Load the Boston house price dataset.
    """
    df = pd.read_csv(file_path)
    return df


def split_features_target(df):
    """
    Separate input features (X) and target (y).
    """
    X = df.drop("MEDV", axis=1)
    y = df["MEDV"]

    return X, y


def split_train_test(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test