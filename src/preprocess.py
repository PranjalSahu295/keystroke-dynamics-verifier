import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(csv_path):
    """Load the keystroke dataset from a CSV file."""
    df = pd.read_csv(csv_path)
    return df


def prepare_features_labels(df):
    """Split dataframe into features (X) and labels (y)."""
    X = df.drop(columns=['subject', 'sessionIndex', 'rep'])
    y = df['subject']
    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    """Split into train/test sets, keeping user proportions balanced."""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )