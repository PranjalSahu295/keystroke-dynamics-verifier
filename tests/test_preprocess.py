import sys
import os

# Add src folder to the path so we can import from it
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocess import load_data, prepare_features_labels, split_data

CSV_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'DSL-StrongPasswordData.csv')


def test_load_data_returns_correct_shape():
    df = load_data(CSV_PATH)
    assert df.shape[0] == 20400
    assert df.shape[1] == 34


def test_prepare_features_labels_splits_correctly():
    df = load_data(CSV_PATH)
    X, y = prepare_features_labels(df)
    assert X.shape[1] == 31
    assert len(y) == len(X)
    assert 'subject' not in X.columns


def test_split_data_produces_expected_sizes():
    df = load_data(CSV_PATH)
    X, y = prepare_features_labels(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    assert len(X_train) == 16320
    assert len(X_test) == 4080
    assert len(X_train) + len(X_test) == len(X)