import argparse
import joblib
import pandas as pd
import random

from preprocess import load_data, prepare_features_labels, split_data


def verify_identity(model_path, csv_path, row_index=None):
    # Load the trained model
    model = joblib.load(model_path)

    # Load and prepare data the same way we did for training
    df = load_data(csv_path)
    X, y = prepare_features_labels(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Reset index so we can pick rows cleanly by position
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # Pick a row: either the one specified, or a random one
    if row_index is None:
        row_index = random.randint(0, len(X_test) - 1)

    sample = X_test.iloc[[row_index]]
    actual_user = y_test.iloc[row_index]

    # Predict
    predicted_user = model.predict(sample)[0]
    probabilities = model.predict_proba(sample)[0]
    confidence = max(probabilities) * 100

    # Print results
    print(f"Sample row index: {row_index}")
    print(f"Actual user:      {actual_user}")
    print(f"Predicted user:   {predicted_user}")
    print(f"Confidence:       {confidence:.2f}%")

    if predicted_user == actual_user:
        print("Result: MATCH - identity verified")
    else:
        print("Result: MISMATCH - identity rejected")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify identity from a keystroke sample.")
    parser.add_argument(
        "--row", type=int, default=None,
        help="Row index from test set to verify (random if not specified)"
    )
    args = parser.parse_args()

    verify_identity(
        model_path="../models/keystroke_model.pkl",
        csv_path="../data/DSL-StrongPasswordData.csv",
        row_index=args.row
    )