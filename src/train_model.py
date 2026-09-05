from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

from preprocess import load_data, prepare_features_labels, split_data


def train_and_evaluate(csv_path, model_output_path):
    # Load and prepare the data
    df = load_data(csv_path)
    X, y = prepare_features_labels(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))

    # Save the trained model
    joblib.dump(model, model_output_path)
    print(f"Model saved to {model_output_path}")

    return model


if __name__ == "__main__":
    train_and_evaluate(
        csv_path="../data/DSL-StrongPasswordData.csv",
        model_output_path="../models/keystroke_model.pkl"
    )