# Problem Statement

Traditional identity verification systems (passwords, PINs) can be shared, stolen, or guessed, and they don't verify that the *person typing* is who they claim to be — only that they know the right string. Keystroke dynamics offers a behavioral biometric: the unique rhythm of how a person types (how long keys are held, and the timing between consecutive keystrokes) can act as an additional identity signal, since these patterns are difficult to consciously replicate. This project builds a system that learns each user's typing rhythm and verifies identity based on it, using a public benchmark dataset of real typing samples from 51 users.

# Objectives

- Build a machine learning model that can identify a user based on their typing rhythm with high accuracy.
- Provide a measurable, evaluated system (not just a proof of concept) using standard classification metrics.
- Package the system as a usable, reproducible, command-line tool that doesn't depend on any particular environment (like Colab).

# Scope

- The system uses an existing benchmark dataset (CMU Keystroke Dynamics dataset), not live keystroke capture, due to time constraints.
- It demonstrates identification (which of 51 known users typed a sample) rather than open-set verification (detecting a completely unknown/unregistered user).

# Target Users

- Organizations or systems looking to add a lightweight behavioral-biometric layer to existing authentication (e.g., alongside passwords) for extra security.
- Researchers or students exploring behavioral biometrics as a category of AI/ML application.

# Functional Requirements

1. **Data Preprocessing Module** (`preprocess.py`)
   - Loads the keystroke dataset from CSV
   - Separates input features (timing measurements) from labels (user identity)
   - Splits data into training and testing sets with balanced user representation

2. **Model Training & Evaluation Module** (`train_model.py`)
   - Trains a Random Forest classifier on keystroke timing features
   - Evaluates performance using accuracy, precision, recall, and F1-score
   - Saves the trained model for reuse without retraining

3. **Identity Verification Module** (`verify.py`)
   - Accepts a typing sample (by row index, or randomly selected)
   - Predicts the most likely user identity with a confidence score
   - Reports match/mismatch against the claimed identity via command line

**Input/Output Structure**
- Input: 31 timing-based features per typing sample (hold time, down-down time, up-down time for each key transition)
- Output: predicted user ID + confidence percentage + match/mismatch verdict

**Workflow**
Raw CSV → preprocessing → train/test split → model training → saved model → CLI verification tool loads model → predicts on new sample → outputs result

# Non-Functional Requirements

1. **Performance** — model training completes in under a minute on a standard machine; verification is near-instant since it loads a pre-trained model rather than retraining.
2. **Reliability** — consistent ~93.5% accuracy across repeated runs (fixed random_state ensures reproducibility).
3. **Usability** — simple command-line interface; no GUI or complex setup required beyond installing dependencies via requirements.txt.
4. **Maintainability** — modular code (separate preprocessing, training, and verification scripts) makes it easy to swap in a different model or dataset later.
5. **Error Handling** — the system fails clearly with descriptive errors if dataset or model file paths are wrong, rather than failing silently.
6. **Resource Efficiency** — model file and training process are lightweight enough to run on a standard laptop without GPU requirements.