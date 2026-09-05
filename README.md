# Keystroke Dynamics Identity Verifier

## Overview
This project uses machine learning to identify a user based on their unique typing rhythm (keystroke dynamics) rather than just a password. It analyzes timing measurements — how long keys are held and the gaps between keystrokes — to predict which of 51 known users typed a given sample, along with a confidence score.

## Features
- Loads and preprocesses a real-world keystroke timing dataset (CMU Keystroke Dynamics Benchmark)
- Trains a Random Forest classifier to distinguish between 51 users based on typing rhythm
- Achieves ~93.5% accuracy on held-out test data
- Command-line tool to verify identity from a typing sample, with confidence score and match/mismatch result

## Technologies Used
- Python 3.x
- pandas, numpy (data handling)
- scikit-learn (machine learning)
- joblib (model saving/loading)
- pytest (testing)

## Dataset
CMU Keystroke Dynamics Benchmark Dataset (via Kaggle) — 51 users, 400 typing samples each, 31 timing features per sample. File: `data/DSL-StrongPasswordData.csv`

## Project Structure
keystroke-dynamics-verifier/
data/ Dataset CSV
notebooks/ Google Colab exploration notebook
src/ Python source code
models/ Trained model (generated, not tracked in git)
tests/ Unit tests
docs/ Design diagrams
requirements.txt Python dependencies
statement.md Problem statement and requirements


## Setup Instructions

1. Clone this repository:

git clone https://github.com/PranjalSahu295/keystroke-dynamics-verifier.git
cd keystroke-dynamics-verifier


2. Install dependencies:

pip install -r requirements.txt


## How to Run

1. Train the model (this generates `models/keystroke_model.pkl`, which is required before verification and is not included in the repo):

cd src
python train_model.py

This will print accuracy and a detailed classification report, and save the trained model.

2. Verify an identity using a random sample from the test set:

python verify.py


3. Verify a specific sample by row number:

python verify.py --row 10


## Testing
Run the test suite from the `src` folder:

pytest ../tests

Sample row index: 10
Actual user: s033
Predicted user: s033
Confidence: 78.00%
Result: MATCH - identity verified


## Future Enhancements
- Live keystroke capture instead of using a static dataset
- True open-set verification (rejecting unknown/unregistered users) instead of closed-set identification
- Web or GUI interface for easier demonstration

