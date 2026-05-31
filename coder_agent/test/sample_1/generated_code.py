"""
AutoML baseline script for the "random acts of pizza" Reddit dataset.

Purpose:
- Load train.json and test.json from the expected location.
- Preprocess data according to requirements (drop rows with NA labels only; do not impute or drop by missing features).
- Train a simple but robust baseline classifier (HistGradientBoostingClassifier) using pandas + scikit-learn.
- Save the trained model to a timestamped folder under the designated output directory.
- Generate predictions (probabilities) for the entire test set, preserving original test order and indices.
- Save predictions to /home/gem/workspace/iteration_0/output/results.csv with exact expected column names.
- Perform validation checks to ensure indices and columns align with the test data and submission format.
- Print explicit checkpoint messages throughout the process.

Notes:
- All file outputs are written only under /home/gem/workspace/iteration_0/output.
- No parallel/multithreaded execution (no n_jobs usage beyond defaults).
- Wrapped under if __name__ == "__main__": to avoid DDP-related issues.
- Validation step is attempted and will be skipped gracefully if it fails.

Installation hints (to be run before executing this script):
- pip install pandas scikit-learn numpy joblib
- Ensure the target paths exist and are writable.
"""

import os
import sys
from datetime import datetime

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier
from joblib import dump


# Paths (adjusted to the task description)
TRAIN_PATH = "/tmp/mlebench-lite/random-acts-of-pizza/prepared/train.json"
TEST_PATH = "/tmp/mlebench-lite/random-acts-of-pizza/prepared/test.json"
OUTPUT_DIR = "/home/gem/workspace/iteration_0/output"

# Important: The label column is specified as 'requester_received_pizza'
LABEL_COL = "requester_received_pizza"
# Optional: an id column that should not be treated as a feature
ID_COL = "request_id"

def ensure_dir(path: str):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

def main():
    print("Starting preprocessing...")
    ensure_dir(OUTPUT_DIR)

    # 1. Load data
    print(f"Loading train data from: {TRAIN_PATH}")
    train_df = pd.read_json(TRAIN_PATH, orient="records")
    print(f"Training samples: {train_df.shape[0]}, features: {train_df.shape[1]}")

    print(f"Loading test data from: {TEST_PATH}")
    test_df = pd.read_json(TEST_PATH, orient="records")
    print(f"Test samples: {test_df.shape[0]}, features: {test_df.shape[1]}")

    # 2. Basic cleanup (drop unnecessary index column if present)
    for df_name, df in [("train", train_df), ("test", test_df)]:
        if "index" in df.columns:
            df.drop(columns=["index"], inplace=True)
            if df_name == "train":
                train_df = df
            else:
                test_df = df

    # 3. Separate label and features for training
    if LABEL_COL not in train_df.columns:
        raise ValueError(f"Label column '{LABEL_COL}' not found in training data.")

    y = train_df[LABEL_COL]

    # Drop label from features
    X = train_df.drop(columns=[LABEL_COL])

    # Remove the id column from features if present (we'll preserve order via test_ids)
    if ID_COL in X.columns:
        X.drop(columns=[ID_COL], inplace=True)

    # Also drop the id column from test features to avoid leakage
    test_features = test_df.copy()
    test_ids = None
    if ID_COL in test_features.columns:
        test_ids = test_features[ID_COL].copy()
        test_features.drop(columns=[ID_COL], inplace=True)
    else:
        test_ids = test_features.index.to_series(index=test_features.index)

    # Ensure the same feature columns alignment via combined encoding
    combined = pd.concat([X, test_features], ignore_index=True)
    # One-hot encode categorical features; keep NaNs as a distinct category using dummy_na
    encoded = pd.get_dummies(combined, drop_first=False, dummy_na=True)

    # Split back into train/test encoded features
    n_train = X.shape[0]
    X_encoded = encoded.iloc[:n_train]
    X_test_encoded = encoded.iloc[n_train:]

    # 4. Hold-out validation (10% of training data) if possible
    print("Starting train/validation split for evaluation...")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_encoded, y, test_size=0.1, random_state=42, stratify=y
    )

    # 5. Train a baseline model
    print("Training model (HistGradientBoostingClassifier)...")
    model = HistGradientBoostingClassifier(random_state=42)

    model.fit(X_tr, y_tr)

    # 6. Validation (AUC-ROC)
    val_proba = model.predict_proba(X_val)[:, 1]
    try:
        val_auc = roc_auc_score(y_val, val_proba)
        print(f"Validation AUC-ROC: {val_auc:.6f}")
    except Exception as e:
        print("Warning: Validation failed or could not compute AUC. Continuing without strict validation.")
        val_auc = None

    # 7. Train on full training data
    print("Training final model on full training data...")
    model.fit(X_encoded, y)

    # 8. Predict on the entire test set
    print("Generating predictions for the test set...")
    test_proba = model.predict_proba(X_test_encoded)[:, 1]

    # 9. Prepare predictions DataFrame
    print("Preparing submission file (results.csv)...")
    # Ensure test_ids order corresponds to test_proba order
    results_df = pd.DataFrame({
        "request_id": test_ids.astype(str) if isinstance(test_ids, pd.Series) else test_ids,
        "requester_received_pizza": test_proba
    })

    # If test_ids was a Series with same length, ensure alignment
    if results_df.shape[0] != test_df.shape[0]:
        raise ValueError("Mismatch between number of predictions and test samples.")

    output_path = os.path.join(OUTPUT_DIR, "results.csv")
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")

    # 10. Save the model (timestamped folder)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(OUTPUT_DIR, f"model_{timestamp}")
    ensure_dir(model_dir)
    model_path = os.path.join(model_dir, "model.pkl")
    dump(model, model_path)
    print(f"Model saved to: {model_path}")

    # 11. Validation checks on the generated results
    print("Running validation checks on the generated results...")
    try:
        saved_results = pd.read_csv(output_path)
        # Check number of rows
        if saved_results.shape[0] != test_df.shape[0]:
            raise ValueError("Validation check failed: number of predictions does not match number of test samples.")

        # Check exact columns
        expected_cols = ["request_id", "requester_received_pizza"]
        if list(saved_results.columns) != expected_cols:
            raise ValueError(f"Validation check failed: expected columns {expected_cols}, got {list(saved_results.columns)}")

        # Check indices/IDs alignment
        test_ids_list = test_df[ID_COL].astype(str).tolist() if ID_COL in test_df.columns else test_df.index.astype(str).tolist()
        if not saved_results["request_id"].astype(str).tolist() == test_ids_list:
            raise ValueError("Validation check failed: prediction IDs do not match test IDs in the same order.")

        print("Validation checks passed!")
    except Exception as ve:
        print("Validation checks encountered an issue but continuing as per guidelines.")
        print(f"Validation note: {ve}")

    print("Workflow completed.")

if __name__ == "__main__":
    main()