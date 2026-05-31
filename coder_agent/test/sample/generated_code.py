#!/usr/bin/env python3
"""
AutoML Baseline Predictor (Pandas + Scikit-Learn)

Purpose:
- Read training and test data
- Preprocess training data (drop samples with missing target; do not impute or drop on features)
- Train a simple baseline logistic regression model
- Generate predictions on the full test set, preserving the test row order and aligning feature columns
- Save the model and predictions to /home/gem/workspace/iteration_0/output
- Perform validation checks on the prediction output (indices, column names, row count)

Notes:
- Output files are saved under /home/gem/workspace/iteration_0/output
- The prediction file is named "results" with the same extension as the test file (e.g., results.csv)
- Output column must be exactly named as the training target: "Transported"
- Do not use n_jobs=-1 or multi-threading; run single-threaded
- If validation data is available, a 10% hold-out validation is attempted; any issues are reported but execution continues
- Includes explicit checkpoint prints to track progress

Installation (comments):
# Ensure required libraries are installed in the environment where this script runs:
#   pip install pandas scikit-learn joblib
# This script uses only pandas and scikit-learn (plus joblib for model persistence)

"""

import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

# -----------------------------------------------
# Helper utilities
# -----------------------------------------------

def remove_index_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove common index-like columns that might be exported unintentionally.
    """
    for col in ["index", "Unnamed: 0"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    return df


def ensure_output_dir(path: str) -> None:
    """
    Create the output directory if it does not exist.
    """
    os.makedirs(path, exist_ok=True)


def save_model(model, out_dir: str) -> str:
    """
    Save the trained model to a timestamped folder under the output directory.
    Returns the path to the saved model file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rnd = random.randint(1000, 9999)
    model_dir = os.path.join(out_dir, f"model_{timestamp}_{rnd}")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "logreg_model.pkl")
    joblib.dump(model, model_path)
    return model_path

# -----------------------------------------------
# Main execution logic
# -----------------------------------------------

def main():
    print("Starting preprocessing...")

    # Input/Output paths
    train_path = "/home/gem/workspace/data/train.csv"
    test_path = "/home/gem/workspace/data/test.csv"
    output_dir = "/home/gem/workspace/iteration_0/output"

    ensure_output_dir(output_dir)

    # Load data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Remove potential index columns
    train_df = remove_index_column(train_df)
    test_df = remove_index_column(test_df)

    # Target column
    target = "Transported"
    if target not in train_df.columns:
        raise ValueError(f"Training data does not contain the target column '{target}'")

    # Drop training samples with missing target
    initial_train_shape = train_df.shape
    train_df = train_df.dropna(subset=[target])
    post_drop_train_shape = train_df.shape
    if post_drop_train_shape[0] < initial_train_shape[0]:
        print(f"Note: Dropped {initial_train_shape[0] - post_drop_train_shape[0]} rows due to missing target values.")

    print("Separating features and target...")
    X = train_df.drop(columns=[target])
    y = train_df[target]

    # Encode target to numeric (logistic regression requires numeric)
    if y.dtype == "object" or y.dtype.name == "category":
        y_numeric = y.astype("category").cat.codes
        # Ensure binary encoding for transport status if possible
    else:
        y_numeric = y.values

    # One-hot encode categorical features (no imputation performed)
    X_enc = pd.get_dummies(X, drop_first=True)
    test_enc = pd.get_dummies(test_df, drop_first=True)

    # Align train/test feature columns
    X_final, test_final = X_enc.align(test_enc, join="left", axis=1, fill_value=0)

    if X_final.shape[1] == 0:
        raise ValueError("No feature columns available after encoding. Check input data.")

    print("Training model...")
    # Attempt 10% hold-out validation if feasible
    X_train, X_valid, y_train, y_valid = None, None, None, None
    try:
        # Use a simple stratification if possible
        stratify_param = None
        try:
            if len(np.unique(y_numeric)) >= 2:
                stratify_param = y_numeric
        except Exception:
            stratify_param = None

        X_train, X_valid, y_train, y_valid = train_test_split(
            X_final, y_numeric, test_size=0.1, random_state=42, stratify=stratify_param
        )
        print(f"Hold-out validation split created: {X_train.shape[0]} train, {X_valid.shape[0]} valid samples.")
    except Exception as split_err:
        print("Could not create a validation split. Proceeding with full training data.")
        X_train, y_train = X_final, y_numeric
        X_valid, y_valid = None, None

    # Initialize logistic regression (single-threaded by default)
    model = LogisticRegression(max_iter=1000, solver="lbfgs")

    model.fit(X_train, y_train)

    # Validation score (optional)
    if X_valid is not None and y_valid is not None:
        try:
            val_score = model.score(X_valid, y_valid)
            print(f"Validation accuracy: {val_score:.4f}")
        except Exception as val_err:
            print(f"Validation failed due to: {val_err}. Continuing without validation score.")

    # Save the trained model
    model_path = save_model(model, output_dir)
    print(f"Model saved to: {model_path}")

    print("Generating predictions...")
    test_preds = model.predict(test_final)

    # Prepare submission DataFrame
    submission = pd.DataFrame({"Transported": test_preds}, index=test_df.index)

    # Determine extension from test file
    _, ext = os.path.splitext(test_path)
    results_path = os.path.join(output_dir, f"results{ext}")

    # Save results
    # Save without index column to align with typical submission formats
    submission.to_csv(results_path, index=False)
    print(f"Results saved to: {results_path}")

    print("Validation checks...")

    try:
        # Check 1: number of predictions matches number of test samples
        assert submission.shape[0] == test_df.shape[0], "Mismatch in number of predictions and test samples."

        # Check 2: correct output column name
        assert list(submission.columns) == ["Transported"], "Output column name must be exactly 'Transported'."

        # Check 3: indices preserved (in-memory check against test_df)
        assert submission.index.equals(test_df.index), "Prediction indices do not match original test data indices."

        print("Validation checks passed!")
    except AssertionError as e:
        print(f"Validation check failed: {e}")
        # As per requirements, it's acceptable for validation to fail; we do not re-raise here.

if __name__ == "__main__":
    main()