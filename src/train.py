import os
import yaml
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Import modules from src
from src.data_preprocessing import load_data, clean_data, prepare_datasets
from src.model import build_ann, build_xgboost, build_random_forest
from src.evaluate import (
    print_model_report,
    save_confusion_matrix,
    save_learning_evolution,
    save_roc_comparison
)

def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_training_pipeline(config_path="config/config.yaml"):
    """
    Run the end-to-end training and evaluation pipeline:
    1. Load configuration and dataset.
    2. Clean and preprocess data.
    3. Train ANN, XGBoost, and Random Forest models.
    4. Save performance evaluation reports and plots.
    5. Save the trained model weights.
    """
    print("Step 1: Loading configuration...")
    config = load_config(config_path)
    
    raw_data_path = config['paths']['raw_data_path']
    models_dir = config['paths']['models_dir']
    plots_dir = config['paths']['plots_dir']
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(
            f"Dataset not found at '{raw_data_path}'. Please download the LendingClub dataset "
            "from Kaggle (lending_club_loan_two.csv) and place it in the 'data/raw/' directory."
        )
        
    print(f"Step 2: Reading dataset from {raw_data_path}...")
    max_rows = config['preprocessing'].get('max_rows', None)
    raw_df = load_data(raw_data_path, nrows=max_rows)
    print(f"Dataset loaded. Initial shape: {raw_df.shape}")
    
    print("Step 3: Cleaning and preprocessing dataset...")
    cleaned_df = clean_data(raw_df, config)
    print(f"Dataset cleaned. Processed shape: {cleaned_df.shape}")
    
    print("Step 4: Splitting and scaling datasets...")
    X_train, y_train, X_test, y_test, scaler = prepare_datasets(cleaned_df, config)
    print(f"Training shape: {X_train.shape}, Testing shape: {X_test.shape}")
    
    # Save the scaler for inference deployment
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {scaler_path}")
    
    # Keep track of models for evaluation and comparison
    models_dict = {}
    
    # ==========================================
    # Model 1: Artificial Neural Network (ANN)
    # ==========================================
    print("\n--- Training Model 1: Artificial Neural Network (ANN) ---")
    ann_cfg = config['models']['ann']
    ann_model = build_ann(X_train.shape[1], config)
    
    ann_history = ann_model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=ann_cfg['epochs'],
        batch_size=ann_cfg['batch_size'],
        verbose=1
    )
    
    # Save ANN History Plot
    ann_history_path = os.path.join(plots_dir, "ann_training_history.png")
    save_learning_evolution(ann_history, ann_history_path)
    print(f"ANN training history plot saved to {ann_history_path}")
    
    # Predictions
    ann_train_pred = ann_model.predict(X_train).ravel()
    ann_test_pred = ann_model.predict(X_test).ravel()
    
    print_model_report(y_train, ann_train_pred.round(), "Neural Network (ANN)", "Train")
    print_model_report(y_test, ann_test_pred.round(), "Neural Network (ANN)", "Test")
    
    # Save ANN model weights
    ann_model_path = os.path.join(models_dir, "ann_model.keras")
    ann_model.save(ann_model_path)
    print(f"ANN Model saved to {ann_model_path}")
    
    models_dict['Artificial Neural Network'] = ann_model
    
    # ==========================================
    # Model 2: XGBoost Classifier
    # ==========================================
    print("\n--- Training Model 2: XGBoost Classifier ---")
    xgb_model = build_xgboost(config)
    xgb_model.fit(X_train, y_train)
    
    # Predictions
    xgb_train_pred = xgb_model.predict(X_train)
    xgb_test_pred = xgb_model.predict(X_test)
    
    print_model_report(y_train, xgb_train_pred, "XGBoost", "Train")
    print_model_report(y_test, xgb_test_pred, "XGBoost", "Test")
    
    # Save XGBoost model weights
    xgb_model_path = os.path.join(models_dir, "xgb_model.json")
    xgb_model.save_model(xgb_model_path)
    print(f"XGBoost Model saved to {xgb_model_path}")
    
    models_dict['XGBoost'] = xgb_model
    
    # ==========================================
    # Model 3: Random Forest Classifier
    # ==========================================
    print("\n--- Training Model 3: Random Forest Classifier ---")
    rf_model = build_random_forest(config)
    rf_model.fit(X_train, y_train)
    
    # Predictions
    rf_train_pred = rf_model.predict(X_train)
    rf_test_pred = rf_model.predict(X_test)
    
    print_model_report(y_train, rf_train_pred, "Random Forest", "Train")
    print_model_report(y_test, rf_test_pred, "Random Forest", "Test")
    
    # Save Random Forest model weights
    rf_model_path = os.path.join(models_dir, "rf_model.pkl")
    with open(rf_model_path, "wb") as f:
        pickle.dump(rf_model, f)
    print(f"Random Forest Model saved to {rf_model_path}")
    
    models_dict['Random Forest'] = rf_model
    
    # ==========================================
    # Step 5: Save Comparison Reports & Plots
    # ==========================================
    print("\nStep 5: Generating model evaluation comparison reports and plots...")
    
    # Save combined ROC Comparison Plot
    roc_comparison_path = os.path.join(plots_dir, "roc_curve_comparison.png")
    save_roc_comparison(models_dict, X_test, y_test, roc_comparison_path)
    print(f"ROC-AUC comparison plot saved to {roc_comparison_path}")
    
    # Save individual Confusion Matrices
    save_confusion_matrix(
        y_test, ann_test_pred.round(),
        ['Default', 'Fully-Paid'],
        os.path.join(plots_dir, "confusion_matrix_ann.png"),
        "Confusion Matrix: ANN Model"
    )
    save_confusion_matrix(
        y_test, xgb_test_pred,
        ['Default', 'Fully-Paid'],
        os.path.join(plots_dir, "confusion_matrix_xgboost.png"),
        "Confusion Matrix: XGBoost Model"
    )
    save_confusion_matrix(
        y_test, rf_test_pred,
        ['Default', 'Fully-Paid'],
        os.path.join(plots_dir, "confusion_matrix_random_forest.png"),
        "Confusion Matrix: Random Forest Model"
    )
    print(f"Confusion Matrix plots saved in {plots_dir}/")
    
    # Log summary performance metrics
    print("\n========================= PIPELINE METRICS SUMMARY =========================")
    for model_name, model in models_dict.items():
        if hasattr(model, 'predict_proba'):
            test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        else:
            test_auc = roc_auc_score(y_test, model.predict(X_test).ravel())
        print(f"{model_name:<30} ROC-AUC Score: {test_auc:.4f}")
    print("============================================================================")
    
    print("\nPipeline execution complete. All models and artifacts saved successfully!")

if __name__ == "__main__":
    run_training_pipeline()
