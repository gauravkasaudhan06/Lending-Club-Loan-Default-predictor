import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)

def print_model_report(y_true, y_pred, model_name, dataset_type="Test"):
    """
    Format and print accuracy, classification report, and confusion matrix.
    """
    clf_report = pd.DataFrame(classification_report(y_true, y_pred, output_dict=True))
    print(f"\n=================== {model_name} ({dataset_type} Result) ===================")
    print(f"Accuracy Score: {accuracy_score(y_true, y_pred) * 100:.2f}%")
    print("_______________________________________________")
    print("CLASSIFICATION REPORT:")
    print(clf_report)
    print("_______________________________________________")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("==================================================================\n")

def save_confusion_matrix(y_true, y_pred, display_labels, save_path, title):
    """
    Generate and save a confusion matrix plot.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(cmap='Blues', values_format='d', ax=ax)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def save_learning_evolution(history, save_path):
    """
    Plot and save Keras Neural Network training history (Loss & AUC curves).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(12, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss Evolution During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # AUC plot
    plt.subplot(1, 2, 2)
    auc_key = [k for k in history.history.keys() if 'auc' in k.lower() and 'val' not in k.lower()][0]
    plt.plot(history.history[auc_key], label='Train AUC')
    val_auc_key = [k for k in history.history.keys() if 'auc' in k.lower() and 'val' in k.lower()]
    if val_auc_key:
        plt.plot(history.history[val_auc_key[0]], label='Val AUC')
    plt.title('AUC Score Evolution During Training')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def save_roc_comparison(models_dict, X_test, y_test, save_path):
    """
    Plot ROC-AUC curves for multiple models on a single chart and save it.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(8, 6))
    
    for model_name, model in models_dict.items():
        # Handle predictions depending on whether the model is Keras (TensorFlow) or Scikit-learn/XGBoost
        if hasattr(model, 'predict_proba'):
            y_probs = model.predict_proba(X_test)[:, 1]
        else:  # For Keras ANN model
            y_probs = model.predict(X_test).ravel()
            
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        auc_score = roc_auc_score(y_test, y_probs)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_score:.3f})")
        
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing (AUC = 0.500)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Performance Comparison')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
