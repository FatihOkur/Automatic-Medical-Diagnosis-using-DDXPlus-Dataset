import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay # Added for your plotting style
)
import matplotlib.pyplot as plt
import seaborn as sns # Seaborn can also be useful for CMs, but sticking to ConfusionMatrixDisplay as per your snippet

def load_data(file_path, target_column='PATHOLOGY_ENCODED'):
    """Loads data and separates features (X) and target (y)."""
    print(f"Loading data from: {file_path}")
    df = pd.read_csv(file_path)
    X = df.drop(columns=[target_column])
    y = df[target_column]
    print(f"Features shape: {X.shape}, Target shape: {y.shape}")
    return X, y

def calculate_all_metrics(y_true, y_pred, class_names, model_name_for_report):
    """Calculates and returns a dictionary of all specified metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    # Using 'weighted' average for precision, recall, F1 to account for label imbalance
    # This is consistent with how you might want to compare overall model performance
    precision_w = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall_w = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_w = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    print(f"\n--- Metrics for Model: {model_name_for_report} ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Weighted Precision: {precision_w:.4f}")
    print(f"Weighted Recall: {recall_w:.4f}")
    print(f"Weighted F1-score: {f1_w:.4f}")
    print(f"Matthews Correlation Coefficient: {mcc:.4f}")

    print("\nClassification Report:")
    report_str = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    print(report_str)
    
    metrics_dict = {
        'Accuracy': accuracy,
        'Precision (Weighted)': precision_w,
        'Recall (Weighted)': recall_w,
        'F1-score (Weighted)': f1_w,
        'MCC': mcc,
        'Classification Report': report_str
    }
    return metrics_dict

if __name__ == "__main__":
    # Define file paths
    test_file_path = "Data/reduced_prepared_test.csv"
    transformers_path = "Models/transformers.joblib"

    model_paths_and_names = {
        "Logistic Regression": "Models/logistic_regression_model.joblib",
        "Random Forest": "Models/random_forest_model.joblib",
        "MLP Classifier": "Models/mlp_classifier_model.joblib" # Your training script called it MLP Classifier
    }
    
    # Ensure the 'Plots' directory exists for saving confusion matrices if needed later
    # For now, we will just display them as per your snippets.
    if not os.path.exists("Plots"):
        os.makedirs("Plots")
        print("Created 'Plots' directory for saving figures.")

    # Load transformers to get pathology class names
    print(f"Loading transformers from: {transformers_path}")
    transformers = joblib.load(transformers_path)
    pathology_encoder_classes = transformers.get('pathology_encoder_classes')
    if pathology_encoder_classes is None:
        print("Error: 'pathology_encoder_classes' not found in transformers.joblib.")
        print("Please ensure your preprocessing script saves this. Exiting.")
        exit()
    print(f"Pathology classes loaded: {len(pathology_encoder_classes)} classes.")

    # Load test data
    X_test, y_test = load_data(test_file_path)
    if X_test is None: # Basic check
        print("Exiting due to error in loading test data.")
        exit()

    # Store predictions and metrics for each model
    model_predictions = {}
    all_model_metrics_summary = {} # For scalar metrics for tables/bar charts
    
    # Load models, make predictions, and calculate initial metrics
    for model_name, model_path in model_paths_and_names.items():
        if os.path.exists(model_path):
            print(f"\nLoading and predicting with model: {model_name} from {model_path}")
            model = joblib.load(model_path)
            y_pred = model.predict(X_test)
            model_predictions[model_name] = y_pred
            
            # Calculate and store metrics
            metrics = calculate_all_metrics(y_test, y_pred, pathology_encoder_classes, model_name)
            all_model_metrics_summary[model_name] = {
                'Accuracy': metrics['Accuracy'],
                'Precision': metrics['Precision (Weighted)'], # Using weighted for bar chart consistency
                'Recall': metrics['Recall (Weighted)'],     # Using weighted for bar chart consistency
                'F1-score': metrics['F1-score (Weighted)'] # Using weighted for bar chart consistency
            }
        else:
            print(f"Warning: Model file not found for {model_name} at {model_path}. Skipping.")
            # Add placeholder predictions and metrics if model is missing to avoid crashing plots
            # Or, adjust plotting logic to handle missing models
            model_predictions[model_name] = np.zeros_like(y_test) # Placeholder
            all_model_metrics_summary[model_name] = {'Accuracy': 0, 'Precision': 0, 'Recall': 0, 'F1-score': 0}


    # --- Plotting based on your snippets ---
    
    # Data for plots - ensure order matches your labels
    # Your snippet had 'Neural Network (MLP)' then 'Random Forest' then 'Logistic Regression'
    # My loop is "Logistic Regression", "Random Forest", "MLP Classifier"
    # I will use the order from my loop for consistency with calculated metrics.
    
    plot_labels_ordered = ["Random Forest", "MLP Classifier", "Logistic Regression"] # Consistent order for plots
    
    # Ensure all models in plot_labels_ordered exist in all_model_metrics_summary
    for label in plot_labels_ordered:
        if label not in all_model_metrics_summary:
            print(f"Warning: Model '{label}' not found in calculated metrics. Plots may be incorrect or fail.")
            # Add placeholder if missing to prevent crash, though ideally all models are present
            if label not in all_model_metrics_summary:
                 all_model_metrics_summary[label] = {'Accuracy': 0, 'Precision': 0, 'Recall': 0, 'F1-score': 0}


    # Comparing the Models for Accuracy
    plt.figure(figsize=(8,6))
    accuracy_values = [all_model_metrics_summary[model]['Accuracy'] for model in plot_labels_ordered]
    plt.bar(plot_labels_ordered, accuracy_values, color=['green', 'cyan', 'yellow'])
    plt.title("Comparison of Models for Accuracy")
    plt.ylabel("Accuracy of the Model")
    plt.ylim(0, 1)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.show()

    # Comparison for Precision (Weighted)
    plt.figure(figsize=(8,6))
    precision_values = [all_model_metrics_summary[model]['Precision'] for model in plot_labels_ordered]
    plt.bar(plot_labels_ordered, precision_values, color=['green', 'cyan', 'yellow'])
    plt.title("Comparison of Models for Weighted Precision")
    plt.ylabel("Weighted Precision of the Model") # Corrected Y-axis label
    plt.ylim(0, 1)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.show()

    # Comparison for F1 score (Weighted)
    plt.figure(figsize=(8,6))
    f1_values = [all_model_metrics_summary[model]['F1-score'] for model in plot_labels_ordered]
    plt.bar(plot_labels_ordered, f1_values, color=['green', 'cyan', 'yellow'])
    plt.title("Comparison of Models for Weighted F1-score")
    plt.ylabel("Weighted F1-score of the Model") # Corrected Y-axis label
    plt.ylim(0, 1)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.show()

    # Combined metrics plot
    # The user's snippet for this part was slightly different, adapting it:
    # metrics_df = pd.DataFrame(metrics, index=['Accuracy', 'Precision', 'Recall', 'F1-score']).T
    # This structure is for a single model's metrics. We need to adapt for multiple models.
    
    # Reconstruct metrics_df for the combined plot as per user's snippet style
    # The keys in all_model_metrics_summary are 'Accuracy', 'Precision', 'Recall', 'F1-score'
    metrics_for_df_plot = {}
    for model_name_key in plot_labels_ordered: # Use the defined order
        # Ensure model_name_key is exactly as in all_model_metrics_summary keys
        # The keys in all_model_metrics_summary are "Logistic Regression", "Random Forest", "MLP Classifier"
        # The plot_labels_ordered is ["Random Forest", "MLP Classifier", "Logistic Regression"]
        # Need to map them if they are different or ensure consistency.
        # For simplicity, let's assume plot_labels_ordered matches the keys in all_model_metrics_summary
        if model_name_key in all_model_metrics_summary:
             metrics_for_df_plot[model_name_key] = [
                all_model_metrics_summary[model_name_key]['Accuracy'],
                all_model_metrics_summary[model_name_key]['Precision'], # This is Weighted Precision
                all_model_metrics_summary[model_name_key]['Recall'],    # This is Weighted Recall
                all_model_metrics_summary[model_name_key]['F1-score'] # This is Weighted F1-score
            ]
        else: # Should not happen if check above is done
            metrics_for_df_plot[model_name_key] = [0,0,0,0]


    metrics_df_transposed = pd.DataFrame(metrics_for_df_plot, index=['Accuracy', 'Precision (W)', 'Recall (W)', 'F1-score (W)'])
    # Transpose it to have models as rows, metrics as columns, then plot
    metrics_df_transposed.T.plot(kind='bar', figsize=(12, 8))
    plt.title('Model Comparison: Accuracy, Weighted Precision, Weighted Recall, Weighted F1-score')
    plt.ylabel('Score')
    plt.ylim(0, 1.05) # Adjusted ylim slightly for legend
    plt.xticks(rotation=0) # Model names on x-axis
    #plt.xlabel('Performance Metric') # X-axis label is now 'Model' due to transpose
    plt.xlabel('Model')
    plt.legend(title='Performance Metric', loc='lower right') # Legend title is now 'Performance Metric'
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    plt.show()

    # Confusion Matrices (for the first 30 samples as per your snippet)
    num_samples_for_cm = 30
    y_test_subset = y_test[:num_samples_for_cm]
    
    # Ensure class names for CM display are also subset if y_test_subset doesn't contain all classes
    # However, ConfusionMatrixDisplay typically handles this by showing only relevant classes or all original classes
    # For simplicity, we use all original class names.
    
    # Confusion Matrix for Logistic Regression
    if "Logistic Regression" in model_predictions:
        y_pred_log_reg_subset = model_predictions["Logistic Regression"][:num_samples_for_cm]
        cm_log_reg = confusion_matrix(y_test_subset, y_pred_log_reg_subset, labels=np.unique(np.concatenate((y_test_subset,y_pred_log_reg_subset))))
        disp_log_reg = ConfusionMatrixDisplay(confusion_matrix=cm_log_reg, display_labels=np.unique(np.concatenate((y_test_subset,y_pred_log_reg_subset)))) # Use unique labels present in subset
        disp_log_reg.plot(cmap='Blues')
        plt.title(f"Confusion Matrix for Logistic Regression (First {num_samples_for_cm} Samples)")
        plt.show()

    # Confusion Matrix for Random Forest
    if "Random Forest" in model_predictions:
        y_pred_rf_subset = model_predictions["Random Forest"][:num_samples_for_cm]
        cm_rf = confusion_matrix(y_test_subset, y_pred_rf_subset, labels=np.unique(np.concatenate((y_test_subset,y_pred_rf_subset))))
        disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=np.unique(np.concatenate((y_test_subset,y_pred_rf_subset))))
        disp_rf.plot(cmap='Greens')
        plt.title(f"Confusion Matrix for Random Forest (First {num_samples_for_cm} Samples)")
        plt.show()

    # Confusion Matrix for Neural Network (MLP Classifier)
    if "MLP Classifier" in model_predictions:
        y_pred_mlp_subset = model_predictions["MLP Classifier"][:num_samples_for_cm]
        cm_mlp = confusion_matrix(y_test_subset, y_pred_mlp_subset, labels=np.unique(np.concatenate((y_test_subset,y_pred_mlp_subset))))
        # User snippet had disprand_forest here, correcting to disp_mlp
        disp_mlp = ConfusionMatrixDisplay(confusion_matrix=cm_mlp, display_labels=np.unique(np.concatenate((y_test_subset,y_pred_mlp_subset))))
        disp_mlp.plot(cmap='Greys') # User snippet had 'Greys', keeping it. Could also use 'Oranges' or 'Purples' for MLP
        plt.title(f"Confusion Matrix for Neural Network (MLP) (First {num_samples_for_cm} Samples)")
        plt.show()

    print("\nDetailed model evaluation and comparison plotting complete.")

