import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# Paths to models
log_reg_path = "Models/logistic_regression_model.joblib"
random_forest_path = "Models/random_forest_model.joblib"
mlp_classifier_path = "Models/mlp_classifier_model.joblib"

# Load the models
log_reg = joblib.load(log_reg_path)
random_forest = joblib.load(random_forest_path)
mlp_classifier = joblib.load(mlp_classifier_path)

print(f"Logistic Regression model loaded from: {log_reg_path}")
print(f"Random Forest model loaded from: {random_forest_path}")
print(f"MLP Classifier model loaded from: {mlp_classifier_path}")

# Load test data
print("Loading test data...")
test_data_path = "Data/reduced_prepared_test.csv"
test_df = pd.read_csv(test_data_path)
X_test = test_df.drop(columns=["PATHOLOGY_ENCODED"])
y_test = test_df["PATHOLOGY_ENCODED"]
print(f"Test data loaded with shape: {X_test.shape}")

# Load transformers to get class names
transformers_path = "Models/transformers.joblib"
transformers = joblib.load(transformers_path)
class_names = transformers.get('pathology_encoder_classes')
print(f"Loaded class names from transformers: {len(class_names)} classes")

# Generate predictions for each model
print("Generating predictions...")
y_pred_log_reg = log_reg.predict(X_test)
y_pred_rf = random_forest.predict(X_test)
y_pred_mlp = mlp_classifier.predict(X_test)

# Dictionary to store all metrics
metrics_data = {}

def evaluate_model(name, y_true, y_pred):
    print(f"\n--- Evaluating {name} ---")
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    # Store metrics for plotting
    metrics_data[name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "MCC": mcc
    }
    
    # Print metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"Matthews Correlation Coefficient: {mcc:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))
    
    # Return confusion matrix for later visualization
    return confusion_matrix(y_true, y_pred)

# Evaluate each model and store confusion matrices
cm_log_reg = evaluate_model("Logistic Regression", y_test, y_pred_log_reg)
cm_rf = evaluate_model("Random Forest", y_test, y_pred_rf)
cm_mlp = evaluate_model("MLP Classifier", y_test, y_pred_mlp)

# Create DataFrame for metrics visualization
metrics_df = pd.DataFrame(metrics_data).T

# PLOT 1: Model Comparison Charts
fig1 = plt.figure(figsize=(15, 10))
plt.suptitle("Model Performance Comparison", fontsize=16)

# 1. Bar chart comparing all metrics across models
ax1 = fig1.add_subplot(2, 1, 1)
bars = metrics_df.plot(kind='bar', ax=ax1)
ax1.set_title('All Metrics by Model')
ax1.set_ylabel('Score')
ax1.set_ylim(0, 1)
# Legend in upper right instead of below the graph
ax1.legend(title='Metric', loc='upper right')
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# Add numerical values on top of bars
for container in bars.containers:
    ax1.bar_label(container, fmt='%.3f', fontsize=8)

# 2. Bar chart for Accuracy only
ax2 = fig1.add_subplot(2, 1, 2)
accuracy_values = [metrics_data[model]['Accuracy'] for model in metrics_data.keys()]
colors = ['green', 'blue', 'purple']
bars = ax2.bar(list(metrics_data.keys()), accuracy_values, color=colors)
ax2.set_title('Model Comparison: Accuracy')
ax2.set_ylabel('Accuracy')
ax2.set_ylim(0, 1)
ax2.grid(axis='y', linestyle='--', alpha=0.7)

# Add numerical values on top of accuracy bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{accuracy_values[i]:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
plt.show()

# Add this after the existing model evaluation code but before plotting the confusion matrices
print("\n--- Analyzing Feature Importance ---")

# Function to plot feature importance
def plot_feature_importance(importance, feature_names, title, filename):
    plt.figure(figsize=(12, 10))
    # Get top 20 features or all if less than 20
    n_features = min(20, len(importance))
    indices = np.argsort(importance)[-n_features:]
    plt.barh(range(n_features), importance[indices])
    plt.yticks(range(n_features), [feature_names[i] for i in indices])
    plt.title(title)
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig(f'visualizations/{filename}')
    plt.close()
    
    # Print top 5 most important features
    print(f"\nTop 5 most important features for {title}:")
    for i in reversed(indices[-5:]):
        print(f"  - {feature_names[i]}: {importance[i]:.4f}")

# Feature names
feature_names = X_test.columns

# 1. Logistic Regression feature importance (coefficient-based)
print("Analyzing Logistic Regression feature importance...")
if hasattr(log_reg, 'coef_'):
    # For multiclass, take the mean of absolute coefficients across all classes
    importance = np.mean(np.abs(log_reg.coef_), axis=0)
    plot_feature_importance(importance, feature_names, 
                           "Logistic Regression Feature Importance", 
                           "logistic_regression_importance.png")

# 2. Random Forest feature importance
print("Analyzing Random Forest feature importance...")
if hasattr(random_forest, 'feature_importances_'):
    importance = random_forest.feature_importances_
    plot_feature_importance(importance, feature_names,
                           "Random Forest Feature Importance",
                           "random_forest_importance.png")

# 3. MLP feature importance using permutation importance
print("Analyzing MLP feature importance using permutation importance (this may take a while)...")
# Use a smaller subset for permutation_importance to save time
sample_size = min(1000, len(X_test))
X_sample = X_test.iloc[:sample_size]
y_sample = y_test.iloc[:sample_size]

if hasattr(mlp_classifier, 'predict'):
    # Use n_repeats=3 for faster computation
    perm_importance = permutation_importance(mlp_classifier, X_sample, y_sample, 
                                           n_repeats=3, random_state=42)
    importance = perm_importance.importances_mean
    plot_feature_importance(importance, feature_names,
                           "Neural Network (MLP) Feature Importance",
                           "mlp_importance.png")


# Shorten long pathology names for better visualization
shortened_class_names = []
for name in class_names:
    if len(name) > 15:
        shortened_class_names.append(name[:13] + "...")
    else:
        shortened_class_names.append(name)

# PLOT 2: Logistic Regression Confusion Matrix (separate figure)
plt.figure(figsize=(14, 12))
plt.title("Logistic Regression: Confusion Matrix", fontsize=16)
sns.heatmap(cm_log_reg, annot=True, fmt='d', cmap='Blues',
           xticklabels=shortened_class_names, yticklabels=shortened_class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# PLOT 3: Random Forest Confusion Matrix (separate figure)
plt.figure(figsize=(14, 12))
plt.title("Random Forest: Confusion Matrix", fontsize=16)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens',
           xticklabels=shortened_class_names, yticklabels=shortened_class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# PLOT 4: MLP Classifier Confusion Matrix (separate figure)
plt.figure(figsize=(14, 12))
plt.title("MLP Classifier: Confusion Matrix", fontsize=16)
sns.heatmap(cm_mlp, annot=True, fmt='d', cmap='Purples',
           xticklabels=shortened_class_names, yticklabels=shortened_class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


#######################################################
def analyze_errors(model_name, y_true, y_pred, class_names):
    # Find indices of misclassified samples
    error_indices = np.where(y_pred != y_true)[0]
    
    if len(error_indices) == 0:
        print(f"No errors for {model_name}!")
        return
    
    # Count errors by true class
    error_by_class = {}
    for idx in error_indices:
        true_class = y_true[idx]
        pred_class = y_pred[idx]
        
        true_class_name = class_names[true_class]
        pred_class_name = class_names[pred_class]
        
        if true_class_name not in error_by_class:
            error_by_class[true_class_name] = {'count': 0, 'predictions': {}}
        
        error_by_class[true_class_name]['count'] += 1
        
        if pred_class_name not in error_by_class[true_class_name]['predictions']:
            error_by_class[true_class_name]['predictions'][pred_class_name] = 0
        
        error_by_class[true_class_name]['predictions'][pred_class_name] += 1
    
    # Sort by number of errors
    sorted_errors = sorted(error_by_class.items(), 
                          key=lambda x: x[1]['count'], reverse=True)
    
    # Display top 5 classes with most errors
    print(f"\nTop 5 misclassified pathologies for {model_name}:")
    for class_name, error_info in sorted_errors[:5]:
        print(f"\n{class_name} - {error_info['count']} errors")
        print("  Most frequently predicted as:")
        
        # Sort and show top 3 incorrect predictions
        sorted_preds = sorted(error_info['predictions'].items(), 
                             key=lambda x: x[1], reverse=True)
        for wrong_class, count in sorted_preds[:3]:
            print(f"  - {wrong_class}: {count} times")

# Analyze errors for each model
analyze_errors("Logistic Regression", y_test, y_pred_log_reg, class_names)
analyze_errors("Random Forest", y_test, y_pred_rf, class_names)
analyze_errors("MLP Classifier", y_test, y_pred_mlp, class_names)
