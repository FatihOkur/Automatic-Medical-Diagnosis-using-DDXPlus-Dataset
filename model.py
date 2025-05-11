import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
 
# Load datasets
train_df = pd.read_csv('Data/prepared_train_reduced.csv')
test_df = pd.read_csv('Data/prepared_test_reduced.csv')
validate_df = pd.read_csv('Data/prepared_validate_reduced.csv')

print(f"Dataset shapes - Train: {train_df.shape}, Validate: {validate_df.shape}, Test: {test_df.shape}")

# Separate features and targets
X_train = train_df.drop('PATHOLOGY', axis=1)
y_train = train_df['PATHOLOGY']

X_validate = validate_df.drop('PATHOLOGY', axis=1)
y_validate = validate_df['PATHOLOGY']

X_test = test_df.drop('PATHOLOGY', axis=1)
y_test = test_df['PATHOLOGY']

# Encode target variables
le = LabelEncoder() 
y_train_encoded = le.fit_transform(y_train)
y_validate_encoded = le.transform(y_validate)
y_test_encoded = le.transform(y_test)

# Sample 10% of training data with stratification
print("Sampling 10% of training data...")
X_train_sampled, y_train_encoded_sampled = resample(
    X_train, y_train_encoded, 
    n_samples=int(0.1*len(X_train)), 
    random_state=42, 
    stratify=y_train_encoded
)
print(f"Sampled dataset size: {len(X_train_sampled)}")

# 1. Train Random Forest with early stopping
print("\nTraining Random Forest with early stopping...")
max_trees = 200
early_stop_patience = 5
min_trees = 20
best_val_score = 0
no_improve_count = 0

rf_clf = RandomForestClassifier(
    n_estimators=min_trees,
    max_depth=10,
    min_samples_leaf=5,
    max_features='sqrt',
    bootstrap=True,
    random_state=42,
    warm_start=True
)

rf_clf.fit(X_train_sampled, y_train_encoded_sampled)
current_trees = min_trees

y_val_pred = rf_clf.predict(X_validate)
current_score = accuracy_score(y_validate_encoded, y_val_pred)
best_val_score = current_score

print(f"Trees: {current_trees}, Validation Accuracy: {current_score:.4f}")

while current_trees < max_trees:
    step_size = 10
    rf_clf.n_estimators += step_size
    current_trees += step_size
    
    rf_clf.fit(X_train_sampled, y_train_encoded_sampled)
    
    y_val_pred = rf_clf.predict(X_validate)
    current_score = accuracy_score(y_validate_encoded, y_val_pred)
    
    print(f"Trees: {current_trees}, Validation Accuracy: {current_score:.4f}")
    
    if current_score > best_val_score + 0.0001:
        best_val_score = current_score
        no_improve_count = 0
    else:
        no_improve_count += 1
    
    if no_improve_count >= early_stop_patience:
        print(f"Early stopping triggered at {current_trees} trees")
        break

y_pred_rf = rf_clf.predict(X_test)
 
# 2. Train XGBoost with validation set for early stopping
print("\nTraining XGBoost...")
xgb_clf = XGBClassifier(
    n_estimators=50,
    learning_rate=0.3,
    max_depth=4,
    reg_lambda=1,
    reg_alpha=0.5,
    subsample=0.3,
    colsample_bytree=0.5,
    tree_method='hist',
    early_stopping_rounds=3,
    eval_metric='mlogloss',
    use_label_encoder=False,
    random_state=42
)

xgb_clf.fit(
    X_train_sampled, y_train_encoded_sampled,
    eval_set=[(X_validate, y_validate_encoded)],
    verbose=False
)
y_pred_xgb = xgb_clf.predict(X_test)

# 3. Train Neural Network with validation monitoring
print("\nTraining Neural Network...")
mlp_clf = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    solver='adam',
    max_iter=500,
    alpha=0.01,
    random_state=42
)

# Initialize network with one partial fit
mlp_clf.partial_fit(X_train_sampled, y_train_encoded_sampled, classes=np.unique(y_train_encoded))

best_val_loss = float('inf')
no_improve_count = 0
patience = 5
max_epochs = 100

for epoch in range(max_epochs):
    mlp_clf.partial_fit(X_train_sampled, y_train_encoded_sampled)
    
    val_pred = mlp_clf.predict(X_validate)
    val_acc = accuracy_score(y_validate_encoded, val_pred)
    val_loss = 1 - val_acc
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Validation Accuracy = {val_acc:.4f}")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve_count = 0
    else:
        no_improve_count += 1
    
    if no_improve_count >= patience:
        print(f"Early stopping at epoch {epoch}")
        break

y_pred_nn = mlp_clf.predict(X_test)

# Convert predictions back to original labels for evaluation
y_pred_rf_decoded = le.inverse_transform(y_pred_rf)
y_pred_xgb_decoded = le.inverse_transform(y_pred_xgb)
y_pred_nn_decoded = le.inverse_transform(y_pred_nn)

# Evaluate Models with weighted average for multi-class metrics
def evaluate_model(y_test, y_pred, model_name):
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"=== {model_name} ===")
    print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1-score: {f1:.4f}")
    return acc, prec, rec, f1
 
# Store metrics
print("\nEvaluating models on test set:")
metrics = {}
metrics['Random Forest'] = evaluate_model(y_test, y_pred_rf_decoded, "Random Forest")
metrics['XGBoost'] = evaluate_model(y_test, y_pred_xgb_decoded, "XGBoost")
metrics['Neural Network'] = evaluate_model(y_test, y_pred_nn_decoded, "Neural Network")

# Plotting Performance Comparison
metrics_df = pd.DataFrame(metrics, index=['Accuracy', 'Precision', 'Recall', 'F1-score']).T
metrics_df.plot(kind='bar', figsize=(12, 8))
 
plt.title('Model Comparison on Medical Diagnosis Task')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Metric')
plt.tight_layout()
plt.show()

# Print best performing model
best_model = metrics_df['Accuracy'].idxmax()
print(f"\nBest model by accuracy: {best_model} ({metrics_df.loc[best_model, 'Accuracy']:.4f})")