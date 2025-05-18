import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split # will be used for sampling
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib # Ensure joblib is imported

def load_data(file_path, target_columnn='PATHOLOGY_ENCODED'):
    """Loads data and separates features (X) and target (y)."""
    print(f"Loading data from: {file_path}")
    df = pd.read_csv(file_path)

    X = df.drop(columns=[target_columnn])
    y = df[target_columnn]
    print(f"Original features shape: {X.shape}, Original target shape: {y.shape}")
    return df, X, y # Return the full df as well for sampling

if __name__ == "__main__":
    train_sample_fraction = 1 # Sampling rate of the training data

    # Load training, validation, and test data
    train_file = "Data/reduced_prepared_train.csv"
    validate_file = "Data/reduced_prepared_validate.csv"
    test_file = "Data/reduced_prepared_test.csv"
    
    # Define the full file path for each model
    log_reg_model_path = "Models/logistic_regression_model.joblib"
    rf_model_path = "Models/random_forest_model.joblib"
    mlp_model_path = "Models/mlp_classifier_model.joblib"

    # Load full training data first
    train_df_full, _, _ = load_data(train_file, 'PATHOLOGY_ENCODED') # We only need train_df_full for sampling

    # Sample the training data (stratified)
    '''
    print(f"\nSampling {train_sample_fraction*100:.0f}% of the training data...")
    _, sampled_df = train_test_split(
            train_df_full,
            test_size=train_sample_fraction,
            random_state=42,
            stratify=train_df_full['PATHOLOGY_ENCODED'] # For stratified sampling
            )
    '''
    sampled_df = train_df_full
    X_train = sampled_df.drop(columns=['PATHOLOGY_ENCODED'])
    y_train = sampled_df['PATHOLOGY_ENCODED']
    print(f"Successfully sampled training data. New X_train shape: {X_train.shape}, new y_train shape: {y_train.shape}")

    # Load validation and test data 
    _, X_val, y_val = load_data(validate_file, 'PATHOLOGY_ENCODED') 
    _, X_test, y_test = load_data(test_file, 'PATHOLOGY_ENCODED')   
    # Dict to hold model accuracies
    model_accuracies = {}
    
    # Logistic Regression
    print("\n--- Training Logistic Regression ---")
    # 'saga' solver supports 'multinomial'.
    log_reg = LogisticRegression(
        C=1.0, # Inverse of regularization strength
        solver='saga', 
        multi_class='multinomial', 
        max_iter=200, 
        random_state=42,
        n_jobs=-1 
    )

    # Fit Logistic Regression and Make Predictions on test dataset
    print("Fitting Logistic Regression model...")
    log_reg.fit(X_train, y_train)
    # Save the trained model to the specified path
    joblib.dump(log_reg, log_reg_model_path)
    print(f"Logistic Regression model saved to: {log_reg_model_path}")
    print("Logistic Regression training complete.")
    print("\nEvaluating Logistic Regression on Test Set...")
    y_pred_test_log_reg = log_reg.predict(X_test)
    # Get Accuracy and Classification report
    accuracy_log_reg_test = accuracy_score(y_test, y_pred_test_log_reg)
    model_accuracies['Logistic Regression'] = accuracy_log_reg_test
    print(f"Logistic Regression Test Accuracy: {accuracy_log_reg_test:.4f}")
    print("\nLogistic Regression Test Classification Report:")
    print(classification_report(y_test, y_pred_test_log_reg, zero_division=0))


    # Random Forest
    print("\n--- Training Random Forest ---")
    rf_clf = RandomForestClassifier(
        n_estimators=100,    
        max_depth=40,        
        min_samples_split=10, 
        min_samples_leaf=5,   
        random_state=42,
        n_jobs=-1,
        class_weight='balanced_subsample'
    )

    print("Fitting Random Forest model...")
    rf_clf.fit(X_train, y_train)
    # Save the trained model to the specified path
    joblib.dump(rf_clf, rf_model_path)
    print(f"Random Forest model saved to: {rf_model_path}")
    print("Random Forest training complete.")

    print("\nEvaluating Random Forest on Test Set...")
    y_pred_test_rf = rf_clf.predict(X_test)
    accuracy_rf_test = accuracy_score(y_test, y_pred_test_rf)
    model_accuracies['Random Forest'] = accuracy_rf_test
    print(f"Random Forest Test Accuracy: {accuracy_rf_test:.4f}")
    print("\nRandom Forest Test Classification Report:")
    print(classification_report(y_test, y_pred_test_rf, zero_division=0)) 


    # Neural Network
    print("\n--- Training Neural Network (MLPClassifier) ---")
    mlp_clf = MLPClassifier(
        hidden_layer_sizes=(128, 64), 
        activation='relu',
        solver='adam',
        alpha=0.001,                 
        batch_size='auto',           
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=200,                
        early_stopping=True,         
        validation_fraction=0.1,     
        n_iter_no_change=10,         
        random_state=42,
        verbose=False                 
    )
    # Fit the neural network and make predictions on test dataset
    print("Fitting Neural Network (MLPClassifier) model...")
    mlp_clf.fit(X_train, y_train)
    # Save the trained model to the specified path
    joblib.dump(mlp_clf, mlp_model_path)
    print(f"MLP Classifier model saved to: {mlp_model_path}")
    print("Neural Network training complete.")
    print("\nEvaluating Neural Network on Test Set...")
    y_pred_test_mlp = mlp_clf.predict(X_test)
     # Get Accuracy and Classification report
    accuracy_mlp_test = accuracy_score(y_test, y_pred_test_mlp)
    model_accuracies['Neural Network (MLP)'] = accuracy_mlp_test
    print(f"Neural Network Test Accuracy: {accuracy_mlp_test:.4f}")
    print("\nNeural Network Test Classification Report:")
    print(classification_report(y_test, y_pred_test_mlp, zero_division=0))

    # Model Comparison
    print("\n--- Model Comparison (Test Set Accuracies) ---")
    if model_accuracies:
        for model_name, acc in model_accuracies.items():
            print(f"{model_name}: {acc:.4f}")
    else:
        print("No models were successfully trained and evaluated.")

    print("\nModel training and evaluation process complete.")

