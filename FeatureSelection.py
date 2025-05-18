import pandas as pd
import numpy as np
import os
from sklearn.feature_selection import VarianceThreshold

def reduce_dataset_by_correlation(input_file_path, output_file_path, selected_features, target_column):
    """
    Loads a dataset, keeps only selected features and the target, and saves it.
    """
    print(f"Processing {input_file_path}...")\
    
    # Load the data
    df = pd.read_csv(input_file_path)

    # Get selected features
    features_selected = [feature for feature in selected_features if feature in df.columns]
    # Determine columns to keep
    columns_to_keep = features_selected + [target_column]

    # Create a dataframe for reduced dataset
    df_reduced = df[columns_to_keep]
    print(f"Original shape for {input_file_path}: {df.shape}, Reduced shape: {df_reduced.shape}")

    # Save reduced data to a csv
    print(f"Attempting to save reduced data to csv: {output_file_path}")
    df_reduced.columns = df_reduced.columns.astype(str)
    df_reduced.to_csv(output_file_path, index=False)
    print(f"Successfully saved reduced data to: {output_file_path}")


if __name__ == "__main__":
    prepared_train_file_in = "Data/prepared_train.csv"
    prepared_validate_file_in = "Data/prepared_validate.csv"
    prepared_test_file_in = "Data/prepared_test.csv"

    reduced_train_file_out = "Data/reduced_prepared_train.csv"
    reduced_validate_file_out = "Data/reduced_prepared_validate.csv"
    reduced_test_file_out = "Data/reduced_prepared_test.csv"

    # --- Thresholds ---
    # Variance Threshold: Remove features where p*(1-p) < threshold.
    # e.g., for p=0.02 (2% or 98% same value), variance = 0.02*0.98 = 0.0196
    variance_filter_threshold = 0.0196 
    # For feature-target correlation:
    target_correlation_threshold = 0.04 
    # For feature-feature correlation (multicollinearity):
    multicollinearity_threshold = 0.70 

    # Load training data
    print(f"Loading training data for feature selection: {prepared_train_file_in}")
    train_df = pd.read_csv(prepared_train_file_in)
    print("Successfully loaded training data")
    # Initial features before feature selection
    X_all_train_features_initial = train_df.drop(columns=['PATHOLOGY_ENCODED'])
    # Targets for train dataset
    y_train = train_df['PATHOLOGY_ENCODED']

    # This part is for safety, we are ensuring all the feature columns are numeric
    print("Converting feature columns to numeric...")
    for col in X_all_train_features_initial.columns:
        current_dtype = X_all_train_features_initial[col].dtype
        X_all_train_features_initial[col] = pd.to_numeric(X_all_train_features_initial[col], errors='coerce')
        if X_all_train_features_initial[col].dtype == 'float64' and np.issubdtype(current_dtype, np.floating) and current_dtype != 'float64':
            X_all_train_features_initial[col] = X_all_train_features_initial[col].astype(current_dtype) 
        elif X_all_train_features_initial[col].dtype == 'float64' and np.issubdtype(current_dtype, np.integer) :
             pass # Keep as float64 if NaNs were introduced from int

    # Check and delete columns that are all Nan
    print("Checking for columns that are all NaN...")
    cols_to_drop_all_nan = []
    for col in X_all_train_features_initial.columns:
        if X_all_train_features_initial[col].isnull().all():
            cols_to_drop_all_nan.append(col)
    if cols_to_drop_all_nan:
        print(f"Dropping columns that are all NaN: {cols_to_drop_all_nan}")
        X_all_train_features_initial = X_all_train_features_initial.drop(columns=cols_to_drop_all_nan)
    else:
        print("No columns found to be all NaN.")
    # Number of initial features
    initial_feature_count = len(X_all_train_features_initial.columns)
    print("Number of features before feature selection: ", initial_feature_count)
    
    # Apply Variance Threshold
    print(f"\nApplying Variance Threshold (threshold = {variance_filter_threshold})")
    # Fill all the Nan values in the initial train features with 0
    X_train_for_variance = X_all_train_features_initial.fillna(0) 
    # Apply Variance Threshold
    selector_variance = VarianceThreshold(threshold=variance_filter_threshold)
    selector_variance.fit(X_train_for_variance)
    # Get the selected features after variance threshold
    selected_features_after_variance = X_all_train_features_initial.columns[selector_variance.get_support()].tolist() 
    num_dropped_variance = initial_feature_count - len(selected_features_after_variance)
    print(f"Number of features dropped by Variance Threshold: {num_dropped_variance}")
    print(f"Number of features remaining after Variance Threshold: {len(selected_features_after_variance)}")
    # Current Features
    X_all_train_features = X_all_train_features_initial[selected_features_after_variance]


    # Correlation between target and the features
    print(f"\nStep 1: Selecting features based on correlation with target (threshold > {target_correlation_threshold})")
    print("Calculating feature-target correlations...")
    # Correlations between features and the target
    feature_target_correlations = X_all_train_features.corrwith(y_train)
    # Take abs values of the correlations
    abs_feature_target_correlations = feature_target_correlations.abs().sort_values(ascending=False)
    selected_features_target_correlations = abs_feature_target_correlations[abs_feature_target_correlations >= target_correlation_threshold].index.tolist()
    print(f"Number of features after target correlation filtering: {len(selected_features_target_correlations)}")

    X_train_selected_target_correlations = X_all_train_features[selected_features_target_correlations]


    # Correlation between feature and feature
    print(f"\nStep 2: Removing highly correlated features (multicollinearity threshold > {multicollinearity_threshold})")
    print("Calculating feature-feature correlation matrix...")
    corr_matrix_features = X_train_selected_target_correlations.corr().abs()

    upper_triangle = corr_matrix_features.where(np.triu(np.ones(corr_matrix_features.shape), k=1).astype(bool))
    features_to_drop_multicollinearity = set()
        
    for column in upper_triangle.columns:
        if column in features_to_drop_multicollinearity:
            continue
        highly_correlated_with_column = upper_triangle[upper_triangle[column] > multicollinearity_threshold].index
        for correlated_feature in highly_correlated_with_column:
            if correlated_feature in features_to_drop_multicollinearity:
                continue
            corr_val_column = abs_feature_target_correlations.get(column, 0) # Use pre-calculated target correlations
            corr_val_correlated_feature = abs_feature_target_correlations.get(correlated_feature, 0)
            if corr_val_column >= corr_val_correlated_feature:
                features_to_drop_multicollinearity.add(correlated_feature)
            else:
                features_to_drop_multicollinearity.add(column)
                break 
    selected_features_final = [f for f in selected_features_target_correlations if f not in features_to_drop_multicollinearity]
    print(f"Number of features dropped due to multicollinearity: {len(features_to_drop_multicollinearity)}")

    print(f"\nTotal features in original training data (excluding target): {initial_feature_count}")
    print(f"Features remaining after Variance Threshold: {len(X_all_train_features.columns)}")
    print(f"Features remaining after target correlation filtering: {len(selected_features_target_correlations)}")
    print(f"Features remaining after multicollinearity filtering: {len(selected_features_final)}")

    # Saving reduced datasets
    print("\n--- Applying final feature selection and saving datasets ---")
    reduce_dataset_by_correlation(prepared_train_file_in, reduced_train_file_out, selected_features_final, 'PATHOLOGY_ENCODED')
    reduce_dataset_by_correlation(prepared_validate_file_in, reduced_validate_file_out, selected_features_final, 'PATHOLOGY_ENCODED')
    reduce_dataset_by_correlation(prepared_test_file_in, reduced_test_file_out, selected_features_final, 'PATHOLOGY_ENCODED')

    print("\nDataset reduction process complete.")