import pandas as pd
import ast
import json
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import numpy as np
import joblib

def generate_all_evidence_feature_names(evidences_json_path):
    """
    Generates a list of all possible evidences using release_evidences.json
    Args:
        evidences_json_path (str): Path to the release_evidences.json file.
    Returns:
        list: A sorted list of unique feature names, or None if an error occurs.
              Example feature names: "EVIDENCE_E_91", "EVIDENCE_E_130_at_V_86"
    """
    print(f"Loading evidences from: ", evidences_json_path)
    # Open the json files
    with open(evidences_json_path, 'r') as f:
        evidences_data = json.load(f)

    # List for storing evidence features
    evidence_features = []
    # Process evidences
    for evidence_code, evidence_details in evidences_data.items():
        # Get the name and data_type of the current evidence
        evidence_name = evidence_details.get('name')
        data_type = evidence_details.get('data_type')

        if not evidence_name or not data_type:
            print(f"Warning: Skipping evidence (key: {evidence_code}) due to missing 'name' or 'data_type': {evidence_details}")
            continue
        # Get the possible values of current evidence
        possible_values = evidence_details.get('possible-values', [])

        if data_type == 'B': # Binary
            # Append the binary evidence
            evidence_features.append(f"EVIDENCE_{evidence_name}")
        elif data_type in ['C', 'M']: # Categorical or Multi-choice
            for value_code in possible_values:
                # Append the categorical or multi-choice evidences
                evidence_features.append(f"EVIDENCE_{evidence_name}_at_{value_code}")
        else:
            print(f"Warning: Unknown data_type '{data_type}' for evidence {evidence_name}.")
    print(f"Generated {len(evidence_features)} unique evidence feature names from schema.")
    return sorted(list(set(evidence_features))) # Sort for consistent order and ensure uniqueness


def preprocess_data(df_path, evidence_feature_names, is_train=True, fitted_transformers=None):
    """
    Preprocesses the DDXPlus dataset using a predefined list of evidence features from the schema.
    Args
        df_path (str): Path to the raw CSV data file (train, validate, or test).
        all_schema_evidence_feature_names (list): A comprehensive list of feature names
                                                 derived from release_evidences.json.
        is_train (bool): True if processing training data (to fit transformers),
                         False for validation/test data (to use fitted transformers).
        fitted_transformers (dict, optional): A dictionary containing transformers
                                              fitted on the training data. Required if is_train is False.
    Returns:
        tuple:
            - X_processed (pd.DataFrame): The processed feature matrix.
            - y_processed (pd.Series): The processed target variable.
            - fitted_transformers (dict): If is_train is True, this dictionary is returned,
                                          containing all fitted transformers.
            Returns (None, None) or (None, None, None) if an error occurs during file loading.
    """
    print(f"Loading data from: {df_path}")
    # Load the data
    df = pd.read_csv(df_path)

    # Handle 'EVIDENCES'
    df['EVIDENCES'] = df['EVIDENCES'].fillna('[]')
    # Will convert the text string into an actual python list ( "['E_48']" -> ['E_48'])
    def parse_evidences(evidence_str):
        return ast.literal_eval(evidence_str)
    # Create a new column for parsed evidences
    df['EVIDENCES_PARSED'] = df['EVIDENCES'].apply(parse_evidences)


    # Encode Target Variable ('PATHOLOGY')
    if is_train: # If it is train dataset
        # Create LabelEncoder for pathology
        pathology_encoder = LabelEncoder()
        # Do fit and transform
        y = pathology_encoder.fit_transform(df['PATHOLOGY'])
        # Check if there are fitted_transformers
        if fitted_transformers is None:
            fitted_transformers = {} # if not create a dict for it
        # The fitted encoder and the list of actual class names are stored in fitted_transforms
        fitted_transformers['pathology_encoder'] = pathology_encoder
        fitted_transformers['pathology_encoder_classes'] = pathology_encoder.classes_
    else: # If it is not train dataset (validate or test)
        # If it is previously seen value from train dataset (which it is since pathologies are same)
        # It gets its corresponding number, if it is not seen it gets the value -1
        # Do just transform
        y = fitted_transformers['pathology_encoder'].transform(df['PATHOLOGY'])
    
    # Create pandas series for encoded pathology, this will be our target for the machine learning model
    y = pd.Series(y, name='PATHOLOGY_ENCODED', index=df.index)

    
    # Feature engineering for 'EVIDENCES_PARSED' using all evidence_feature_names
    print("Constructing evidence features...")
    # Create a dataframe for all evidence features which is intialized to 0
    # int8 is used for memory reasons
    evidence_df = pd.DataFrame(0, index=df.index, columns=evidence_feature_names, dtype='int8')

    for i, patient_evidences_list in enumerate(df['EVIDENCES_PARSED']):
        if i % 100000 == 0 and i > 0: # Print the progress
            print(f"Processed {i} patients for evidence features...")
        for evidence_item_str in patient_evidences_list:
            # evidence_item_str can be binary or categorical/multi-choice
            parts = evidence_item_str.split('@_') # Split by '@_', since it is separator in patient data
            base_evidence_code = str(parts[0]) # evidence code
            if len(parts) == 1: # Binary evidence
                feature_name = f"EVIDENCE_{base_evidence_code}"
            elif len(parts) == 2: # Categorical/Multi-choice evidence
                value_code = str(parts[1]) # Value code
                feature_name = f"EVIDENCE_{base_evidence_code}_at_{value_code}"
            else:
                # If length of parts is not an expected value, print an error message and continue
                print(f"Warning: Unexpected evidence format for patient: {evidence_item_str}")
                continue 
            # If feature_name, which is constructed from patient's data, exists in evidence_df
            if feature_name in evidence_df.columns:
                # The value of the cell (patient (i), feature_name of that patient) is set to 1, indicating patient (i) has this specific evidence (feature_name)
                evidence_df.loc[df.index[i], feature_name] = 1    
    print("Evidence features construction is completed.")


    # Encode Categorical Feature ('SEX')
    # Get the sex column
    sex_column = df[['SEX']]
    if is_train: # if it is train dataset
        # Apply one hot encoding to the sex feature
        sex_ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore', dtype=np.int8)
        # Do fit and transform
        sex_encoded = sex_ohe.fit_transform(sex_column)
        # Get the ohe feature names
        ohe_feature_names = sex_ohe.get_feature_names_out(['SEX']) 
        # The fitted encoder and the feature names for it stored in fitted_transformmers
        fitted_transformers['sex_ohe'] = sex_ohe
        fitted_transformers['sex_ohe_feature_names'] = ohe_feature_names
    else:
        # Get the sex_ohe that fitted to the training dataset
        sex_ohe = fitted_transformers['sex_ohe']
        # Just do transform
        sex_encoded = sex_ohe.transform(sex_column)
        # Get ohe_feature_names
        ohe_feature_names = fitted_transformers.get('sex_ohe_feature_names', sex_ohe.get_feature_names_out(['SEX']))
    # Create a dataframe for encoded sex values
    sex_df = pd.DataFrame(sex_encoded, columns=ohe_feature_names, index=df.index, dtype='int8')


    # Combine Features (AGE will be scaled separately)
    age_df = df[['AGE']].astype(np.float32)
    X_processed = pd.concat([age_df, sex_df, evidence_df], axis=1)


    # Scale Only Age column
    if is_train: # if it is train dataset
        age_scaler = StandardScaler()
        # Fit and transform the age data
        X_processed['AGE'] = age_scaler.fit_transform(X_processed[['AGE']].astype(np.float64)).astype(np.float32)
        # Add age_scaler to fitted_transformers
        fitted_transformers['age_scaler'] = age_scaler
    else: # if not train
        # get the age_scaler which fitted to train dataset
        age_scaler = fitted_transformers['age_scaler']
        X_processed['AGE'] = age_scaler.transform(X_processed[['AGE']].astype(np.float64)).astype(np.float32)

    # Return X_processed, y, fitted_transformers(if train)
    if is_train: # if train dataset
        # Store the used definitive list of evidence feature names 
        fitted_transformers['evidence_feature_names'] = evidence_feature_names
        return X_processed, y, fitted_transformers
    else:
        return X_processed, y


if __name__ == "__main__":
    evidences_json_file = "Data/release_evidences.json" # Path to evidences json file

    # Generate all possible evidence feature names from the json file
    evidence_feature_names_from_json = generate_all_evidence_feature_names(evidences_json_file)

    # input paths
    train_file_in = "Data/train.csv"
    validate_file_in = "Data/validate.csv"
    test_file_in = "Data/test.csv"
    # output paths
    train_file_out = "Data/prepared_train.csv"
    validate_file_out = "Data/prepared_validate.csv"
    test_file_out = "Data/prepared_test.csv"

    print(f"\n Processing Training Data From {train_file_in}")
    # Pass the feature names derived from json files to preprocess_data
    # Get the train X and y and transformers
    X_train, y_train, transformers = preprocess_data(train_file_in, evidence_feature_names_from_json, is_train=True)
    # Save the transformers
    transformers_file_path = "Models/transformers.joblib"
    joblib.dump(transformers, transformers_file_path)
    print(f"Transformers (scalers, encoders, feature names) saved to: {transformers_file_path}")

    print("X_train shape:", X_train.shape)
    print("y_train shape: ", y_train.shape)
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    # Create a dataframe for prepared train dataset
    prepared_train_df = pd.concat([X_train, y_train], axis=1)

    print(f"Saving prepared training data to csv: {train_file_out}")
    prepared_train_df.columns = prepared_train_df.columns.astype(str)
    prepared_train_df.to_csv(train_file_out, index=False)
    print(f"Successfuly saved prepared training data to csv: {train_file_out}")


    # Retrieve the definitive list of evidence feature names from transformers for validation/test
    evidence_feature_names_for_val_test = transformers.get('evidence_feature_names')

    print(f"\n Processing Validation Data from {validate_file_in}")
    # Get validate X and y
    X_validate, y_validate = preprocess_data(validate_file_in, evidence_feature_names_for_val_test, is_train=False, fitted_transformers=transformers)
    
    print("X_validate shape:", X_validate.shape)
    print("y_validate shape:", y_validate.shape)
    X_validate = X_validate.reset_index(drop=True)
    y_validate = y_validate.reset_index(drop=True)
    # Create a dataframe for prepared validate dataset
    prepared_validate_df = pd.concat([X_validate, y_validate], axis=1)
    print(f"Attempting to save prepared validation data to csv: {validate_file_out}")
    prepared_validate_df.columns = prepared_validate_df.columns.astype(str)
    prepared_validate_df.to_csv(validate_file_out, index=False)
    print(f"Successfully saved prepared validation data to: {validate_file_out}")

    # Check if X_train's columns and X_validate's columns are consistent
    if X_train.shape[1] != X_validate.shape[1]:
        print(f"Warning: Mismatch in feature count between train ({X_train.shape[1]}) and validate ({X_validate.shape[1]})!")

    print(f"\nProcessing Test Data from {test_file_in}")
    X_test, y_test = preprocess_data(test_file_in, evidence_feature_names_for_val_test, is_train=False, fitted_transformers=transformers)
    
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    # Create a dataset for prepared test dataset
    prepared_test_df = pd.concat([X_test, y_test], axis=1)

    print(f"Attempting to save prepared test data to csv: {test_file_out}")
    prepared_test_df.columns = prepared_test_df.columns.astype(str)
    prepared_test_df.to_csv(test_file_out, index=False)
    print(f"Successfully saved prepared test data to: {test_file_out}")

    # Check if X_train's columns and X_test's columns are consistent
    if X_train.shape[1] != X_test.shape[1]:
        print(f"Warning: Mismatch in feature count between train ({X_train.shape[1]}) and test ({X_test.shape[1]})!")

    print("\nPreprocessing completed.")