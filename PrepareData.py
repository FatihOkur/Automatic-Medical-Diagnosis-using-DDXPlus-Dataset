import json
import pandas as pd
import numpy as np
from collections import defaultdict

# Load the evidence data
def load_evidence_data(file_path='Data/release_evidences.json'):
    with open(file_path, 'r', encoding='utf-8') as f:
        evidence_data = json.load(f)
    return evidence_data

# Load the condition data
def load_condition_data(file_path='Data/release_conditions.json'):
    with open(file_path, 'r', encoding='utf-8') as f:
        condition_data = json.load(f)
    return condition_data

# Load patient data
def load_patient_data(file_path='Data/validate.csv'):
    patient_data = pd.read_csv(file_path)
    return patient_data

def create_evidence_mappings(evidence_data):
    """Extract metadata from evidence JSON"""
    
    # Create dictionaries to store evidence info
    evidence_types = {}       # Evidence code -> data type (B, C, M)
    evidence_defaults = {}    # Evidence code -> default value
    evidence_possibles = {}   # Evidence code -> list of possible values
    evidence_is_antecedent = {} # Evidence code -> is it an antecedent?
    
    # Process each evidence
    for evidence_code, evidence_info in evidence_data.items():
        evidence_types[evidence_code] = evidence_info['data_type']
        evidence_defaults[evidence_code] = evidence_info['default_value']
        evidence_is_antecedent[evidence_code] = evidence_info['is_antecedent']
        
        # Store possible values for categorical and multi-choice
        if evidence_info['data_type'] in ['C', 'M']:
            evidence_possibles[evidence_code] = evidence_info['possible-values']
    
    return evidence_types, evidence_defaults, evidence_possibles, evidence_is_antecedent

def create_condition_mappings(condition_data):
    """Extract metadata from condition JSON"""
    
    # Create dictionaries
    condition_symptoms = {}    # Condition -> list of symptoms
    condition_antecedents = {} # Condition -> list of antecedents
    condition_severity = {}    # Condition -> severity score
    
    # Process each condition
    for condition_name, condition_info in condition_data.items():
        condition_symptoms[condition_name] = list(condition_info['symptoms'].keys())
        condition_antecedents[condition_name] = list(condition_info['antecedents'].keys())
        condition_severity[condition_name] = condition_info['severity']
    
    return condition_symptoms, condition_antecedents, condition_severity

def analyze_evidence_format(patient_data):
    """Analyze how evidence is formatted in patient data"""
    
    # Take a sample of evidences to understand format
    sample_evidences = patient_data['EVIDENCES'].iloc[0]
    
    # Convert from string to list if needed
    if isinstance(sample_evidences, str):
        sample_evidences = eval(sample_evidences.replace("'", '"'))
    
    print(f"Sample evidence list: {sample_evidences}")
    
    # Identify different formats
    binary_format = []
    categorical_format = []
    
    for evidence in sample_evidences:
        if '_@_' in evidence:
            categorical_format.append(evidence)
        else:
            binary_format.append(evidence)
    
    print(f"Binary format examples: {binary_format[:2]}")
    print(f"Categorical format examples: {categorical_format[:2]}")

def process_patient_evidences(evidences_str, evidence_types, evidence_defaults, evidence_possibles):
    """
    Process a patient's evidence list into features
    
    Args:
        evidences_str: String or list of evidence codes
        evidence_types: Dict mapping evidence code to data type (B/C/M)
        evidence_defaults: Dict mapping evidence code to default value
        evidence_possibles: Dict mapping evidence code to possible values
    
    Returns:
        Dictionary with processed features
    """
    # Parse evidences if in string format
    if isinstance(evidences_str, str):
        evidences = eval(evidences_str.replace("'", '"'))
    else:
        evidences = evidences_str
    
    # Initialize feature dictionaries
    binary_features = {}       # For binary (B) evidence
    categorical_features = {}  # For categorical (C) evidence
    multi_choice_features = defaultdict(list)  # For multi-choice (M) evidence
    
    # Process each evidence
    for evidence in evidences:
        if '_@_' in evidence:
            # Categorical or multi-choice evidence
            ev_code, ev_value = evidence.split('_@_')
            
            if evidence_types.get(ev_code) == 'C':
                # For categorical, store the value
                categorical_features[ev_code] = ev_value
            elif evidence_types.get(ev_code) == 'M':
                # For multi-choice, append to list for this evidence
                multi_choice_features[ev_code].append(ev_value)
        else:
            # Binary evidence - mark as present
            binary_features[evidence] = 1
    
    return binary_features, categorical_features, dict(multi_choice_features)

def create_feature_matrix(patient_data, evidence_data, condition_data):
    """
    Memory-efficient feature matrix creation from patient data
    
    Args:
        patient_data: DataFrame with patient records
        evidence_data: JSON data for evidences
        condition_data: JSON data for conditions
    
    Returns:
        X: Feature matrix
        y: Target labels
    """
    # Create mapping dictionaries
    evidence_types, evidence_defaults, evidence_possibles, evidence_is_antecedent = create_evidence_mappings(evidence_data)
    
    # Create demographics DataFrame with explicit dtype
    demographics_df = pd.DataFrame({
        'AGE': patient_data['AGE'].astype(np.int8)  # Use smaller int type
    })
    
    # Get dummies for SEX with memory-efficient dtype
    sex_dummies = pd.get_dummies(patient_data['SEX'], prefix='SEX', dtype=np.int8)
    
    # Initialize dictionaries for collecting features
    # Using more memory-efficient approach
    binary_features_dict = {}
    categorical_values = {}
    multi_choice_features_dict = {}
    
    # Track unique values for categoricals
    unique_categorical_values = {}
    
    # First pass - collect all unique values and column names
    print("First pass - collecting unique values...")
    for idx, patient in patient_data.iterrows():
        if idx % 100000 == 0:
            print(f"Processing row {idx}...")
            
        binary_feats, categorical_feats, multi_choice_feats = process_patient_evidences(
            patient['EVIDENCES'], evidence_types, evidence_defaults, evidence_possibles
        )
        
        # Track binary features
        for code in binary_feats:
            binary_features_dict[code] = True
        
        # Track categorical values
        for code, value in categorical_feats.items():
            if code not in unique_categorical_values:
                unique_categorical_values[code] = set()
            unique_categorical_values[code].add(value)
        
        # Track multi-choice features
        for code, values in multi_choice_feats.items():
            for value in values:
                feature_name = f'{code}_{value}'
                multi_choice_features_dict[feature_name] = True
    
    # Create empty DataFrames with correct dtypes
    print("Creating empty DataFrames with correct structure...")
    
    # Binary features
    binary_cols = sorted(binary_features_dict.keys())
    binary_df = pd.DataFrame(0, index=patient_data.index, columns=binary_cols, dtype=np.int8)
    
    # Multi-choice features
    multi_choice_cols = sorted(multi_choice_features_dict.keys())
    multi_df = pd.DataFrame(0, index=patient_data.index, columns=multi_choice_cols, dtype=np.int8)
    
    # Categorical features - prepare dummy column names
    categorical_dummy_cols = []
    for code, values in unique_categorical_values.items():
        default = evidence_defaults.get(code, '')
        for value in values:
            if value != default:  # Skip default values
                categorical_dummy_cols.append(f'{code}_{value}')
    
    categorical_df = pd.DataFrame(0, index=patient_data.index, 
                                 columns=categorical_dummy_cols, dtype=np.int8)
    
    # Second pass - fill values
    print("Second pass - filling values...")
    for idx, patient in patient_data.iterrows():
        if idx % 100000 == 0:
            print(f"Filling row {idx}...")
            
        binary_feats, categorical_feats, multi_choice_feats = process_patient_evidences(
            patient['EVIDENCES'], evidence_types, evidence_defaults, evidence_possibles
        )
        
        # Fill binary features
        for code in binary_feats:
            if code in binary_cols:
                binary_df.loc[idx, code] = 1
        
        # Fill categorical features
        for code, value in categorical_feats.items():
            # Only set if it's not the default value
            default = evidence_defaults.get(code, '')
            if value != default:
                col_name = f'{code}_{value}'
                if col_name in categorical_df.columns:
                    categorical_df.loc[idx, col_name] = 1
        
        # Fill multi-choice features
        for code, values in multi_choice_feats.items():
            for value in values:
                feature_name = f'{code}_{value}'
                if feature_name in multi_df.columns:
                    multi_df.loc[idx, feature_name] = 1
    
    # Create initial evidence dummies with memory-efficient dtype
    print("Creating initial evidence dummies...")
    initial_evidence_dummies = pd.get_dummies(
        patient_data['INITIAL_EVIDENCE'], prefix='initial', dtype=np.int8
    )
    
    # Combine feature sets
    print("Combining feature sets...")
    feature_dfs = [demographics_df, sex_dummies, binary_df, 
                  categorical_df, multi_df, initial_evidence_dummies]
    
    X = pd.concat(feature_dfs, axis=1)
    
    # No need for fillna(0) as we pre-filled everything with zeros
    
    # Target variable
    y = pd.Series(patient_data['PATHOLOGY'])
    
    return X, y

def prepare_ddxplus_data():
    """
    Main function to prepare the DDXPlus dataset and save to CSV
    
    Returns:
        X: Feature matrix
        y: Target labels
    """
    # Load data
    evidence_data = load_evidence_data('Data/release_evidences.json')
    condition_data = load_condition_data('Data/release_conditions.json')
    train_data = load_patient_data('Data/validate.csv')
    
    print("Processing dataset...")
    # Create feature matrices
    X, y = create_feature_matrix(train_data, evidence_data, condition_data)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Number of target classes: {len(y.unique())}")
    print(f"Sample features: {list(X.columns)[:5]}")
    
    # Combine features and target for saving
    print("Saving processed data to CSV file...")
    
    # Create a copy of X to avoid modifying the original
    full_data = X.copy()
    
    # Add the target column
    full_data['PATHOLOGY'] = y
    
    # Save the data
    full_data.to_csv('Data/prepared_validate.csv', index=False)
    print("Data successfully saved to 'Data/prepared_validate.csv'")
    
    return X, y


if __name__ == '__main__':
    X, y = prepare_ddxplus_data()
    print("Train Features: ", X)
    print("Train Target: ", y)