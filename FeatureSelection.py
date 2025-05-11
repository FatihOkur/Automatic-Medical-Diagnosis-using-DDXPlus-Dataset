import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold

# Step 1: Process the training data to identify features to keep
print("Processing training data...")
train_df = pd.read_csv('Data/prepared_train.csv')

# Separate features and target
X_train = train_df.drop('PATHOLOGY', axis=1)
y_train = train_df['PATHOLOGY']
print(f"Original training dataset shape: {X_train.shape}")

# Remove constant or near-constant features
print("Removing low variance features...")
selector = VarianceThreshold(threshold=0.01)
selector.fit(X_train)
var_features = X_train.columns[selector.get_support()].tolist()
X_reduced = X_train[var_features]
print(f"Features after variance filtering: {len(var_features)}")

# Sample data for correlation calculation
print("Using a sample for correlation calculation...")
X_sample = X_reduced.sample(n=min(100000, len(X_reduced)), random_state=42)

# Calculate correlation matrix
print("Computing correlation matrix...")
correlation_matrix = X_sample.corr()

# Find feature pairs with high correlation
print("Finding highly correlated features...")
threshold = 0.6
features_to_remove = []

# Create a mask for the upper triangle
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

# For each pair of features in the upper triangle
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if mask[i, j] and abs(correlation_matrix.iloc[i, j]) > threshold:
            # Identify which feature to remove based on average correlation
            feat_i = correlation_matrix.columns[i]
            feat_j = correlation_matrix.columns[j]
            
            # Calculate average correlation for each feature
            avg_corr_i = correlation_matrix[feat_i].abs().mean()
            avg_corr_j = correlation_matrix[feat_j].abs().mean()
            
            # Keep the one with lower average correlation
            if avg_corr_i > avg_corr_j:
                features_to_remove.append(feat_i)
            else:
                features_to_remove.append(feat_j)

# Remove duplicates from the list of features to remove
features_to_remove = list(set(features_to_remove))
print(f"Found {len(features_to_remove)} highly correlated features to remove")

# Create final feature set
final_features = [f for f in var_features if f not in features_to_remove]
print(f"Final feature count: {len(final_features)}")

# Save the list of final features for reference
with open('Data/selected_features.txt', 'w') as f:
    for feature in final_features:
        f.write(f"{feature}\n")

# Create and save reduced training dataset
X_train_final = X_train[final_features]
train_reduced_df = pd.concat([X_train_final, y_train], axis=1)
train_reduced_df.to_csv('Data/prepared_train_reduced.csv', index=False)
print("Saved reduced training dataset.")

# Step 2: Process validation data using the same feature set
print("\nProcessing validation data...")
validate_df = pd.read_csv('Data/prepared_validate.csv')
X_validate = validate_df.drop('PATHOLOGY', axis=1)
y_validate = validate_df['PATHOLOGY']
print(f"Original validation dataset shape: {X_validate.shape}")

# Apply the same feature selection
X_validate_final = X_validate[final_features]
validate_reduced_df = pd.concat([X_validate_final, y_validate], axis=1)
validate_reduced_df.to_csv('Data/prepared_validate_reduced.csv', index=False)
print(f"Saved reduced validation dataset with {X_validate_final.shape[1]} features.")

# Step 3: Process test data using the same feature set
print("\nProcessing test data...")
test_df = pd.read_csv('Data/prepared_test.csv')
X_test = test_df.drop('PATHOLOGY', axis=1)
y_test = test_df['PATHOLOGY']
print(f"Original test dataset shape: {X_test.shape}")

# Apply the same feature selection
X_test_final = X_test[final_features]
test_reduced_df = pd.concat([X_test_final, y_test], axis=1)
test_reduced_df.to_csv('Data/prepared_test_reduced.csv', index=False)
print(f"Saved reduced test dataset with {X_test_final.shape[1]} features.")

# Optional: Visualize top correlated pairs (from training data)
print("\nCreating visualizations of correlated feature pairs...")
corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > 0.7:
            feat_i = correlation_matrix.columns[i]
            feat_j = correlation_matrix.columns[j]
            corr = abs(correlation_matrix.iloc[i, j])
            corr_pairs.append((feat_i, feat_j, corr))

# Sort by correlation strength
corr_pairs.sort(key=lambda x: x[2], reverse=True)

# Plot top 20 pairs
if corr_pairs:
    top_n = min(20, len(corr_pairs))
    plt.figure(figsize=(12, 8))
    plt.barh(
        [f"{p[0][:15]} - {p[1][:15]}" for p in corr_pairs[:top_n]],
        [p[2] for p in corr_pairs[:top_n]]
    )
    plt.title(f'Top {top_n} Correlated Feature Pairs')
    plt.xlabel('Absolute Correlation')
    plt.tight_layout()
    plt.savefig('Data/top_correlated_pairs.png')

print("Feature reduction complete for all datasets!")