import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import ast
from collections import Counter

# Set up plot styles
plt.style.use('ggplot')
sns.set(font_scale=1.2)

# Load data files
train_df = pd.read_csv("Data/train.csv")
validate_df = pd.read_csv("Data/validate.csv")
test_df = pd.read_csv("Data/test.csv")

# Load evidence and condition definitions
with open("Data/release_evidences.json", 'r') as f:
    evidences_data = json.load(f)

with open("Data/release_conditions.json", 'r') as f:
    conditions_data = json.load(f)

# Function to parse evidences list
def parse_evidences(evidence_str):
    return ast.literal_eval(evidence_str)

# Apply function to create parsed evidences column
train_df['EVIDENCES_PARSED'] = train_df['EVIDENCES'].apply(parse_evidences)

# 1. Basic Dataset Information
print("--- Basic Dataset Information ---")
print(f"Total samples: {len(train_df) + len(validate_df) + len(test_df)}")
print(f"Training samples: {len(train_df)}")
print(f"Validation samples: {len(validate_df)}")
print(f"Test samples: {len(test_df)}")
print(f"Number of pathologies: {train_df['PATHOLOGY'].nunique()}")
print(f"Number of evidence types: {len(evidences_data)}")

# Display dataset columns and data types
print("\nDataset structure:")
print(train_df.dtypes)

# Check for missing values
print("\n--- Missing Values Analysis ---")
print("Missing values in train dataset:")
print(train_df.isnull().sum())

print("Missing values in train dataset:")
print(validate_df.isnull().sum())

print("Missing values in train dataset:")
print(test_df.isnull().sum())

# 2. Target Variable Analysis
pathology_counts = train_df['PATHOLOGY'].value_counts()

# Plot pathology distribution
plt.figure(figsize=(12, 6))
pathology_counts.head(15).plot(kind='bar')
plt.title('Top 15 Pathologies in Training Dataset')
plt.ylabel('Count')
plt.xlabel('Pathology')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('visualizations/top_pathologies.png')
plt.close()

# 3. Demographic Analysis
print("\n--- Demographic Statistics ---")
print("Age Statistics:")
print(train_df['AGE'].describe())

print("\nSex Distribution:")
sex_dist = train_df['SEX'].value_counts(normalize=True) * 100
print(sex_dist)

# Plot age distribution
plt.figure(figsize=(10, 6))
sns.histplot(train_df['AGE'], bins=20, kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.savefig('visualizations/age_distribution.png')
plt.close()

# Plot sex distribution
plt.figure(figsize=(8, 6))
train_df['SEX'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Sex Distribution')
plt.ylabel('')
plt.savefig('visualizations/sex_distribution.png')
plt.close()

# 4. Evidence Analysis
# Count occurrences of each evidence type
evidence_counts = Counter()
for evidences in train_df['EVIDENCES_PARSED']:
    for evidence in evidences:
        # For binary evidences
        if '@_' not in evidence:
            evidence_counts[evidence] += 1
        # For categorical/multi-choice evidences
        else:
            base_evidence = evidence.split('@_')[0]
            evidence_counts[base_evidence] += 1

# Plot top evidences
plt.figure(figsize=(12, 8))
top_evidences = dict(evidence_counts.most_common(15))
sns.barplot(x=list(top_evidences.keys()), y=list(top_evidences.values()))
plt.title('Top 15 Most Common Evidences')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('visualizations/top_evidences.png')
plt.close()

# 5. Relationship between Age, Sex and Pathologies
# Create age groups
train_df['AGE_GROUP'] = pd.cut(train_df['AGE'], bins=[0, 18, 40, 65, 100], 
                               labels=['Child', 'Young Adult', 'Adult', 'Senior'])

# Plot pathology distribution by age group for top pathologies
top_pathologies = pathology_counts.head(5).index.tolist()
plt.figure(figsize=(14, 8))
for i, pathology in enumerate(top_pathologies):
    subset = train_df[train_df['PATHOLOGY'] == pathology]
    plt.subplot(2, 3, i+1)
    sns.countplot(x='AGE_GROUP', data=subset)
    plt.title(f'{pathology}')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('visualizations/pathology_by_age.png')
plt.close()

# Load processed training data
print("\n--- Features Before Selection ---")
processed_train_df = pd.read_csv("Data/prepared_train.csv")
num_features_after_preprocessing = processed_train_df.shape[1] - 1  # -1 for the target column
print(f"Number of features after preprocessing (before feature selection): {num_features_after_preprocessing}")

# Print number of features after feature selection
print("\n--- Features After Selection ---")
# Count features after feature selection
reduced_train_df = pd.read_csv("Data/reduced_prepared_train.csv")
num_features_after_selection = reduced_train_df.shape[1] - 1  # -1 for the target column
print(f"Number of features after feature selection: {num_features_after_selection}")


# 6. Find correlations between features and the target condition
print("\n--- Feature-Target Correlation Analysis ---")

# Get feature columns (all except target)
feature_cols = processed_train_df.drop('PATHOLOGY_ENCODED', axis=1).columns

# Calculate correlation with target
target_correlations = processed_train_df[feature_cols].corrwith(processed_train_df['PATHOLOGY_ENCODED'])

# Convert to absolute values and sort
abs_correlations = target_correlations.abs().sort_values(ascending=False)

# Display top correlated features
print("\nTop 15 features with highest correlation to pathology:")
for feature, corr in abs_correlations.head(15).items():
    print(f"{feature}: {corr:.4f}")

# Visualize top correlated features
plt.figure(figsize=(12, 8))
top_features = abs_correlations.head(15).index
top_corrs = abs_correlations.head(15).values

# Create horizontal bar chart
sns.barplot(x=top_corrs, y=top_features)
plt.title('Top 15 Features Correlated with Pathology')
plt.xlabel('Absolute Correlation Value')
plt.tight_layout()
plt.savefig('visualizations/top_correlated_features.png')
plt.close()

# Create correlation matrix for top features and target
top_features_list = list(top_features)
top_features_list.append('PATHOLOGY_ENCODED')  # Add target

# Get correlation matrix for top features and target
correlation_matrix = processed_train_df[top_features_list].corr()

# Plot correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', 
            fmt=".2f", linewidths=0.5, vmin=-1, vmax=1)
plt.title('Correlation Matrix: Top Features vs Target')
plt.tight_layout()
plt.savefig('visualizations/feature_target_correlation_matrix.png')
plt.close()

# Print summary insights
print("\n--- Key Insights from Correlation Analysis ---")
print(f"1. The feature most strongly correlated with pathology is '{abs_correlations.index[0]}' (correlation: {abs_correlations.iloc[0]:.4f})")
print(f"2. There are {len(abs_correlations[abs_correlations > 0.2])} features with correlation > 0.2 with the target")
print(f"3. The average correlation strength of the top 15 features is {abs_correlations.head(15).mean():.4f}")

# 7. t-distributed Stochastic Neighbor Embedding (t-SNE) visualization
print("\n--- t-SNE Visualization of Patient Features ---")
from sklearn.manifold import TSNE

# Use processed training data for t-SNE visualization
X = processed_train_df.drop('PATHOLOGY_ENCODED', axis=1).values
y = processed_train_df['PATHOLOGY_ENCODED'].values

# Sample data if it's too large (t-SNE is computationally expensive)
if len(X) > 5000:
    print(f"Sampling 5000 records from {len(X)} for t-SNE visualization")
    from sklearn.model_selection import train_test_split
    X_sample, _, y_sample, _ = train_test_split(
        X, y, train_size=5000, random_state=42, stratify=y
    )
else:
    X_sample = X
    y_sample = y

print(f"Running t-SNE on {len(X_sample)} samples...")
# Apply t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_sample)
print("t-SNE transformation complete")

# Create scatter plot of t-SNE results
plt.figure(figsize=(12, 10))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_sample, cmap='viridis', alpha=0.7)

# Add a colorbar legend
legend1 = plt.colorbar(scatter)
legend1.set_label('Pathology Class')

plt.title("t-SNE Projection of Patient Features")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.tight_layout()
plt.savefig('visualizations/tsne_visualization.png')
plt.close()
print("t-SNE visualization saved to visualizations/tsne_visualization.png")

# 8. Feature Type Distribution Analysis
print("\n--- Feature Type Distribution Analysis ---")

# Count feature types
feature_types = {"Binary": 0, "Categorical": 0, "Multi-choice": 0}

# Process each evidence and count by type
for evidence_code, evidence_details in evidences_data.items():
    data_type = evidence_details.get('data_type')
    
    if data_type == 'B':
        feature_types["Binary"] += 1
    elif data_type == 'C':
        feature_types["Categorical"] += 1
    elif data_type == 'M':
        feature_types["Multi-choice"] += 1
    else:
        print(f"Warning: Unknown data_type '{data_type}' for evidence {evidence_details.get('name')}")

# Print summary
total_features = sum(feature_types.values())
print(f"Total features: {total_features}")
for feature_type, count in feature_types.items():
    percentage = (count / total_features) * 100
    print(f"- {feature_type}: {count} ({percentage:.1f}%)")

# Create pie chart of feature types
plt.figure(figsize=(8, 6))
colors = ['#ff9999','#66b3ff','#99ff99']
plt.pie(feature_types.values(), labels=feature_types.keys(), 
        autopct='%1.1f%%', startangle=90, colors=colors)
plt.axis('equal')
plt.title('Feature Types Distribution')
plt.tight_layout()
plt.savefig('visualizations/feature_types_distribution.png')
plt.close()
print("Feature type distribution visualization saved to 'visualizations/feature_types_distribution.png'")

# Output summary of findings
print("\n--- Key Insights from Data Exploration ---")
print(f"1. The dataset contains {len(train_df) + len(validate_df) + len(test_df)} patients with {train_df['PATHOLOGY'].nunique()} different pathologies.")
print(f"2. Most common pathology: {pathology_counts.index[0]} ({pathology_counts.iloc[0]} patients)")
print(f"3. Mean patient age: {train_df['AGE'].mean():.1f} years")
print(f"4. Sex distribution: {sex_dist.iloc[0]:.1f}% {sex_dist.index[0]}, {sex_dist.iloc[1]:.1f}% {sex_dist.index[1]}")
print(f"5. Most common evidence: {list(top_evidences.keys())[0]} (present in {list(top_evidences.values())[0]} patients)")