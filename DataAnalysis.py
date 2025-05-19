import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import ast
from collections import Counter
import os

# Set up plot styles
plt.style.use('ggplot')
sns.set(font_scale=1.2)

# Create output directory for visualizations
os.makedirs('visualizations', exist_ok=True)

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

# 6. Find correlations between features and conditions
# This requires accessing the processed features after one-hot encoding

# Output summary of findings
print("\n--- Key Insights from Data Exploration ---")
print(f"1. The dataset contains {len(train_df) + len(validate_df) + len(test_df)} patients with {train_df['PATHOLOGY'].nunique()} different pathologies.")
print(f"2. Most common pathology: {pathology_counts.index[0]} ({pathology_counts.iloc[0]} patients)")
print(f"3. Mean patient age: {train_df['AGE'].mean():.1f} years")
print(f"4. Sex distribution: {sex_dist.iloc[0]:.1f}% {sex_dist.index[0]}, {sex_dist.iloc[1]:.1f}% {sex_dist.index[1]}")
print(f"5. Most common evidence: {list(top_evidences.keys())[0]} (present in {list(top_evidences.values())[0]} patients)")