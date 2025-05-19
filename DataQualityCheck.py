import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import seaborn as sns

def check_missing_values(df):
    """
    Check for missing values in the dataset
    Args:
        df: The pandas dataframe to check
    Returns:
        None (prints information)
    """
    print("\n===== CHECKING FOR MISSING VALUES =====")
    
    # Summary of missing values by column
    missing_summary = df.isnull().sum()
    print("Missing values by column:")
    print(missing_summary)
    
    # Percentage of missing values
    missing_percentage = (df.isnull().sum() / len(df)) * 100
    print("\nPercentage of missing values by column:")
    print(missing_percentage)
    
    # Number of rows with any missing values
    rows_with_missing = df.isnull().any(axis=1).sum()
    print(f"\nRows with any missing values: {rows_with_missing}")
    print(f"Total rows: {len(df)}")
    print(f"Percentage of rows with missing data: {rows_with_missing/len(df)*100:.2f}%")


def check_evidences_column(df):
    """
    Check for issues in the EVIDENCES column
    Args:
        df: The pandas dataframe to check
    Returns:
        None (prints information)
    """
    print("\n===== CHECKING EVIDENCES COLUMN =====")
    
    # Check for empty evidence lists
    empty_evidences = df[df['EVIDENCES'] == '[]'].shape[0]
    print(f"Rows with empty EVIDENCES list: {empty_evidences}")
    print(f"Percentage of rows with empty EVIDENCES: {empty_evidences/len(df)*100:.2f}%")
    
    # Check for invalid formatted EVIDENCES
    invalid_format = 0
    invalid_rows = []
    
    for idx, evidence_str in enumerate(df['EVIDENCES']):
        try:
            # Try to convert string to list
            if pd.isna(evidence_str):
                invalid_format += 1
                invalid_rows.append(idx)
                continue
                
            evidence_list = ast.literal_eval(evidence_str)
            if not isinstance(evidence_list, list):
                invalid_format += 1
                invalid_rows.append(idx)
                print(f"Row {idx}: Non-list value: {evidence_str}")
        except:
            invalid_format += 1
            invalid_rows.append(idx)
            print(f"Row {idx}: Unparseable value: {evidence_str}")
    
    print(f"Number of rows with invalid EVIDENCES format: {invalid_format}")
    print(f"Percentage of rows with invalid EVIDENCES: {invalid_format/len(df)*100:.2f}%")
    
    # Check evidence counts
    if invalid_format < len(df):  # Only proceed if not all rows are invalid
        df['evidence_count'] = df['EVIDENCES'].apply(
            lambda x: len(ast.literal_eval(x)) if not pd.isna(x) and x != '[]' else 0
        )
        
        print("\nStatistics for evidence counts per patient:")
        print(df['evidence_count'].describe())
        
        # Check for unusually high evidence counts (possible errors)
        q3 = df['evidence_count'].quantile(0.75)
        upper_bound = q3 + 1.5 * (q3 - df['evidence_count'].quantile(0.25))
        outliers = df[df['evidence_count'] > upper_bound]
        print(f"\nRows with unusually high evidence counts (>{upper_bound}): {len(outliers)}")
    
    return invalid_rows


def visualize_data_distributions(df, invalid_rows=None):
    """
    Create visualizations to identify anomalies in the data
    Args:
        df: The pandas dataframe to visualize
        invalid_rows: List of row indices to exclude from visualization
    Returns:
        None (shows plots)
    """
    print("\n===== CREATING DATA DISTRIBUTION VISUALIZATIONS =====")
    
    # Create a filtered dataframe for visualization if needed
    if invalid_rows and len(invalid_rows) > 0:
        vis_df = df.drop(invalid_rows).copy()
        print(f"Visualizing with {len(vis_df)} rows (excluded {len(invalid_rows)} invalid rows)")
    else:
        vis_df = df.copy()

    # 1. Age distribution visualization
    plt.figure(figsize=(10, 6))
    sns.histplot(vis_df['AGE'], kde=True)
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.3)
    plt.savefig('age_distribution.png')
    plt.close()
    print("Saved age distribution plot: age_distribution.png")
    
    # 2. Sex distribution pie chart
    plt.figure(figsize=(8, 8))
    vis_df['SEX'].value_counts().plot.pie(autopct='%1.1f%%')
    plt.title('Sex Distribution')
    plt.ylabel('')
    plt.savefig('sex_distribution.png')
    plt.close()
    print("Saved sex distribution plot: sex_distribution.png")
    
    # 3. Top pathologies bar chart
    plt.figure(figsize=(14, 8))
    vis_df['PATHOLOGY'].value_counts().head(20).plot(kind='bar')
    plt.title('Top 20 Pathologies')
    plt.xlabel('Pathology')
    plt.ylabel('Frequency')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('top_pathologies.png')
    plt.close()
    print("Saved top pathologies plot: top_pathologies.png")
    
    # 4. Evidence count distribution
    if 'evidence_count' not in vis_df.columns:
        vis_df['evidence_count'] = vis_df['EVIDENCES'].apply(
            lambda x: len(ast.literal_eval(x)) if not pd.isna(x) and x != '[]' else 0
        )
    
    plt.figure(figsize=(10, 6))
    sns.histplot(vis_df['evidence_count'], kde=True)
    plt.title('Distribution of Evidence Count per Patient')
    plt.xlabel('Number of Evidences')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.3)
    plt.savefig('evidence_count_distribution.png')
    plt.close()
    print("Saved evidence count distribution plot: evidence_count_distribution.png")
    
    # 5. Age vs Evidence Count scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(vis_df['AGE'], vis_df['evidence_count'], alpha=0.5)
    plt.title('Age vs. Evidence Count')
    plt.xlabel('Age')
    plt.ylabel('Number of Evidences')
    plt.grid(True, alpha=0.3)
    plt.savefig('age_vs_evidence_count.png')
    plt.close()
    print("Saved age vs evidence count plot: age_vs_evidence_count.png")
    
    # 6. Correlation heatmap (if there are other numerical columns)
    numeric_cols = vis_df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > 1:
        plt.figure(figsize=(10, 8))
        sns.heatmap(vis_df[numeric_cols].corr(), annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Heatmap of Numerical Variables')
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png')
        plt.close()
        print("Saved correlation heatmap: correlation_heatmap.png")


def main():
    """
    Main function to check the quality of the DDXPlus dataset
    """
    print("Starting data quality check for DDXPlus dataset...")
    
    # Load the dataset
    try:
        print("Loading dataset from 'Data/train.csv'...")
        df = pd.read_csv("Data/train.csv")
        print(f"Successfully loaded dataset with {len(df)} rows and {len(df.columns)} columns")
        print(f"Column names: {df.columns.tolist()}")
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return
    
    # Check for missing values
    check_missing_values(df)
    
    # Check the EVIDENCES column
    invalid_rows = check_evidences_column(df)
    
    # Create visualizations
    visualize_data_distributions(df, invalid_rows)
    
    print("\nData quality check completed!")


if __name__ == "__main__":
    main()