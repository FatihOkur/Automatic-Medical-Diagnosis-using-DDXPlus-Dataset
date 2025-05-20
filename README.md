# Automatic Medical Diagnosis using DDXPlus Dataset

## Project Overview
This project implements and evaluates machine learning models for automatic medical diagnosis using the DDXPlus dataset. The models predict pathologies based on patient demographics, symptoms, and medical history.

## Motivation
Medical diagnosis is a complex process that requires extensive knowledge and experience. Automated diagnostic tools can assist healthcare providers by:

- Suggesting potential diagnoses based on patient symptoms  
- Providing decision support in resource-limited settings  
- Reducing diagnostic errors and delays  
- Prioritizing cases based on severity  

This project explores the capabilities of various machine learning approaches to tackle this challenging problem using synthetic patient data from the DDXPlus dataset.

## Dataset Description
The DDXPlus dataset contains synthetic patients generated using a proprietary medical knowledge base and a commercial rule-based diagnostic system. Each patient record includes:

- Demographics: Age, sex  
- Pathology: The disease the patient is suffering from  
- Evidences: Symptoms and medical antecedents in binary, categorical, and multi-choice formats  
- Initial Evidence: The first symptom reported by the patient  
- Differential Diagnosis: Ranked list of potential pathologies with probabilities  

The dataset includes over 40 different medical conditions with varying severity levels and is split into training, validation, and test sets.

## Project Structure

**Data/**: Contains raw and processed data files  
- release_evidences.json: Defines all possible symptoms/antecedents  
- release_conditions.json: Defines all pathologies in the dataset  
- train.csv, validate.csv, test.csv: Raw data files  
- prepared_*.csv: Preprocessed data files  
- reduced_prepared_*.csv: Feature-selected data files  

**Models/**: Saved trained models and transformers  
**visualizations/**: Generated data visualizations and analysis charts  

**Python Scripts:**  
- PrepareData.py: Data preprocessing pipeline  
- FeatureSelection.py: Feature selection implementation  
- model.py: Model training and evaluation  
- evaluate.py: Comprehensive model evaluation and comparison including feature importance analysis  
- DataAnalysis.py: Exploratory data analysis and visualization  

## Technical Approach

### 1. Data Preprocessing
Our preprocessing pipeline handles:

- Parsing complex nested evidence structures  
- Encoding binary, categorical, and multi-choice medical evidences  
- Feature standardization for age  
- One-hot encoding for categorical variables like sex  
- Target variable encoding using LabelEncoder  

The preprocessing converts raw patient data into a structured feature matrix where each column represents either a demographic feature or a specific medical evidence.

### 2. Feature Selection
To improve model performance and reduce dimensionality, we implemented feature selection based on:

- Variance threshold filtering (removing near-constant features with threshold of 0.0196)  
- Feature-target correlation analysis (keeping features with correlation threshold greater than 0.04)  
- Multicollinearity removal (eliminating redundant features with correlation greater than 0.70)  

This process significantly reduces the feature space while maintaining predictive power.

### 3. Model Implementation
We trained and evaluated three different classification models:

- Logistic Regression: A linear model implemented with the saga solver and multinomial classification approach. Though simple, it provides a good baseline and interpretable coefficients.  
- Random Forest: An ensemble of 100 decision trees with controlled depth (max depth 40) and balanced class weights. This model captures non-linear relationships between symptoms and diagnoses.  
- Neural Network (MLP): A multi-layer perceptron with two hidden layers (128 and 64 neurons) using ReLU activation and the Adam optimizer. The model uses early stopping based on validation performance.  

### 4. Evaluation
Models were evaluated using multiple performance metrics:

- Accuracy  
- Precision (weighted)  
- Recall (weighted)  
- F1-score (weighted)  
- Matthews Correlation Coefficient (MCC)  
- Confusion matrices  

Additionally, we perform feature importance analysis to understand which features contribute most to each model's predictions:

- For Logistic Regression: Mean absolute coefficients across classes  
- For Random Forest: Built-in feature importance based on Gini impurity reduction  
- For Neural Network: Permutation importance (measuring performance drop when features are shuffled)  

The evaluation includes detailed analysis of error patterns, highlighting common misclassifications and challenging pathologies.

## Key Findings

### Model Performance
The models show varying performance characteristics:

- Logistic Regression: Provides a solid baseline with interpretable coefficients. Performs well on conditions with distinct symptom patterns.  
- Random Forest: Generally achieves the highest overall accuracy and F1-score. Excellent at capturing complex relationships between symptoms and conditions.  
- Neural Network: Shows competitive performance with properly tuned hyperparameters, particularly on conditions with complex symptom interactions.  

The confusion matrices reveal that most misclassifications occur between similar conditions with overlapping symptoms.

### Feature Importance
Analysis reveals that both demographic features (age, sex) and specific evidences contribute significantly to diagnostic accuracy. The feature selection process identifies the most predictive symptoms for various conditions.

## Running the Code

### Prerequisites
- Python 3.8+  
- Required libraries: pandas, numpy, scikit-learn, matplotlib, seaborn, joblib  

### Setup

Clone the repository:
```
git clone https://github.com/FatihOkur/Automatic-Medical-Diagnosis-using-DDXPlus-Dataset.git
cd Automatic-Medical-Diagnosis-using-DDXPlus-Dataset
```

Install dependencies:
```
pip install -r requirements.txt
```

Ensure the Data directory contains the DDXPlus dataset files:
- release_evidences.json  
- release_conditions.json  
- train.csv  
- validate.csv  
- test.csv  

### Execution

Run the following scripts in sequence:

- **Data preprocessing**  
```
python PrepareData.py
```
Generates the preprocessed datasets in the Data directory.

- **Feature selection**  
```
python FeatureSelection.py
```
Creates reduced feature datasets using the selection criteria.

- **Exploratory data analysis**  
```
python DataAnalysis.py
```
Creates data visualizations in the visualizations directory.

- **Model training**  
```
python model.py
```
Trains the three models and saves them to the Models directory.

- **Model evaluation and feature importance analysis**  
```
python evaluate.py
```
Evaluates models, generates performance comparisons, and analyzes feature importance.

## Interpreting Results
The evaluation outputs provide several key insights:

- **Performance Metrics**: Compare accuracy, precision, recall, F1-score, and MCC across models to understand overall effectiveness.  
- **Confusion Matrices**: Visualize which conditions are commonly confused with others, revealing diagnostic challenges.  
- **Feature Importance**: Analyze which symptoms and demographic factors most strongly influence diagnosis predictions for each model, providing clinical interpretability.  
- **Error Analysis**: Identify the top misclassified pathologies, where the models struggle most, often with conditions sharing similar symptom profiles.  
- **Visualization Charts**: Bar charts and other visualizations help compare model performance across different metrics and pathologies.  
