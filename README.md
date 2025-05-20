# Automatic Medical Diagnosis using DDXPlus Dataset

## Project Overview
This project implements and evaluates machine learning models for automatic medical diagnosis using the DDXPlus dataset. The models aim to predict pathologies based on patient demographics, symptoms, and medical history.

---

## Motivation
Medical diagnosis is a complex task that requires deep expertise. Automated diagnostic tools can:

- Suggest potential diagnoses based on patient symptoms  
- Provide decision support in resource-limited settings  
- Reduce diagnostic delays and errors  

This project explores machine learning approaches to tackle diagnostic challenges using synthetic patient data.

---

## Dataset Description
The DDXPlus dataset includes synthetic patients generated using a proprietary medical knowledge base and a commercial rule-based diagnostic system. Each record contains:

- Demographics: Age, sex  
- Pathology: The disease the patient is suffering from  
- Evidences: Symptoms and medical history (binary, categorical, and multi-choice formats)  
- Initial Evidence: The first symptom reported by the patient  
- Differential Diagnosis: Ranked list of potential pathologies with probabilities  

The dataset includes over 40 medical conditions with varying severity and is split into train, validation, and test sets.

---

## Project Structure

```
Automatic-Medical-Diagnosis-using-DDXPlus-Dataset/
│
├── Data/
│   ├── release_evidences.json           # All possible symptoms/antecedents
│   ├── release_conditions.json          # Pathologies in the dataset
│   ├── train.csv / validate.csv / test.csv
│   ├── prepared_*.csv                   # Preprocessed data
│   └── reduced_prepared_*.csv          # Feature-selected data
│
├── Models/                              # Trained models and transformers
├── visualizations/                      # Data analysis and performance charts
│
├── PrepareData.py                       # Data preprocessing pipeline
├── FeatureSelection.py                  # Feature selection logic
├── DataAnalysis.py                      # Exploratory data analysis
├── model.py                             # Model training and saving
└── evaluate.py                          # Model evaluation and comparison
```

---

## Technical Approach

### 1. Data Preprocessing
- Parsing complex nested evidence structures  
- Encoding binary, categorical, and multi-choice evidences  
- Standardizing age  
- One-hot encoding for sex  
- Label encoding for pathologies  

Result: A structured feature matrix with demographics and medical evidence.

### 2. Feature Selection
Feature selection reduces dimensionality and enhances model performance using:

- Variance threshold filtering: Remove low-variance features (threshold = 0.0196)  
- Feature-target correlation: Keep features with correlation > 0.04  
- Multicollinearity removal: Eliminate redundant features with correlation > 0.70  

### 3. Model Implementation

- Logistic Regression:  
  Linear baseline model with 'saga' solver and multinomial classification.

- Random Forest:  
  Ensemble of 100 decision trees with max_depth=40 and class weight balancing.

- Neural Network (MLP):  
  Two hidden layers (128 and 64 units), ReLU activation, Adam optimizer, and early stopping.

### 4. Evaluation Metrics
All models are evaluated using:

- Accuracy  
- Weighted Precision, Recall, F1-score  
- Matthews Correlation Coefficient (MCC)  
- Confusion Matrices  
- Error analysis and misclassification trends

---

## Key Findings

### Model Performance
- Logistic Regression: Interpretable and effective for distinct symptom patterns  
- Random Forest: Highest accuracy and F1-score; captures complex symptom-condition relationships  
- Neural Network: Strong performance with tuned hyperparameters and early stopping

Most misclassifications occur between diseases with overlapping symptoms.

### Feature Importance
- Both demographics (age, sex) and specific symptoms strongly influence prediction  
- Feature selection effectively retains predictive features while reducing dimensionality

---

## Running the Code

### Prerequisites
- Python 3.8+  
- Required libraries: pandas, numpy, scikit-learn, matplotlib, seaborn, joblib

### Setup

```
git clone https://github.com/FatihOkur/Automatic-Medical-Diagnosis-using-DDXPlus-Dataset.git
cd Automatic-Medical-Diagnosis-using-DDXPlus-Dataset
pip install -r requirements.txt
```

Ensure the Data directory includes the following files:
- release_evidences.json  
- release_conditions.json  
- train.csv  
- validate.csv  
- test.csv

### Execution Order

```
# 1. Preprocess data
python PrepareData.py

# 2. Select informative features
python FeatureSelection.py

# 3. Exploratory data analysis
python DataAnalysis.py

# 4. Train ML models
python model.py

# 5. Evaluate and visualize results
python evaluate.py
```

---

## Interpreting Results

- Performance Metrics: Compare accuracy, precision, recall, F1, and MCC  
- Confusion Matrices: Visualize condition-level confusion  
- Error Analysis: Understand which conditions are hardest to classify  
- Visualizations: Bar charts and plots in the visualizations directory for model comparison


