---

# Diabetes Feature Engineering and Prediction

This project aims to develop a robust machine learning model to predict diabetes diagnoses based on clinical and demographic features. The workflow includes data analysis, feature engineering, and building a classification model using the Pima Indians Diabetes dataset.

## Dataset

- **Source:** National Institutes of Diabetes-Digestive-Kidney Diseases, USA  
- **Population:** Pima Indian women aged 21 and over living in Phoenix, Arizona  
- **Observations:** 768  
- **Features:**  
  - Pregnancies: Number of pregnancies  
  - Glucose: Glucose concentration  
  - BloodPressure: Diastolic blood pressure  
  - SkinThickness: Skin thickness  
  - Insulin: Insulin level  
  - BMI: Body mass index  
  - DiabetesPedigreeFunction: Genetic predisposition  
  - Age: Age in years  
  - Outcome: Diabetes test result (1 = positive, 0 = negative)

## Project Structure

- **Exploratory Data Analysis (EDA):**  
  - Data overview and variable types  
  - Analysis of categorical and numerical variables  
  - Outlier and missing value analysis  
  - Correlation analysis

- **Feature Engineering:**  
  - Handling missing and outlier values  
  - Creation of new features (e.g., BMI squared, Age categories, ratios)  
  - Encoding categorical variables (Label Encoding & One-Hot Encoding)  
  - Standardization of numerical features

- **Modeling:**  
  - Splitting data into train and test sets  
  - Training a Random Forest Classifier  
  - Evaluating model performance (accuracy, recall, precision, F1, AUC)  
  - Feature importance visualization

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

Install requirements with:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Usage

1. Set the correct path to your dataset in the script:
   ```python
   df = pd.read_csv("path_to/diabetes.csv")
   ```
2. Run the script:
   ```bash
   python Diabetes_data_analysis_and_prediction.py
   ```

## Main Steps in the Script

1. **Exploratory Data Analysis:**  
   Checks the dataset for variable types, distributions, missing values, and correlations.

2. **Feature Engineering:**  
   - Replaces invalid zeros with NaN for selected columns and fills missing values with the median.
   - Detects and handles outliers.
   - Generates new features based on domain knowledge.
   - Encodes categorical variables and standardizes numerical ones.

3. **Model Training & Evaluation:**  
   - Splits the data.
   - Trains a Random Forest model.
   - Prints evaluation metrics.
   - Plots feature importances.

## Results & Interpretation

The script prints metrics such as accuracy, recall, precision, F1 score, and AUC. It also displays the most important features contributing to the model's predictions, helping users understand which clinical factors are most relevant for diabetes prediction.

## Customization

You can adapt the script for other health datasets by changing the data loading path, feature definitions, or model type. The code is modular and easily extendable.


---
