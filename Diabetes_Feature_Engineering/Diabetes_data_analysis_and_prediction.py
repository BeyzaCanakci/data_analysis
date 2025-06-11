#################################
# Diabetes Feature Engineering and Prediction
##############################

# Problem: It is desired to develop a machine learning model that can predict whether people have diabetes when their features are specified. You are expected to perform the necessary data analysis and feature engineering steps before developing the model.

# The dataset is part of a large dataset kept at the National Institutes of Diabetes-Digestive-Kidney Diseases in the USA.
# These are the data used for the diabetes research conducted on Pima Indian women aged 21 and over living in Phoenix, the 5th largest city in the State of Arizona in the USA. It consists of 768 observations and 8 numerical independent variables.
# The target variable is specified as "outcome"; 1 indicates a positive diabetes test result, 0 indicates a negative one.

# Pregnancies: Number of pregnancies
# Glucose: Glucose
# BloodPressure: Blood pressure (Diastolic)
# SkinThickness: Skin Thickness
# Insulin: Insulin.
# BMI: Body mass index.
# DiabetesPedigreeFunction: A function that calculates the probability of diabetes based on the people in our lineage.
# Age: Age (years)
# Outcome: Information about whether the person has diabetes. Has the disease (1) or not (0)

# TASK 1: EXPLORATORY DATA ANALYSIS
# Step 1: Examine the big picture.
# Step 2: Capture numerical and categorical variables.
# Step 3: Analyze numerical and categorical variables.
# Step 4: Analyze the target variable. (Average of target variable according to categorical variables, average of numeric variables according to target variable)
# Step 5: Perform outlier observation analysis.
# Step 6: Perform missing observation analysis.
# Step 7: Perform correlation analysis.

# TASK 2: FEATURE ENGINEERING
# Step 1: Perform the necessary operations for missing and outlier values. There are no missing observations in the data set, but observation units containing 0 values ​​in variables such as Glucose, Insulin, etc.
# may represent the missing value. For example; a person's glucose or insulin value
# cannot be 0. Considering this situation, you can assign zero values ​​as NaN in the relevant values ​​and then apply
# operations to the missing values.
# Step 2: Create new variables.
# Step 3: Perform encoding operations.
# Step 4: Perform standardization for numeric variables.
# Step 5: Create a model.

# Required Libraries and Functions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.simplefilter(action="ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


df = pd.read_csv("/Users/beyzacanakci/Desktop/miuul/diabetes/diabetes.csv")
df.head()
# TASK 1: EXPLORATORY DATA ANALYSIS
# Step 1: Examine the big picture.
def check_df(dataframe, head=5):
    print("##################### SHAPE #####################")
    print(dataframe.shape)
    print("##################### TYPES #####################")
    print(dataframe.dtypes)
    print("##################### HEAD #####################")
    print(dataframe.head(head))
    print("##################### TAIL #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
check_df(df)
# Step 2: Capture numerical and categorical variables.
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Returns the names of categorical, numerical, and categorical but cardinal variables in the dataset.
    Note: Categorical variables include numerically coded categorical variables as well.

    Parameters
    ------
        dataframe: dataframe
            The dataframe from which variable names are to be extracted
        cat_th: int, optional
            Class threshold for numerical but categorical variables
        car_th: int, optional
            Class threshold for categorical but cardinal variables

    Returns
    ------
        cat_cols: list
            List of categorical variables
        num_cols: list
            List of numerical variables
        cat_but_car: list
            List of categorical but cardinal variables

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))

    Notes
    ------
        cat_cols + num_cols + cat_but_car = total number of variables
        num_but_cat is included in cat_cols.
    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)  
  
# Step 3: Analyze categorical variables.
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

cat_summary(df, "Outcome")

# Step 4: Analyze numerical variables.
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

for col in num_cols:
    num_summary(df, col, plot=True)
    
# Step 5: Analyze numerical variable depend on the target variable. 
def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: ["mean", "std", "min", "max"]}), end="\n\n\n")
for col in num_cols:
    target_summary_with_num(df, "Outcome", col)   
    
#Correlation Analysis
def correlation_matrix(dataframe, plot=True):
    corr = dataframe.corr()
    if plot:
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Correlation Matrix")
        plt.show()
    return corr
corr = correlation_matrix(df)

#Set up the correlation threshold
correlation_threshold = 0.5
# Identify highly correlated features
highly_correlated_features = set()
for i in range(len(corr.columns)):
    for j in range(i):
        if abs(corr.iloc[i, j]) > correlation_threshold:
            colname = corr.columns[i]
            highly_correlated_features.add(colname)
print("Highly correlated features (correlation > 0.5):")
for feature in highly_correlated_features:
    print(feature)    
    
#Set up base model

y = df["Outcome"]
X = df.drop("Outcome", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)


print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred,y_test),3)}")
print(f"Precision: {round(precision_score(y_pred,y_test), 2)}")
print(f"F1: {round(f1_score(y_pred,y_test), 2)}")
print(f"Auc: {round(roc_auc_score(y_pred,y_test), 2)}")


def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(rf_model, X)

# TASK 2: FEATURE ENGINEERING
# Step 1: Perform the necessary operations for missing and outlier values.    

# It is known that variables other than Pregnancies and Outcome cannot have a value of 0 for a person.
# Therefore, an action should be taken regarding these values. Zero values can be assigned as NaN.
zero_columns = [col for col in df.columns if (df[col].min() == 0 and col not in ["Pregnancies", "Outcome"])]

zero_columns

# For each variable that cannot be zero in observation units, we replaced the zero values with NaN.
for col in zero_columns:
    df[col] = np.where(df[col] == 0, np.nan, df[col])

# Missing value analysis
missing_values = df.isnull().sum()
missing_values[missing_values > 0]  # Display only columns with missing values

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

na_columns = missing_values_table(df, na_name=True)

#Examine the relationship of missing values about the dependent variables

def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")


missing_vs_target(df, "Outcome", na_columns)

# Filling missing values with median
# For the variables that have missing values, we can fill them with the median of the respective columns.

for col in zero_columns:
    df.loc[df[col].isnull(), col] = df[col].median()

filled_missing_values = df.isnull().sum()
filled_missing_values[filled_missing_values > 0]  # Display only columns with missing values after filling

# ANALYZING OUTLIER and replacing them with thresholds ######

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

# replacing outliers with thresholds
for col in df.columns:
    print(col, check_outlier(df, col))
    if check_outlier(df, col):
        replace_with_thresholds(df, col)

for col in df.columns:
    print(col, check_outlier(df, col))

#Feature extraction
def create_new_features(dataframe):
    # Creating a new feature: BMI squared
    dataframe['BMI_squared'] = dataframe['BMI'] ** 2

    # Creating a new feature: Age squared
    dataframe['Age_squared'] = dataframe['Age'] ** 2

    # Creating a new feature: Glucose to Insulin ratio
    dataframe['Glucose_Insulin_Ratio'] = dataframe['Glucose'] / (dataframe['Insulin'] + 1e-5)  # Adding a small value to avoid division by zero

    # Yaş değişkenini kategorilere ayırıp yeni yaş değişkeni oluşturulması
    dataframe.loc[(dataframe["Age"] >= 21) & (dataframe["Age"] < 50), "NEW_AGE_CAT"] = "mature"
    dataframe.loc[(dataframe["Age"] >= 50), "NEW_AGE_CAT"] = "senior"

    # BMI: below 18.5 is underweight, 18.5 to 24.9 is healthy, 24.9 to 29.9 is overweight, and 30 or above is obese
    dataframe['NEW_BMI'] = pd.cut(x=dataframe['BMI'], bins=[0, 18.5, 24.9, 29.9, 100], labels=["Underweight", "Healthy", "Overweight", "Obese"])

    # Transforming the Glucose variable into a categorical variable
    # Normal: <140, Prediabetes: 140-199, Diabetes: >=200
    dataframe["NEW_GLUCOSE"] = pd.cut(x=dataframe["Glucose"], bins=[0, 140, 200, 300], labels=["Normal", "Prediabetes", "Diabetes"])

    # Creating a categorical variable by considering age and body mass index together - 3 splits identified
    dataframe.loc[(dataframe["BMI"] < 18.5) & ((dataframe["Age"] >= 21) & (dataframe["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "underweightmature"
    dataframe.loc[(dataframe["BMI"] < 18.5) & (dataframe["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "underweightsenior"
    dataframe.loc[((dataframe["BMI"] >= 18.5) & (dataframe["BMI"] < 25)) & ((dataframe["Age"] >= 21) & (dataframe["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "healthymature"
    dataframe.loc[((dataframe["BMI"] >= 18.5) & (dataframe["BMI"] < 25)) & (dataframe["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "healthysenior"
    dataframe.loc[((dataframe["BMI"] >= 25) & (dataframe["BMI"] < 30)) & ((dataframe["Age"] >= 21) & (dataframe["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "overweightmature"
    dataframe.loc[((dataframe["BMI"] >= 25) & (dataframe["BMI"] < 30)) & (dataframe["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "overweightsenior"
    dataframe.loc[(dataframe["BMI"] > 18.5) & ((dataframe["Age"] >= 21) & (dataframe["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "obesemature"
    dataframe.loc[(dataframe["BMI"] > 18.5) & (dataframe["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "obesesenior"

    # Creating a categorical variable by considering age and glucose together - 3 splits identified
    dataframe.loc[(dataframe["Glucose"] < 70) & ((dataframe["Age"] >= 21) & (dataframe["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "lowmature"
    dataframe.loc[(dataframe["Glucose"] < 70) & (dataframe["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "lowsenior"
    dataframe.loc[((dataframe["Glucose"] >= 70) & (dataframe["Glucose"] < 100)) & ((dataframe["Age"] >= 21) & (dataframe["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "normalmature"
    dataframe.loc[((dataframe["Glucose"] >= 70) & (dataframe["Glucose"] < 100)) & (dataframe["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "normalsenior"
    dataframe.loc[((dataframe["Glucose"] >= 100) & (dataframe["Glucose"] <= 125)) & ((dataframe["Age"] >= 21) & (dataframe["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "hiddenmature"
    dataframe.loc[((dataframe["Glucose"] >= 100) & (dataframe["Glucose"] <= 125)) & (dataframe["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "hiddensenior"
    dataframe.loc[(dataframe["Glucose"] > 125) & ((dataframe["Age"] >= 21) & (dataframe["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "highmature"
    dataframe.loc[(dataframe["Glucose"] > 125) & (dataframe["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "highsenior"

    # Creating a categorical variable by considering age and insulin together - 3 splits identified
    dataframe.loc[(dataframe["Insulin"] < 10) & ((dataframe["Age"] >= 21) & (dataframe["Age"] < 50)), "NEW_AGE_INSULIN_NOM"] = "lowmature_age_insulin"
    dataframe.loc[(dataframe["Insulin"] < 10) & (dataframe["Age"] >= 50), "NEW_AGE_INSULIN_NOM"] = "lowsenior_age_insulin"
    dataframe.loc[((dataframe["Insulin"] >= 10) & (dataframe["Insulin"] < 20)) & ((dataframe["Age"] >= 21) & (dataframe["Age"] < 50)), "NEW_AGE_INSULIN_NOM"] = "normalmature_age_insulin"
    dataframe.loc[((dataframe["Insulin"] >= 10) & (dataframe["Insulin"] < 20)) & (dataframe["Age"] >= 50), "NEW_AGE_INSULIN_NOM"] = "normalsenior_age_insulin"
    dataframe.loc[((dataframe["Insulin"] >= 20) & (dataframe["Insulin"] < 30)) & ((dataframe["Age"] >= 21) & (dataframe["Age"] < 50)), "NEW_AGE_INSULIN_NOM"] = "hiddenmature_age_insulin"
    dataframe.loc[((dataframe["Insulin"] >= 20) & (dataframe["Insulin"] < 30)) & (dataframe["Age"] >= 50), "NEW_AGE_INSULIN_NOM"] = "hiddensenior_age_insulin"
    dataframe.loc[(dataframe["Insulin"] >= 30) & ((dataframe["Age"] >= 21) & (dataframe["Age"] < 50)), "NEW_AGE_INSULIN_NOM"] = "highmature_age_insulin"
    dataframe.loc[(dataframe["Insulin"] >= 30) & (dataframe["Age"] >= 50), "NEW_AGE_INSULIN_NOM"] = "highsenior_age_insulin"

    # Insulin değerini kategorik hale getiren yeni değişken
    dataframe["INSULIN_SCORE"] = dataframe["Insulin"].apply(lambda x: "Normal" if 16 <= x <= 166 else "Abnormal")

    return dataframe
# Apply the feature creation function
df = create_new_features(df)

# Uppercase column names
df.columns = [col.upper() for col in df.columns]


##################################
# # Step 3: Perform encoding operations.
##################################

# Separate categorical and numerical columns after feature engineering
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# LABEL ENCODING
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]
binary_cols

for col in binary_cols:
    df = label_encoder(df, col)

# One-Hot Encoding İşlemi
# updating categorical columns after label encoding
cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["OUTCOME"]]
cat_cols

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, cat_cols, drop_first=True)

df.head()

##################################
# STANDARDIZATION FOR NUMERIC VARIABLES
##################################

num_cols

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df.head()
df.shape

##################################
# Model Creation
##################################

y = df["OUTCOME"]
X = df.drop("OUTCOME", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred,y_test),3)}")
print(f"Precision: {round(precision_score(y_pred,y_test), 2)}")
print(f"F1: {round(f1_score(y_pred,y_test), 2)}")
print(f"Auc: {round(roc_auc_score(y_pred,y_test), 2)}")


##################################
# FEATURE IMPORTANCE
##################################

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    print(feature_imp.sort_values("Value",ascending=False))
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(rf_model, X)
# The code above performs feature engineering, encoding, and model creation for diabetes prediction.
# The model is trained using a Random Forest Classifier, and feature importance is visualized.
# The final model can be used for predicting diabetes outcomes based on the engineered features.
# The code is structured to allow for easy modifications and additions, such as changing the model or adding more features.
# The feature engineering process includes creating new features based on existing ones, handling missing values, and encoding categorical variables.
# The model evaluation metrics such as accuracy, recall, precision, F1 score, and AUC are printed to assess the model's performance.
# The feature importance plot helps in understanding which features contribute most to the model's predictions.
# The code is modular, allowing for easy updates and maintenance.
# The final model can be saved and used for future predictions or further analysis.
# The code is designed to be run in a Python environment with the necessary libraries installed.
# The entire process is aimed at building a robust machine learning model for diabetes prediction, leveraging feature engineering and model evaluation techniques.
# The code is ready for execution and can be adapted for different datasets or models as needed.