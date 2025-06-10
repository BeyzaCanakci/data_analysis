"""
---

# BG-NBD and Gamma-Gamma CLTV Prediction

\##############################################################

# Business Problem

\###############################################################
FLO wants to define a roadmap for its sales and marketing activities.
In order for the company to make medium- to long-term plans, it needs to estimate the potential value that existing customers will bring in the future.

\###############################################################

# Dataset Story

\###############################################################

The dataset consists of information obtained from the past shopping behaviors of OmniChannel customers (those who shop both online and offline) who made their last purchases in 2020–2021.

* **master\_id**: Unique customer ID
* **order\_channel**: Channel used for shopping (Android, iOS, Desktop, Mobile, Offline)
* **last\_order\_channel**: Channel used in the last purchase
* **first\_order\_date**: Date of the customer's first purchase
* **last\_order\_date**: Date of the customer's most recent purchase
* **last\_order\_date\_online**: Date of the customer's most recent online purchase
* **last\_order\_date\_offline**: Date of the customer's most recent offline purchase
* **order\_num\_total\_ever\_online**: Total number of online purchases
* **order\_num\_total\_ever\_offline**: Total number of offline purchases
* **customer\_value\_total\_ever\_offline**: Total amount paid in offline purchases
* **customer\_value\_total\_ever\_online**: Total amount paid in online purchases
* **interested\_in\_categories\_12**: List of categories the customer shopped in over the past 12 months

---

# TASKS

\###############################################################

### TASK 1: Data Preparation

1. Read the file `flo_data_20K.csv` and create a copy of the dataframe.
2. Define the functions `outlier_thresholds` and `replace_with_thresholds` to suppress outliers.
   **Note:** Since `frequency` values must be integers when calculating CLTV, round the lower and upper limits using `round()`.
3. Suppress outliers, if any, for the following variables:

   * `order_num_total_ever_online`
   * `order_num_total_ever_offline`
   * `customer_value_total_ever_offline`
   * `customer_value_total_ever_online`
4. Omnichannel customers are defined as those who shop on both online and offline platforms.
   Create new variables for each customer’s **total number of purchases** and **total spending**.
5. Examine variable types. Convert variables that represent dates into `datetime` format.

---

### TASK 2: Creating the CLTV Data Structure

1. Take **two days after the latest purchase date** in the dataset as the **analysis date**.
2. Create a new CLTV dataframe containing the following variables:

   * `customer_id`
   * `recency_cltv_weekly`
   * `T_weekly`
   * `frequency`
   * `monetary_cltv_avg`

> The monetary value should represent the **average value per purchase**.
> The recency and tenure values should be expressed in **weeks**.

---
"""
# 1. OmniChannel.csv verisini okuyunuz.Dataframe’in kopyasını oluşturunuz.
import pandas as pd
import datetime as dt
from lifetimes import BetaGeoFitter, GammaGammaFitter

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df
file_path = '/Users/beyza/Desktop/miuul/FLOCLTVPrediction/flo_data_20k.csv'
df_ = load_data(file_path)
df = df_.copy()
df.head()
# 2. Define the outlier_thresholds and replace_with_thresholds functions required to suppress outliers.
# Note: When calculating cltv, frequency values ​​must be integer. Therefore, round the lower and upper limits with round().

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = round(low_limit,0)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit, 0)
    


df.describe().T
df.head()
df.isnull().sum()


# 3. If there are any outliers in the variables "order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online"
#suppress them

columns = ["order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline","customer_value_total_ever_online"]
for col in columns:
    replace_with_thresholds(df, col)
    
# 4. Omnichannel refers to customers shopping from both online and offline platforms.
# Create new variables for each customer's total number of purchases and spending.
df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

# 5. Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.
def preprocess_data(df):
    df['total_order_date'] = pd.to_datetime(df[['last_order_date_online', 'last_order_date_offline']].max(axis=1))
    df['first_order_date'] = pd.to_datetime(df['first_order_date'])
    df['last_order_date'] = pd.to_datetime(df['last_order_date'])
    df['last_order_date_online'] = pd.to_datetime(df['last_order_date_online'])
    df['last_order_date_offline'] = pd.to_datetime(df['last_order_date_offline'])
    df['order_num_total'] = df['order_num_total_ever_online'] + df['order_num_total_ever_offline']
    df['customer_value_total'] = df['customer_value_total_ever_online'] + df['customer_value_total_ever_offline']
    return df

df = preprocess_data(df)

###############################################################
# TASK 2: Creating CLTV Data Structure
# ###############################################################

# 1. Take the date 2 days after the last purchase in the data set as the analysis date.
# 2. Create a new cltv dataframe containing the values ​​of customer_id, recency_cltv_weekly, T_weekly, frequency and monetary_cltv_avg.
def create_cltv_df(df):
    today_date = df['last_order_date'].max() + dt.timedelta(days=2)
    df['recency_cltv_weekly'] = (df['total_order_date'] - df['first_order_date']).dt.days / 7
    df['T_weekly'] = (today_date - df['first_order_date']).dt.days / 7
    df['frequency'] = df['order_num_total']
    df['monetary_cltv_avg'] = df['customer_value_total'] / df['order_num_total']
    df = df[df['frequency'] > 1]
    cltv_df = df[['master_id', 'recency_cltv_weekly', 'T_weekly', 'frequency', 'monetary_cltv_avg']]
    return cltv_df
cltv_df = create_cltv_df(df)
cltv_df.head()
cltv_df[['frequency', 'recency_cltv_weekly', 'T_weekly']].describe()

###############################################################
# TASK 3: Setting up BG/NBD, Gamma-Gamma Models, Calculating 6-month CLTV
# ###############################################################

# 1. Set up the BG/NBD model.
# Estimate the expected purchases from customers in 3 months and add it to the cltv dataframe as exp_sales_3_month.
# Estimate the expected purchases from customers in 6 months and add it to the cltv dataframe as exp_sales_6_month.
# Examine the 10 people who will make the most purchases in the 3rd and 6th months.
# 2. Fit the Gamma-Gamma model. Estimate the average value that customers will leave and add it to the cltv dataframe as exp_average_value.
# 3. Calculate the 6-month CLTV and add it to the dataframe as cltv.
# Observe the 20 people with the highest CLTV value.
def fit_cltv_model(cltv_df):
    from lifetimes import BetaGeoFitter, GammaGammaFitter
    import pandas as pd

    print("Fitting BG/NBD model...")
    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(cltv_df['frequency'], cltv_df['recency_cltv_weekly'], cltv_df['T_weekly'])

    # Tahmini satış sayısı: 3 ay (12 hafta)
    cltv_df['exp_sales_3_month'] = bgf.predict(12, cltv_df['frequency'], cltv_df['recency_cltv_weekly'], cltv_df['T_weekly'])
    print("\n Expected Sales in 3 Months (First 10):")
    print(cltv_df[['exp_sales_3_month']].sort_values(by='exp_sales_3_month', ascending=False).head(10))

    # Tahmini satış sayısı: 6 ay (24 hafta)
    cltv_df['exp_sales_6_month'] = bgf.predict(24, cltv_df['frequency'], cltv_df['recency_cltv_weekly'], cltv_df['T_weekly'])
    print("\n Expected Sales in 6 Months (First 10):")
    print(cltv_df[['exp_sales_6_month']].sort_values(by='exp_sales_6_month', ascending=False).head(10))

    # Gamma-Gamma modelini uygula
    print("Fitting Gamma-Gamma model...")
    ggf = GammaGammaFitter()
    ggf.fit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])

    # Ortalama beklenen kazanç
    cltv_df['exp_average_value'] = ggf.conditional_expected_average_profit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])
    print("\n Expected Average Order Value (First 10):")
    print(cltv_df[['exp_average_value']].sort_values(by='exp_average_value', ascending=False).head(10))

    # CLTV hesaplama (6 aylık periyot, haftalık frekansla)
    cltv_df['cltv'] = ggf.customer_lifetime_value(
        bgf,
        cltv_df['frequency'],
        cltv_df['recency_cltv_weekly'],
        cltv_df['T_weekly'],
        cltv_df['monetary_cltv_avg'],
        time=6,  # months
        freq='W',  # weekly
        discount_rate=0.01
    )
    print("\n Calculated CLTV (First 10):")
    print(cltv_df[['cltv']].sort_values(by='cltv', ascending=False).head(10))

    # En çok alışveriş yapan kullanıcılar
    print("\n Top Frequency (First 10):")
    print(cltv_df[['frequency']].sort_values(by='frequency', ascending=False).head(10))

    # Segmentasyon
    cltv_df['cltv_segment'] = pd.qcut(cltv_df['cltv'], 4, labels=["D", "C", "B", "A"])
    print("\n Segment counts by CLTV quantiles:")
    print(cltv_df['cltv_segment'].value_counts())

    return cltv_df
cltv_df = create_cltv_df(df)
# Fonksiyon tanımlandıktan sonra, uygun veriyi verip çağır:
cltv_df_result = fit_cltv_model(cltv_df)

###############################################################
# TASK 4: Creating Segments Based on CLTV
########################################################

# 1. Separate all your customers into 4 groups (segments) based on 6-month CLTV and add the group names to the dataset.
# Assign it with the name cltv_segment.
# 2. Examine the recency, frequency and monetary averages of the segments.
# Get the statistical summary of the segments
segment_summary = cltv_df_result.groupby("cltv_segment").agg({
    "recency_cltv_weekly": "mean",
    "T_weekly": "mean",
    "frequency": "mean",
    "monetary_cltv_avg": "mean",
    "cltv": "mean"
}).round(2)

print(segment_summary)
