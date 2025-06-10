
---

# FLO CLTV Prediction

This project predicts the potential future value (Customer Lifetime Value, CLTV) that existing customers may bring to FLO using BG-NBD and Gamma-Gamma models.

## Contents

- [About the Project](#about-the-project)
- [Dataset](#dataset)
- [Models Used](#models-used)
- [Installation and Usage](#installation-and-usage)
- [Outputs](#outputs)
- [References](#references)

## About the Project

FLO aims to develop a roadmap for its sales and marketing activities. To make medium- and long-term plans, the company needs to estimate the value that current customers will provide in the future by analyzing their past shopping behavior.

## Dataset

The dataset consists of information from OmniChannel customers who shopped at FLO in 2020–2021. Key fields include:

- master_id: Unique customer ID
- order_channel: Shopping channel (Android, iOS, Desktop, Mobile, Offline)
- first_order_date, last_order_date: Dates of the first and most recent purchase
- order_num_total_ever_online / offline: Total number of online/offline purchases
- customer_value_total_ever_online / offline: Total amount spent online/offline
- interested_in_categories_12: Categories shopped in over the past 12 months

## Models Used

- BG-NBD (Beta Geometric/Negative Binomial Distribution): Predicts customer purchase frequency.
- Gamma-Gamma: Predicts average transaction value.
- CLTV (Customer Lifetime Value): Calculates a 6-month customer lifetime value.

## Installation and Usage

1. Install dependencies:
   ```bash
   pip install pandas lifetimes
   ```

2. Run the Python script:
   ```bash
   python CRM/FLO_CLTV_Prediction_miuul.py
   ```

3. Update the data file path in the script if necessary (default: `/Users/beyza/Desktop/miuul/FLOCLTVPrediction/flo_data_20k.csv`).

## Outputs

- Expected number of purchases for each customer in 3 and 6 months
- Expected average transaction value
- 6-month CLTV calculations
- Customer segmentation based on CLTV (A, B, C, D)
- Segment summary statistics

## References

- [Lifetimes Python Library](https://lifetimes.readthedocs.io/en/latest/index.html)
- [BG-NBD and Gamma-Gamma Models](https://www.probabilisticworld.com/bgnbd-and-gamma-gamma-models-for-customer-base-analysis/)

---
