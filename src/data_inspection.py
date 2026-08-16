import pandas as pd


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

file_path = "data/Telco-Customer-Churn.csv"

df = pd.read_csv(file_path)


# --------------------------------------------------
# BASIC INFORMATION
# --------------------------------------------------

print("=" * 60)
print("CUSTOMER CHURN DATASET")
print("=" * 60)

print("\nFirst 5 rows:")
print(df.head())


# --------------------------------------------------
# DATASET SIZE
# --------------------------------------------------

print("\nDataset shape:")
print(df.shape)


# --------------------------------------------------
# COLUMN NAMES
# --------------------------------------------------

print("\nColumn names:")
print(df.columns.tolist())


# --------------------------------------------------
# DATA TYPES
# --------------------------------------------------

print("\nData types:")
print(df.dtypes)


# --------------------------------------------------
# MISSING VALUES
# --------------------------------------------------

print("\nMissing values:")
print(df.isnull().sum())


# --------------------------------------------------
# DUPLICATE RECORDS
# --------------------------------------------------

print("\nNumber of duplicate rows:")
print(df.duplicated().sum())


# --------------------------------------------------
# STATISTICAL SUMMARY
# --------------------------------------------------

print("\nStatistical summary:")
print(df.describe())


# --------------------------------------------------
# CHURN DISTRIBUTION
# --------------------------------------------------

print("\nChurn distribution:")
print(df["Churn"].value_counts())


print("\nChurn percentage:")
print(df["Churn"].value_counts(normalize=True) * 100)