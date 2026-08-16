import pandas as pd


# Load cleaned dataset
file_path = (
    "data/cleaned/"
    "Telco-Customer-Churn-Cleaned.csv"
)

df = pd.read_csv(file_path)


print("=" * 60)
print("CLEANED DATASET VALIDATION")
print("=" * 60)


# Dataset shape
print("\nDataset shape:")
print(df.shape)


# Check missing values
print("\nMissing values:")
print(df.isnull().sum())


# Check duplicates
print("\nDuplicate rows:")
print(df.duplicated().sum())


# Check data types
print("\nData types:")
print(df.dtypes)


# Check target
print("\nChurn values:")
print(df["Churn"].value_counts())


# Check first rows
print("\nFirst five rows:")
print(df.head())


print("\n" + "=" * 60)
print("VALIDATION COMPLETED")
print("=" * 60)