import pandas as pd
import os


# ============================================================
# 1. LOAD THE DATASET
# ============================================================

input_file = "data/Telco-Customer-Churn.csv"

df = pd.read_csv(input_file)

print("=" * 70)
print("CUSTOMER CHURN DATA CLEANING")
print("=" * 70)

print("\nOriginal dataset shape:")
print(df.shape)


# ============================================================
# 2. DISPLAY COLUMN NAMES
# ============================================================

print("\nOriginal columns:")
print(df.columns.tolist())


# ============================================================
# 3. REMOVE CUSTOMER ID
# ============================================================

# customerID identifies individual customers but does not
# provide useful predictive information for our model.

if "customerID" in df.columns:
    df.drop("customerID", axis=1, inplace=True)

print("\nCustomerID removed.")


# ============================================================
# 4. CONVERT TOTALCHARGES TO NUMERIC
# ============================================================

# Some TotalCharges values may contain blank spaces.
# Errors='coerce' converts invalid values to NaN.

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

print("\nTotalCharges converted to numeric.")


# ============================================================
# 5. CHECK MISSING VALUES
# ============================================================

print("\nMissing values before treatment:")

missing_before = df.isnull().sum()

print(
    missing_before[missing_before > 0]
)


# ============================================================
# 6. HANDLE MISSING VALUES
# ============================================================

# TotalCharges is numeric, so we use the median
# to replace missing values.

if df["TotalCharges"].isnull().sum() > 0:

    median_total_charges = df["TotalCharges"].median()

    df["TotalCharges"] = df["TotalCharges"].fillna(
        median_total_charges
    )


# ============================================================
# 7. CHECK DUPLICATE RECORDS
# ============================================================

duplicates = df.duplicated().sum()

print("\nNumber of duplicate rows:")
print(duplicates)

if duplicates > 0:
    df.drop_duplicates(inplace=True)

print("\nDuplicates removed if any.")


# ============================================================
# 8. CONVERT TARGET VARIABLE
# ============================================================

# Convert:
# No  -> 0
# Yes -> 1

df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

print("\nChurn converted to numerical values.")


# ============================================================
# 9. CHECK FINAL MISSING VALUES
# ============================================================

print("\nMissing values after cleaning:")

missing_after = df.isnull().sum()

print(
    missing_after[missing_after > 0]
)


# ============================================================
# 10. CHECK FINAL DATA TYPES
# ============================================================

print("\nFinal data types:")

print(df.dtypes)


# ============================================================
# 11. FINAL DATASET SHAPE
# ============================================================

print("\nFinal dataset shape:")

print(df.shape)


# ============================================================
# 12. CHURN DISTRIBUTION
# ============================================================

print("\nChurn distribution:")

print(df["Churn"].value_counts())


# ============================================================
# 13. CHURN PERCENTAGE
# ============================================================

print("\nChurn percentage:")

print(
    df["Churn"].value_counts(normalize=True) * 100
)


# ============================================================
# 14. CREATE CLEANED DATA FOLDER
# ============================================================

cleaned_folder = "data/cleaned"

os.makedirs(
    cleaned_folder,
    exist_ok=True
)


# ============================================================
# 15. SAVE CLEANED DATASET
# ============================================================

output_file = (
    "data/cleaned/"
    "Telco-Customer-Churn-Cleaned.csv"
)

df.to_csv(
    output_file,
    index=False
)

print("\nCleaned dataset saved to:")

print(output_file)


# ============================================================
# 16. DISPLAY FIRST FIVE ROWS
# ============================================================

print("\nFirst five rows of cleaned dataset:")

print(df.head())


print("\n" + "=" * 70)
print("DATA CLEANING COMPLETED SUCCESSFULLY")
print("=" * 70)