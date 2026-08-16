import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. LOAD CLEANED DATASET
# ============================================================

file_path = (
    "data/cleaned/"
    "Telco-Customer-Churn-Cleaned.csv"
)

df = pd.read_csv(file_path)


print("=" * 70)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 70)


# ============================================================
# 2. BASIC INFORMATION
# ============================================================

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)


# ============================================================
# 3. STATISTICAL SUMMARY
# ============================================================

print("\nStatistical summary:")
print(df.describe())


# ============================================================
# 4. CHURN DISTRIBUTION
# ============================================================

print("\nChurn distribution:")

churn_counts = df["Churn"].value_counts()

print(churn_counts)


print("\nChurn percentages:")

churn_percentages = (
    df["Churn"]
    .value_counts(normalize=True)
    * 100
)

print(churn_percentages)


# ============================================================
# 5. CHURN LABELS
# ============================================================

df["Churn_Label"] = df["Churn"].map({
    0: "No",
    1: "Yes"
})


# ============================================================
# 6. CREATE OUTPUT FOLDER
# ============================================================

import os

os.makedirs(
    "data/eda",
    exist_ok=True
)


# ============================================================
# 7. CHURN DISTRIBUTION PLOT
# ============================================================

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Churn_Label"
)

plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")

plt.tight_layout()

plt.savefig(
    "data/eda/churn_distribution.png"
)

plt.close()


# ============================================================
# 8. CHURN BY CONTRACT TYPE
# ============================================================

contract_churn = pd.crosstab(
    df["Contract"],
    df["Churn_Label"],
    normalize="index"
) * 100

print("\nChurn percentage by contract:")
print(contract_churn)


plt.figure(figsize=(10, 6))

sns.countplot(
    data=df,
    x="Contract",
    hue="Churn_Label"
)

plt.title("Customer Churn by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")

plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig(
    "data/eda/churn_by_contract.png"
)

plt.close()


# ============================================================
# 9. CHURN BY INTERNET SERVICE
# ============================================================

internet_churn = pd.crosstab(
    df["InternetService"],
    df["Churn_Label"],
    normalize="index"
) * 100

print("\nChurn percentage by internet service:")
print(internet_churn)


plt.figure(figsize=(9, 6))

sns.countplot(
    data=df,
    x="InternetService",
    hue="Churn_Label"
)

plt.title("Customer Churn by Internet Service")
plt.xlabel("Internet Service")
plt.ylabel("Number of Customers")

plt.tight_layout()

plt.savefig(
    "data/eda/churn_by_internet_service.png"
)

plt.close()


# ============================================================
# 10. CHURN BY PAYMENT METHOD
# ============================================================

payment_churn = pd.crosstab(
    df["PaymentMethod"],
    df["Churn_Label"],
    normalize="index"
) * 100

print("\nChurn percentage by payment method:")
print(payment_churn)


plt.figure(figsize=(12, 6))

sns.countplot(
    data=df,
    x="PaymentMethod",
    hue="Churn_Label"
)

plt.title("Customer Churn by Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Number of Customers")

plt.xticks(
    rotation=30,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    "data/eda/churn_by_payment_method.png"
)

plt.close()


# ============================================================
# 11. CHURN BY GENDER
# ============================================================

gender_churn = pd.crosstab(
    df["gender"],
    df["Churn_Label"],
    normalize="index"
) * 100

print("\nChurn percentage by gender:")
print(gender_churn)


plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="gender",
    hue="Churn_Label"
)

plt.title("Customer Churn by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Customers")

plt.tight_layout()

plt.savefig(
    "data/eda/churn_by_gender.png"
)

plt.close()


# ============================================================
# 12. CHURN BY SENIOR CITIZEN STATUS
# ============================================================

senior_churn = pd.crosstab(
    df["SeniorCitizen"],
    df["Churn_Label"],
    normalize="index"
) * 100

print("\nChurn percentage by senior citizen status:")
print(senior_churn)


plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="SeniorCitizen",
    hue="Churn_Label"
)

plt.title("Customer Churn by Senior Citizen Status")
plt.xlabel("Senior Citizen")
plt.ylabel("Number of Customers")

plt.tight_layout()

plt.savefig(
    "data/eda/churn_by_senior_citizen.png"
)

plt.close()


# ============================================================
# 13. CHURN BY PARTNER STATUS
# ============================================================

partner_churn = pd.crosstab(
    df["Partner"],
    df["Churn_Label"],
    normalize="index"
) * 100

print("\nChurn percentage by partner status:")
print(partner_churn)


plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Partner",
    hue="Churn_Label"
)

plt.title("Customer Churn by Partner Status")
plt.xlabel("Partner")
plt.ylabel("Number of Customers")

plt.tight_layout()

plt.savefig(
    "data/eda/churn_by_partner.png"
)

plt.close()


# ============================================================
# 14. CHURN BY TECH SUPPORT
# ============================================================

tech_support_churn = pd.crosstab(
    df["TechSupport"],
    df["Churn_Label"],
    normalize="index"
) * 100

print("\nChurn percentage by technical support:")
print(tech_support_churn)


plt.figure(figsize=(9, 6))

sns.countplot(
    data=df,
    x="TechSupport",
    hue="Churn_Label"
)

plt.title("Customer Churn by Technical Support")
plt.xlabel("Technical Support")
plt.ylabel("Number of Customers")

plt.tight_layout()

plt.savefig(
    "data/eda/churn_by_tech_support.png"
)

plt.close()


# ============================================================
# 15. CHURN BY ONLINE SECURITY
# ============================================================

plt.figure(figsize=(9, 6))

sns.countplot(
    data=df,
    x="OnlineSecurity",
    hue="Churn_Label"
)

plt.title("Customer Churn by Online Security")
plt.xlabel("Online Security")
plt.ylabel("Number of Customers")

plt.tight_layout()

plt.savefig(
    "data/eda/churn_by_online_security.png"
)

plt.close()


# ============================================================
# 16. TENURE DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

sns.histplot(
    data=df,
    x="tenure",
    hue="Churn_Label",
    bins=30,
    kde=True,
    element="step"
)

plt.title("Tenure Distribution by Churn Status")
plt.xlabel("Tenure (Months)")
plt.ylabel("Number of Customers")

plt.tight_layout()

plt.savefig(
    "data/eda/tenure_distribution.png"
)

plt.close()


# ============================================================
# 17. MONTHLY CHARGES
# ============================================================

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=df,
    x="Churn_Label",
    y="MonthlyCharges"
)

plt.title("Monthly Charges by Churn Status")
plt.xlabel("Churn")
plt.ylabel("Monthly Charges")

plt.tight_layout()

plt.savefig(
    "data/eda/monthly_charges_by_churn.png"
)

plt.close()


# ============================================================
# 18. TOTAL CHARGES
# ============================================================

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=df,
    x="Churn_Label",
    y="TotalCharges"
)

plt.title("Total Charges by Churn Status")
plt.xlabel("Churn")
plt.ylabel("Total Charges")

plt.tight_layout()

plt.savefig(
    "data/eda/total_charges_by_churn.png"
)

plt.close()


# ============================================================
# 19. TENURE VS MONTHLY CHARGES
# ============================================================

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="tenure",
    y="MonthlyCharges",
    hue="Churn_Label",
    alpha=0.6
)

plt.title(
    "Tenure vs Monthly Charges by Churn Status"
)

plt.xlabel("Tenure (Months)")
plt.ylabel("Monthly Charges")

plt.tight_layout()

plt.savefig(
    "data/eda/tenure_vs_monthly_charges.png"
)

plt.close()


# ============================================================
# 20. NUMERICAL CORRELATION
# ============================================================

numeric_columns = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Churn"
]

correlation = df[numeric_columns].corr()

print("\nCorrelation matrix:")
print(correlation)


# ============================================================
# 21. CORRELATION HEATMAP
# ============================================================

plt.figure(figsize=(9, 7))

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title("Correlation Matrix")

plt.tight_layout()

plt.savefig(
    "data/eda/correlation_heatmap.png"
)

plt.close()


# ============================================================
# 22. AVERAGE NUMERICAL VARIABLES BY CHURN
# ============================================================

print("\nAverage numerical variables by churn:")

average_by_churn = df.groupby(
    "Churn_Label"
)[
    [
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]
].mean()

print(average_by_churn)


# ============================================================
# 23. SAVE CHURN ANALYSIS
# ============================================================

contract_churn.to_csv(
    "data/eda/contract_churn.csv"
)

internet_churn.to_csv(
    "data/eda/internet_service_churn.csv"
)

payment_churn.to_csv(
    "data/eda/payment_method_churn.csv"
)

gender_churn.to_csv(
    "data/eda/gender_churn.csv"
)

tech_support_churn.to_csv(
    "data/eda/tech_support_churn.csv"
)


# ============================================================
# 24. FINISH
# ============================================================

print("\n" + "=" * 70)
print("EXPLORATORY DATA ANALYSIS COMPLETED")
print("=" * 70)

print("\nEDA files have been saved in:")
print("data/eda/")