import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


# ============================================================
# 1. LOAD CLEANED DATASET
# ============================================================

file_path = (
    "data/cleaned/"
    "Telco-Customer-Churn-Cleaned.csv"
)

df = pd.read_csv(file_path)

print("=" * 70)
print("MACHINE LEARNING PREPROCESSING")
print("=" * 70)


# ============================================================
# 2. DISPLAY DATASET INFORMATION
# ============================================================

print("\nDataset shape:")
print(df.shape)

print("\nDataset columns:")
print(df.columns.tolist())


# ============================================================
# 3. REMOVE CHURN LABEL IF IT EXISTS
# ============================================================

# Churn_Label was created only for EDA.
# It should NOT be used as a model feature.

if "Churn_Label" in df.columns:
    df.drop("Churn_Label", axis=1, inplace=True)


# ============================================================
# 4. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop("Churn", axis=1)

y = df["Churn"]


print("\nFeature matrix shape:")
print(X.shape)

print("\nTarget shape:")
print(y.shape)


# ============================================================
# 5. DISPLAY TARGET DISTRIBUTION
# ============================================================

print("\nTarget distribution:")
print(y.value_counts())

print("\nTarget percentage:")
print(
    y.value_counts(normalize=True) * 100
)


# ============================================================
# 6. IDENTIFY NUMERICAL FEATURES
# ============================================================

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()


print("\nNumerical features:")
print(numeric_features)


# ============================================================
# 7. IDENTIFY CATEGORICAL FEATURES
# ============================================================

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()


print("\nCategorical features:")
print(categorical_features)


# ============================================================
# 8. CHECK MISSING VALUES
# ============================================================

print("\nMissing values in features:")

missing_values = X.isnull().sum()

print(
    missing_values[
        missing_values > 0
    ]
)


# ============================================================
# 9. TRAIN/TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining set shape:")
print(X_train.shape)

print("\nTesting set shape:")
print(X_test.shape)


# ============================================================
# 10. NUMERICAL PREPROCESSING
# ============================================================

numeric_transformer = Pipeline(
    steps=[
        (
            "scaler",
            StandardScaler()
        )
    ]
)


# ============================================================
# 11. CATEGORICAL PREPROCESSING
# ============================================================

categorical_transformer = Pipeline(
    steps=[
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


# ============================================================
# 12. COMBINE PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numeric_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)


# ============================================================
# 13. CREATE MACHINE LEARNING PIPELINE
# ============================================================

model_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ]
)


# ============================================================
# 14. TRAIN THE PIPELINE
# ============================================================

print("\nTraining Logistic Regression model...")

model_pipeline.fit(
    X_train,
    y_train
)

print("Training completed successfully.")


# ============================================================
# 15. MAKE PREDICTIONS
# ============================================================

y_pred = model_pipeline.predict(
    X_test
)


# ============================================================
# 16. CHECK PREDICTIONS
# ============================================================

print("\nFirst 20 predictions:")

print(y_pred[:20])


print("\nActual values:")

print(y_test.iloc[:20].values)


# ============================================================
# 17. CHECK PREDICTION PROBABILITIES
# ============================================================

y_probability = (
    model_pipeline.predict_proba(X_test)[:, 1]
)


print("\nFirst 20 churn probabilities:")

print(y_probability[:20])


# ============================================================
# 18. COMPLETION MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("PREPROCESSING AND BASELINE MODEL COMPLETED")
print("=" * 70)