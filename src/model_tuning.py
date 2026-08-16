import os
import joblib
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold
)

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. LOAD CLEANED DATASET
# ============================================================

file_path = (
    "data/cleaned/"
    "Telco-Customer-Churn-Cleaned.csv"
)

df = pd.read_csv(file_path)


print("=" * 75)
print("STAGE 7 - HYPERPARAMETER TUNING")
print("=" * 75)


# ============================================================
# 2. REMOVE EDA-ONLY COLUMN
# ============================================================

if "Churn_Label" in df.columns:

    df.drop(
        "Churn_Label",
        axis=1,
        inplace=True
    )


# ============================================================
# 3. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(
    "Churn",
    axis=1
)

y = df["Churn"]


# ============================================================
# 4. IDENTIFY FEATURE TYPES
# ============================================================

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()


categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()


print("\nNumerical features:")
print(numeric_features)


print("\nCategorical features:")
print(categorical_features)


# ============================================================
# 5. TRAIN/TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining set:")
print(X_train.shape)


print("\nTesting set:")
print(X_test.shape)


# ============================================================
# 6. NUMERICAL TRANSFORMATION
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
# 7. CATEGORICAL TRANSFORMATION
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
# 8. COMBINE PREPROCESSING
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
# 9. CREATE LOGISTIC REGRESSION PIPELINE
# ============================================================

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                random_state=42
            )
        )
    ]
)


# ============================================================
# 10. DEFINE HYPERPARAMETER GRID
# ============================================================

param_grid = {

    "classifier__C": [
        0.01,
        0.1,
        1,
        10,
        100
    ],

    "classifier__solver": [
        "liblinear",
        "lbfgs"
    ],

    "classifier__class_weight": [
        None,
        "balanced"
    ]
}


# ============================================================
# 11. STRATIFIED CROSS-VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


print("\n")
print("=" * 75)
print("STARTING GRID SEARCH")
print("=" * 75)


# ============================================================
# 12. GRID SEARCH
# ============================================================

grid_search = GridSearchCV(

    estimator=pipeline,

    param_grid=param_grid,

    scoring="roc_auc",

    cv=cv,

    n_jobs=-1,

    verbose=1
)


grid_search.fit(
    X_train,
    y_train
)


# ============================================================
# 13. BEST PARAMETERS
# ============================================================

print("\n")
print("=" * 75)
print("BEST HYPERPARAMETERS")
print("=" * 75)

print(
    grid_search.best_params_
)


# ============================================================
# 14. BEST CROSS-VALIDATION SCORE
# ============================================================

print("\nBest cross-validation ROC-AUC:")

print(
    f"{grid_search.best_score_:.4f}"
)


# ============================================================
# 15. GET BEST MODEL
# ============================================================

best_model = grid_search.best_estimator_


# ============================================================
# 16. TEST SET PREDICTIONS
# ============================================================

y_pred = best_model.predict(
    X_test
)


y_probability = (
    best_model.predict_proba(X_test)[:, 1]
)


# ============================================================
# 17. EVALUATE FINAL MODEL
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)


recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)


f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


roc_auc = roc_auc_score(
    y_test,
    y_probability
)


# ============================================================
# 18. PRINT FINAL METRICS
# ============================================================

print("\n")
print("=" * 75)
print("TUNED MODEL TEST RESULTS")
print("=" * 75)


print(
    f"\nAccuracy:  {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall:    {recall:.4f}"
)

print(
    f"F1-Score:  {f1:.4f}"
)

print(
    f"ROC-AUC:   {roc_auc:.4f}"
)


# ============================================================
# 19. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "No Churn",
            "Churn"
        ],
        zero_division=0
    )
)


# ============================================================
# 20. CONFUSION MATRIX
# ============================================================

print("\nConfusion Matrix:")

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)


# ============================================================
# 21. SAVE RESULTS
# ============================================================

os.makedirs(
    "data/model_results",
    exist_ok=True
)


tuned_results = pd.DataFrame({

    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "ROC-AUC"
    ],

    "Tuned_Logistic_Regression": [
        accuracy,
        precision,
        recall,
        f1,
        roc_auc
    ]

})


tuned_results.to_csv(
    "data/model_results/"
    "tuned_logistic_regression_results.csv",
    index=False
)


# ============================================================
# 22. SAVE FINAL MODEL
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)


final_model_path = (
    "models/"
    "final_churn_model.pkl"
)


joblib.dump(
    best_model,
    final_model_path
)


print("\n")
print("=" * 75)
print("FINAL MODEL SAVED")
print("=" * 75)


print(
    f"Model: {final_model_path}"
)


print("\n")
print("=" * 75)
print("STAGE 7 COMPLETED SUCCESSFULLY")
print("=" * 75)