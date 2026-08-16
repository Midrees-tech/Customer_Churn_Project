import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    roc_curve
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
print("CUSTOMER CHURN - MODEL TRAINING AND EVALUATION")
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


print("\nTraining data:")
print(X_train.shape)

print("\nTesting data:")
print(X_test.shape)


# ============================================================
# 6. PREPROCESSING PIPELINES
# ============================================================

numeric_transformer = Pipeline(
    steps=[
        (
            "scaler",
            StandardScaler()
        )
    ]
)


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
# 7. DEFINE MACHINE LEARNING MODELS
# ============================================================

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

    "Decision Tree":
        DecisionTreeClassifier(
            random_state=42,
            max_depth=8
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ),

    "Support Vector Machine":
        SVC(
            probability=True,
            random_state=42
        ),

    "K-Nearest Neighbors":
        KNeighborsClassifier(
            n_neighbors=5
        )
}


# ============================================================
# 8. CREATE OUTPUT DIRECTORIES
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)

os.makedirs(
    "data/model_results",
    exist_ok=True
)


# ============================================================
# 9. TRAIN AND EVALUATE MODELS
# ============================================================

results = []

trained_models = {}

roc_data = {}


for model_name, model in models.items():

    print("\n" + "=" * 75)

    print(
        f"TRAINING: {model_name}"
    )

    print("=" * 75)


    # --------------------------------------------------------
    # CREATE PIPELINE
    # --------------------------------------------------------

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                model
            )
        ]
    )


    # --------------------------------------------------------
    # TRAIN MODEL
    # --------------------------------------------------------

    pipeline.fit(
        X_train,
        y_train
    )


    print(
        "Training completed."
    )


    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    y_pred = pipeline.predict(
        X_test
    )


    # --------------------------------------------------------
    # PREDICTION PROBABILITIES
    # --------------------------------------------------------

    y_probability = (
        pipeline.predict_proba(X_test)[:, 1]
    )


    # --------------------------------------------------------
    # EVALUATION METRICS
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # STORE RESULTS
    # --------------------------------------------------------

    results.append({

        "Model": model_name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1-Score": f1,

        "ROC-AUC": roc_auc
    })


    # --------------------------------------------------------
    # SAVE TRAINED MODEL IN MEMORY
    # --------------------------------------------------------

    trained_models[
        model_name
    ] = pipeline


    # --------------------------------------------------------
    # ROC CURVE DATA
    # --------------------------------------------------------

    fpr, tpr, thresholds = roc_curve(
        y_test,
        y_probability
    )

    roc_data[
        model_name
    ] = (
        fpr,
        tpr,
        roc_auc
    )


    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print("\nAccuracy:")
    print(f"{accuracy:.4f}")

    print("\nPrecision:")
    print(f"{precision:.4f}")

    print("\nRecall:")
    print(f"{recall:.4f}")

    print("\nF1-Score:")
    print(f"{f1:.4f}")

    print("\nROC-AUC:")
    print(f"{roc_auc:.4f}")


    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

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
# 10. CREATE MODEL COMPARISON TABLE
# ============================================================

results_df = pd.DataFrame(
    results
)


print("\n")
print("=" * 75)
print("MODEL COMPARISON")
print("=" * 75)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# 11. SORT BY ROC-AUC
# ============================================================

results_df = results_df.sort_values(
    by="ROC-AUC",
    ascending=False
)


print("\n")
print("=" * 75)
print("MODELS RANKED BY ROC-AUC")
print("=" * 75)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# 12. SAVE RESULTS
# ============================================================

results_df.to_csv(
    "data/model_results/"
    "model_comparison.csv",
    index=False
)


# ============================================================
# 13. SAVE BEST MODEL
# ============================================================

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[
    best_model_name
]


best_model_path = (
    "models/"
    "best_churn_model.pkl"
)


joblib.dump(
    best_model,
    best_model_path
)


print("\n")
print("=" * 75)
print("BEST MODEL")
print("=" * 75)

print(
    f"Best model: {best_model_name}"
)

print(
    f"Saved to: {best_model_path}"
)


# ============================================================
# 14. CREATE CONFUSION MATRIX FOR BEST MODEL
# ============================================================

best_predictions = best_model.predict(
    X_test
)


cm = confusion_matrix(
    y_test,
    best_predictions
)


print("\n")
print("=" * 75)
print("CONFUSION MATRIX - BEST MODEL")
print("=" * 75)

print(cm)


disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "No Churn",
        "Churn"
    ]
)


fig, ax = plt.subplots(
    figsize=(7, 6)
)


disp.plot(
    ax=ax
)


ax.set_title(
    f"Confusion Matrix - {best_model_name}"
)


plt.tight_layout()


plt.savefig(
    "data/model_results/"
    "best_model_confusion_matrix.png"
)


plt.close()


# ============================================================
# 15. ROC CURVE FOR ALL MODELS
# ============================================================

plt.figure(
    figsize=(10, 7)
)


for model_name, (
    fpr,
    tpr,
    auc_value
) in roc_data.items():

    plt.plot(
        fpr,
        tpr,
        label=(
            f"{model_name} "
            f"(AUC = {auc_value:.3f})"
        )
    )


# Random classifier reference line

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)


plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curve - Customer Churn Models"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


plt.savefig(
    "data/model_results/"
    "roc_curve_comparison.png"
)


plt.close()


# ============================================================
# 16. MODEL ACCURACY COMPARISON
# ============================================================

plt.figure(
    figsize=(10, 6)
)


plt.bar(
    results_df["Model"],
    results_df["Accuracy"]
)


plt.xlabel(
    "Machine Learning Model"
)

plt.ylabel(
    "Accuracy"
)

plt.title(
    "Model Accuracy Comparison"
)


plt.xticks(
    rotation=30,
    ha="right"
)


plt.ylim(
    0,
    1
)


plt.tight_layout()


plt.savefig(
    "data/model_results/"
    "model_accuracy_comparison.png"
)


plt.close()


# ============================================================
# 17. F1-SCORE COMPARISON
# ============================================================

plt.figure(
    figsize=(10, 6)
)


plt.bar(
    results_df["Model"],
    results_df["F1-Score"]
)


plt.xlabel(
    "Machine Learning Model"
)

plt.ylabel(
    "F1-Score"
)

plt.title(
    "F1-Score Comparison"
)


plt.xticks(
    rotation=30,
    ha="right"
)


plt.ylim(
    0,
    1
)


plt.tight_layout()


plt.savefig(
    "data/model_results/"
    "model_f1_comparison.png"
)


plt.close()


# ============================================================
# 18. FINAL MESSAGE
# ============================================================

print("\n")
print("=" * 75)
print("MODEL TRAINING AND EVALUATION COMPLETED SUCCESSFULLY")
print("=" * 75)

print(
    "\nResults saved in:"
)

print(
    "data/model_results/"
)

print(
    "\nBest model saved in:"
)

print(
    "models/best_churn_model.pkl"
)

print("\n")