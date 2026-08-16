# Customer Churn Prediction & Analytics System

## 1. Project Overview
An end-to-end machine-learning application for analyzing telecommunications customer churn and predicting whether an individual customer is likely to churn.

The project includes data validation, cleaning, exploratory data analysis, model development, model comparison, hyperparameter tuning, evaluation, and an interactive Streamlit application.

## 2. Problem Statement
Customer churn is a major challenge for telecommunications companies because losing existing customers can reduce revenue and increase customer-acquisition costs. This project develops a machine-learning system to identify customers who are more likely to leave a telecommunications service.

## 3. Objectives
1. Validate and clean the telecommunications customer dataset.
2. Perform exploratory data analysis.
3. Preprocess features for machine learning.
4. Train and compare classification algorithms.
5. Evaluate models using accuracy, precision, recall, F1-score, and ROC-AUC.
6. Tune the selected model.
7. Save the final trained model.
8. Develop an interactive Streamlit prediction system.
9. Provide customer-level churn probability and risk classification.
10. Present business-oriented churn insights.

## 4. Dataset
The project uses the Telco Customer Churn dataset. Variables include demographic information, tenure, telephone and internet services, security and support services, streaming services, contract type, billing information, payment method, monthly charges, total charges, and churn status.

The processed application dataset contains approximately **7,021 customers**:
- Churned: **1,857**
- Retained: **5,164**
- Churn rate: **26.45%**

## 5. Methodology

```text
Raw Dataset
    ↓
Data Inspection
    ↓
Data Validation
    ↓
Data Cleaning
    ↓
Exploratory Data Analysis
    ↓
Feature Preprocessing
    ↓
Model Training
    ↓
Model Comparison
    ↓
Hyperparameter Tuning
    ↓
Final Model
    ↓
Streamlit Application
    ↓
Customer Churn Prediction
```

## 6. Exploratory Data Analysis
The application provides visualizations for:
- Customer churn distribution
- Churn by contract type
- Churn by internet service
- Churn by payment method
- Tenure distribution by churn status
- Tenure versus monthly charges

The analysis indicates that contract type, internet service, payment method, tenure, and billing characteristics are useful variables for understanding churn patterns.

## 7. Machine-Learning Models
Five classification algorithms were evaluated:
1. Logistic Regression
2. Random Forest
3. Decision Tree
4. Support Vector Machine
5. K-Nearest Neighbors

## 8. Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 80.28% | 66.10% | 52.42% | 58.47% | **84.03%** |
| Random Forest | 77.79% | 61.11% | 44.35% | 51.40% | 81.31% |
| Decision Tree | 77.65% | 58.48% | 53.76% | 56.02% | 79.45% |
| Support Vector Machine | 79.86% | 66.30% | 48.66% | 56.12% | 78.50% |
| K-Nearest Neighbors | 75.30% | 53.78% | 47.85% | 50.64% | 76.94% |

Logistic Regression achieved the highest ROC-AUC among the evaluated models and was therefore selected for tuning and deployment.

## 9. Hyperparameter Tuning

The tuned Logistic Regression model achieved:

| Metric | Result |
|---|---:|
| Accuracy | 80.28% |
| Precision | 65.99% |
| Recall | 52.69% |
| F1-Score | 58.59% |
| ROC-AUC | **84.02%** |

The tuned model performed very similarly to the baseline Logistic Regression model. Therefore, the project should not claim that tuning produced a major improvement in ROC-AUC.

## 10. Final Model
The deployed final model is **Tuned Logistic Regression**.

Model file:

```text
models/final_churn_model.pkl
```

The project also retains:

```text
models/best_churn_model.pkl
```

## 11. Model Evaluation

### Confusion Matrix

The Logistic Regression confusion matrix contains:

| | Predicted No Churn | Predicted Churn |
|---|---:|---:|
| Actual No Churn | 933 | 100 |
| Actual Churn | 177 | 195 |

Thus:
- True Negatives = 933
- False Positives = 100
- False Negatives = 177
- True Positives = 195

### ROC Curve
The Logistic Regression ROC curve has an AUC of approximately **0.840**, demonstrating good discrimination between churn and non-churn customers.

## 12. Streamlit Application

The application contains four main sections:

### Dashboard
Displays:
- Total customers
- Churned customers
- Retained customers
- Churn rate
- Key business insights
- Churn distribution

### Churn Prediction
Users can enter customer information and obtain:
- Churn probability
- Predicted churn/no-churn status
- Risk level
- Customer information used for prediction

The prediction probability changes when relevant customer characteristics, such as contract type, are changed.

### EDA & Insights
Displays the exploratory visualizations generated during analysis.

### Model Performance
Displays:
- Model comparison table
- Accuracy comparison
- Confusion matrix
- ROC curve comparison
- Performance metrics

## 13. Example Prediction

```text
Churn Probability: 34.09%
Prediction: No Churn
Risk Level: Low Risk
```

The probability is an estimated model output, not a certainty.

## 14. Project Structure

```text
Customer_Churn_Project/
│
├── data/
│   ├── eda/
│   │   ├── tenure_vs_monthly_charges.png
│   │   └── total_charges_by_churn.png
│   ├── model_results/
│   │   ├── best_model_confusion_matrix.png
│   │   ├── model_accuracy_comparison.png
│   │   ├── model_comparison.csv
│   │   ├── model_f1_comparison.png
│   │   ├── roc_curve_comparison.png
│   │   └── tuned_logistic_regression_results.csv
│   └── Telco-Customer-Churn.csv
│
├── models/
│   ├── best_churn_model.pkl
│   └── final_churn_model.pkl
│
├── notebooks/
│
├── src/
│   ├── data_cleaning.py
│   ├── data_inspection.py
│   ├── data_validation.py
│   ├── eda.py
│   ├── model_training.py
│   ├── model_tuning.py
│   └── preprocessing.py
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## 15. Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit

## 16. Installation

### Create a virtual environment

```bash
python -m venv .venv
```

### Activate it on Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

### Install dependencies

```bash
pip install -r requirements.txt
```

## 17. Running the Application

From the project root:

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

## 18. Model Training

Baseline training:

```bash
python src/model_training.py
```

Model tuning:

```bash
python src/model_tuning.py
```

## 19. Business Interpretation
The system can support customer-retention decisions by identifying customers with elevated estimated churn probability.

Potential business actions include:
- Targeted retention campaigns
- Contract-renewal incentives
- Customer-support interventions
- Personalized offers
- Early identification of higher-risk customers

The model should support, rather than replace, business decisions.

## 20. Limitations
1. The model is trained on historical telecommunications data.
2. Performance depends on data quality and representativeness.
3. ROC-AUC of approximately 84% indicates good but imperfect discrimination.
4. Some churn cases are incorrectly classified.
5. The model does not establish causal relationships.
6. Customer behavior can change over time.
7. Predicted probabilities should not be interpreted as certainty.

## 21. Future Improvements
- Test additional algorithms.
- Address class imbalance where appropriate.
- Perform broader hyperparameter optimization.
- Calibrate predicted probabilities.
- Add feature-importance and explainability analysis.
- Add SHAP-based explanations.
- Optimize the classification threshold according to business costs.
- Add model monitoring and periodic retraining.
- Integrate with CRM systems.
- Deploy to the cloud.
- Add authentication and role-based access.
- Support batch prediction.

## 22. Reproducibility
The project separates major stages into individual scripts. A typical workflow is:

```bash
python src/data_inspection.py
python src/data_validation.py
python src/data_cleaning.py
python src/eda.py
python src/preprocessing.py
python src/model_training.py
python src/model_tuning.py
streamlit run app.py
```

Execute the scripts in their required dependency order.

## 23. Conclusion
This project successfully developed an end-to-end customer churn prediction and analytics system using machine learning.

Five classification algorithms were evaluated, with Logistic Regression producing the strongest initial ROC-AUC performance. Hyperparameter tuning was then applied, producing a final tuned Logistic Regression model with approximately **84.02% ROC-AUC**.

The final model was integrated into an interactive Streamlit application that enables users to explore churn patterns, examine model performance, enter customer information, estimate churn probability, and view a predicted risk level.

The project demonstrates how machine learning, exploratory data analysis, model evaluation, and interactive application development can be combined to provide practical decision-support capabilities for customer-retention analysis.

## 24. Author

**Author:** Idris Amosa  
**Project:** Customer Churn Prediction & Analytics System  
**Technology:** Python + Machine Learning + Streamlit

## 25. License
This project is intended for educational, research, and demonstration purposes. Before commercial use, the model should be independently validated using current and representative customer data.
