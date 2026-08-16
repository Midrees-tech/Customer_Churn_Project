import streamlit as st
import pandas as pd
import joblib
import os


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Churn Analytics",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# FILE PATHS
# ============================================================

MODEL_PATH = "models/final_churn_model.pkl"

DATA_PATH = (
    "data/cleaned/"
    "Telco-Customer-Churn-Cleaned.csv"
)

RESULTS_PATH = (
    "data/model_results/"
    "model_comparison.csv"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    return joblib.load(
        MODEL_PATH
    )


model = load_model()


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    return pd.read_csv(
        DATA_PATH
    )


df = load_data()


# ============================================================
# LOAD MODEL RESULTS
# ============================================================

@st.cache_data
def load_model_results():

    return pd.read_csv(
        RESULTS_PATH
    )


model_results = load_model_results()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("📊 Churn Analytics")

    st.markdown(
        """
        ### Navigation

        Use the tabs in the main area to explore:

        - 📊 Dashboard
        - 🔮 Churn Prediction
        - 📈 EDA & Insights
        - 🤖 Model Performance
        """
    )

    st.divider()

    st.subheader("Final Model")

    st.write(
        "Tuned Logistic Regression"
    )

    st.metric(
        "ROC-AUC",
        "84.02%"
    )

    st.divider()

    st.info(
        """
        This system predicts the probability
        that a telecommunications customer
        will churn.
        """
    )


# ============================================================
# MAIN TITLE
# ============================================================

st.title(
    "📊 Customer Churn Prediction & Analytics System"
)

st.markdown(
    """
    An end-to-end machine-learning application for
    analyzing and predicting customer churn.
    """
)


st.divider()


# ============================================================
# NAVIGATION TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Dashboard",
        "🔮 Churn Prediction",
        "📈 EDA & Insights",
        "🤖 Model Performance"
    ]
)


# ============================================================
# TAB 1 — DASHBOARD
# ============================================================

with tab1:

    st.header(
        "📊 Customer Churn Dashboard"
    )

    # --------------------------------------------------------
    # BASIC STATISTICS
    # --------------------------------------------------------

    total_customers = len(df)

    churned_customers = (
        df["Churn"] == 1
    ).sum()

    non_churned_customers = (
        df["Churn"] == 0
    ).sum()

    churn_rate = (
        churned_customers /
        total_customers
    ) * 100


    # --------------------------------------------------------
    # METRIC CARDS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Total Customers",
            f"{total_customers:,}"
        )


    with col2:

        st.metric(
            "Churned Customers",
            f"{churned_customers:,}"
        )


    with col3:

        st.metric(
            "Customers Retained",
            f"{non_churned_customers:,}"
        )


    with col4:

        st.metric(
            "Churn Rate",
            f"{churn_rate:.2f}%"
        )


    st.divider()


    # --------------------------------------------------------
    # BUSINESS INSIGHTS
    # --------------------------------------------------------

    st.subheader(
        "📌 Key Business Insights"
    )


    insight1, insight2 = st.columns(2)


    with insight1:

        st.info(
            f"""
            **Customer Base**

            The dataset contains approximately
            **{total_customers:,} customers**.

            Approximately **{churn_rate:.2f}%**
            of customers are classified as churners.
            """
        )


    with insight2:

        st.success(
            """
            **Model Capability**

            The final tuned Logistic Regression
            model achieved approximately
            **84.02% ROC-AUC**.
            """
        )


    # --------------------------------------------------------
    # CHURN DISTRIBUTION
    # --------------------------------------------------------

    st.subheader(
        "Customer Churn Distribution"
    )


    churn_counts = df["Churn"].value_counts()

    churn_display = pd.DataFrame(
        {
            "Customer Status": [
                "No Churn",
                "Churn"
            ],

            "Customers": [
                churn_counts.get(0, 0),
                churn_counts.get(1, 0)
            ]
        }
    )


    st.bar_chart(
        churn_display.set_index(
            "Customer Status"
        )
    )


# ============================================================
# TAB 2 — CHURN PREDICTION
# ============================================================

with tab2:

    st.header(
        "🔮 Individual Customer Churn Prediction"
    )

    st.write(
        """
        Enter the customer's information below and
        click **Predict Customer Churn**.
        """
    )


    # --------------------------------------------------------
    # CUSTOMER INFORMATION
    # --------------------------------------------------------

    st.subheader(
        "👤 Customer Information"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        gender = st.selectbox(
            "Gender",
            [
                "Male",
                "Female"
            ]
        )


        senior_citizen = st.selectbox(
            "Senior Citizen",
            [
                0,
                1
            ]
        )


        partner = st.selectbox(
            "Partner",
            [
                "Yes",
                "No"
            ]
        )


        dependents = st.selectbox(
            "Dependents",
            [
                "Yes",
                "No"
            ]
        )


    with col2:

        tenure = st.number_input(
            "Tenure (months)",
            min_value=0,
            max_value=100,
            value=12
        )


        phone_service = st.selectbox(
            "Phone Service",
            [
                "Yes",
                "No"
            ]
        )


        multiple_lines = st.selectbox(
            "Multiple Lines",
            [
                "Yes",
                "No",
                "No phone service"
            ]
        )


        internet_service = st.selectbox(
            "Internet Service",
            [
                "DSL",
                "Fiber optic",
                "No"
            ]
        )


    with col3:

        online_security = st.selectbox(
            "Online Security",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )


        online_backup = st.selectbox(
            "Online Backup",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )


        device_protection = st.selectbox(
            "Device Protection",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )


        tech_support = st.selectbox(
            "Tech Support",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )


    # --------------------------------------------------------
    # STREAMING
    # --------------------------------------------------------

    st.subheader(
        "📺 Streaming Services"
    )


    col4, col5, col6 = st.columns(3)


    with col4:

        streaming_tv = st.selectbox(
            "Streaming TV",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )


    with col5:

        streaming_movies = st.selectbox(
            "Streaming Movies",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )


    with col6:

        contract = st.selectbox(
            "Contract",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ]
        )


    # --------------------------------------------------------
    # BILLING
    # --------------------------------------------------------

    st.subheader(
        "💳 Billing Information"
    )


    col7, col8, col9 = st.columns(3)


    with col7:

        paperless_billing = st.selectbox(
            "Paperless Billing",
            [
                "Yes",
                "No"
            ]
        )


    with col8:

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )


    with col9:

        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            max_value=200.0,
            value=70.0
        )


    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        max_value=10000.0,
        value=840.0
    )


    st.divider()


    # --------------------------------------------------------
    # PREDICTION BUTTON
    # --------------------------------------------------------

    predict = st.button(
        "🔮 Predict Customer Churn",
        type="primary",
        width="stretch"
    )


    if predict:

        customer_data = pd.DataFrame({

            "gender": [gender],

            "SeniorCitizen": [
                senior_citizen
            ],

            "Partner": [
                partner
            ],

            "Dependents": [
                dependents
            ],

            "tenure": [
                tenure
            ],

            "PhoneService": [
                phone_service
            ],

            "MultipleLines": [
                multiple_lines
            ],

            "InternetService": [
                internet_service
            ],

            "OnlineSecurity": [
                online_security
            ],

            "OnlineBackup": [
                online_backup
            ],

            "DeviceProtection": [
                device_protection
            ],

            "TechSupport": [
                tech_support
            ],

            "StreamingTV": [
                streaming_tv
            ],

            "StreamingMovies": [
                streaming_movies
            ],

            "Contract": [
                contract
            ],

            "PaperlessBilling": [
                paperless_billing
            ],

            "PaymentMethod": [
                payment_method
            ],

            "MonthlyCharges": [
                monthly_charges
            ],

            "TotalCharges": [
                total_charges
            ]
        })


        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(
            customer_data
        )


        probability = model.predict_proba(
            customer_data
        )[0][1]


        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        st.subheader(
            "Prediction Result"
        )


        result1, result2, result3 = st.columns(3)


        with result1:

            st.metric(
                "Churn Probability",
                f"{probability * 100:.2f}%"
            )


        with result2:

            if prediction[0] == 1:

                st.error(
                    "⚠️ Predicted Churn"
                )

            else:

                st.success(
                    "✅ Predicted No Churn"
                )


        with result3:

            if probability >= 0.70:

                st.error(
                    "High Risk"
                )

            elif probability >= 0.40:

                st.warning(
                    "Medium Risk"
                )

            else:

                st.success(
                    "Low Risk"
                )


        st.progress(
            float(probability)
        )


        st.subheader(
            "Customer Information Used"
        )


        st.dataframe(
            customer_data,
            width="stretch"
        )


# ============================================================
# TAB 3 — EDA AND INSIGHTS
# ============================================================

with tab3:

    st.header(
        "📈 Exploratory Data Analysis"
    )


    st.write(
        """
        The following visualizations were generated during
        the exploratory data analysis stage.
        """
    )


    # --------------------------------------------------------
    # CHURN DISTRIBUTION
    # --------------------------------------------------------

    col1, col2 = st.columns(2)


    with col1:

        image_path = (
            "data/eda/"
            "churn_distribution.png"
        )

        if os.path.exists(image_path):

            st.image(
                image_path,
                caption="Customer Churn Distribution"
            )


    with col2:

        image_path = (
            "data/eda/"
            "churn_by_contract.png"
        )

        if os.path.exists(image_path):

            st.image(
                image_path,
                caption="Churn by Contract"
            )


    # --------------------------------------------------------
    # INTERNET SERVICE / PAYMENT
    # --------------------------------------------------------

    col3, col4 = st.columns(2)


    with col3:

        image_path = (
            "data/eda/"
            "churn_by_internet_service.png"
        )

        if os.path.exists(image_path):

            st.image(
                image_path,
                caption="Churn by Internet Service"
            )


    with col4:

        image_path = (
            "data/eda/"
            "churn_by_payment_method.png"
        )

        if os.path.exists(image_path):

            st.image(
                image_path,
                caption="Churn by Payment Method"
            )


    # --------------------------------------------------------
    # TENURE / CHARGES
    # --------------------------------------------------------

    col5, col6 = st.columns(2)


    with col5:

        image_path = (
            "data/eda/"
            "tenure_distribution.png"
        )

        if os.path.exists(image_path):

            st.image(
                image_path,
                caption="Tenure Distribution"
            )


    with col6:

        image_path = (
            "data/eda/"
            "tenure_vs_monthly_charges.png"
        )

        if os.path.exists(image_path):

            st.image(
                image_path,
                caption="Tenure vs Monthly Charges"
            )


# ============================================================
# TAB 4 — MODEL PERFORMANCE
# ============================================================

with tab4:

    st.header(
        "🤖 Machine Learning Model Performance"
    )


    st.write(
        """
        Comparison of the five machine-learning algorithms
        evaluated during model development.
        """
    )


    # --------------------------------------------------------
    # MODEL TABLE
    # --------------------------------------------------------

    st.dataframe(
        model_results.style.format(
            {
                "Accuracy": "{:.2%}",
                "Precision": "{:.2%}",
                "Recall": "{:.2%}",
                "F1-Score": "{:.2%}",
                "ROC-AUC": "{:.2%}"
            }
        ),
        width="stretch"
    )


    # --------------------------------------------------------
    # MODEL COMPARISON CHART
    # --------------------------------------------------------

    st.subheader(
        "Model Accuracy Comparison"
    )


    chart_data = model_results.set_index(
        "Model"
    )["Accuracy"]


    st.bar_chart(
        chart_data
    )


    # --------------------------------------------------------
    # ROC-AUC
    # --------------------------------------------------------

    st.subheader(
        "ROC-AUC Comparison"
    )


    roc_data = model_results.set_index(
        "Model"
    )["ROC-AUC"]


    st.bar_chart(
        roc_data
    )


    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    st.subheader(
        "Best Model Confusion Matrix"
    )


    confusion_path = (
        "data/model_results/"
        "best_model_confusion_matrix.png"
    )


    if os.path.exists(confusion_path):

        st.image(
            confusion_path,
            caption="Confusion Matrix"
        )


    # --------------------------------------------------------
    # ROC CURVE
    # --------------------------------------------------------

    st.subheader(
        "ROC Curve Comparison"
    )


    roc_curve_path = (
        "data/model_results/"
        "roc_curve_comparison.png"
    )


    if os.path.exists(roc_curve_path):

        st.image(
            roc_curve_path,
            caption="ROC Curve Comparison"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Customer Churn Prediction & Analytics System | "
    "Machine Learning + Python + Streamlit"
)