import streamlit as st
import pandas as pd
import pickle

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="centered"
)

# ---------------- CSS Styling ----------------
st.markdown("""
<style>

/* App background */
.stApp {
    background: linear-gradient(135deg, #eef3f9, #ffffff);
    font-family: 'Segoe UI', sans-serif;
}

/* Title */
.title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    color: #0d47a1;
    margin-bottom: 10px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 20px;
    color: #333333;
    margin-bottom: 30px;
}

/* Section headers */
.section {
    font-size: 26px;
    font-weight: 700;
    color: #1f4e79;
    margin-top: 30px;
    margin-bottom: 15px;
    border-bottom: 2px solid #0d47a1;
    padding-bottom: 6px;
}

/* Markdown text */
div.stMarkdown {
    font-size: 18px;
    color: #000000;
    line-height: 1.6;
}

/* Buttons */
.stButton>button {
    background-color: #0d47a1;
    color: white;
    font-size: 20px;
    padding: 12px 24px;
    border-radius: 8px;
    border: none;
    font-weight: 600;
    width: 100%;
}

.stButton>button:hover {
    background-color: #08306b;
}

/* Alerts */
.stAlert {
    font-size: 18px;
    font-weight: 600;
}

/* Expander headers and content on mobile */
@media (max-width: 768px) {

    .title {
        font-size: 30px;
    }

    .subtitle {
        font-size: 16px;
    }

    .section {
        font-size: 22px;
    }

    div.stMarkdown {
        font-size: 16px !important;
        line-height: 1.5 !important;
        padding: 0 10px;
    }

    .stExpander {
        font-size: 18px !important;
        font-weight: 600 !important;
        margin-bottom: 5px !important;
    }

    .stExpander div.stMarkdown {
        font-size: 16px !important;
        line-height: 1.5 !important;
    }

    label {
        font-size: 16px !important;
        color: #000000 !important;
        font-weight: 600;
    }

    input, select, textarea {
        font-size: 16px !important;
        color: #000000 !important;
        background-color: #ffffff !important;
    }
}

</style>
""", unsafe_allow_html=True)

# ---------------- Load Model & Encoders ----------------
with open("customer_churn_model.pkl", "rb") as f:
    model_data = pickle.load(f)
    model = model_data["model"]
    feature_names = model_data["features_names"]

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# ---------------- Title ----------------
st.markdown("<div class='title'>Customer Churn Prediction</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Predict the likelihood of a customer leaving a telecom service</div>",
    unsafe_allow_html=True
)

# ---------------- About Application ----------------
st.markdown("<div class='section'>About This Application</div>", unsafe_allow_html=True)
st.markdown("""
Customer churn happens when a customer stops using a company's service. In telecom, this may occur due to high monthly charges, poor support, or better offers from competitors. Predicting churn helps companies retain customers and reduce revenue loss.  

In this project, we built a machine learning system to predict the likelihood of a customer leaving a telecom service. The dataset was cleaned and preprocessed, class imbalance handled, models trained, and a Streamlit app created for real-time churn predictions.
""")

# ---------------- Customer Churn Info ----------------
with st.expander("What is Customer Churn?"):
    st.markdown("""
Customer churn happens when a customer **stops using a company's service**.  
                
In telecom, customers may leave due to: 
- Short-term or flexible contracts  
- High monthly charges or hidden fees  
- Poor service quality or customer support  
- Better offers from other providers  
                
In this system:  
- **Churn = 1** → Customer is likely to leave (high risk)  
- **Churn = 0** → Customer is likely to stay (low risk)  
""")

# ---------------- Project Work Done ----------------
with st.expander("What Has Been Done in This Project?"):
    st.markdown("""
• Cleaned and preprocessed the telecom dataset, encoding categorical features using **pandas** and **scikit-learn**  
• Analyzed customer patterns and churn trends with **matplotlib** and **seaborn**  
• Handled class imbalance using **SMOTE** (**imblearn**) and trained models like **Random Forest**, **XGBoost**, and **Decision Tree**  
• Built an interactive **Streamlit** app for real-time churn prediction  
• Validated model performance using **scikit-learn** metrics, enabling data-driven business decisions

""")

# ---------------- Input Section ----------------
st.markdown("<div class='section'>Customer Information</div>", unsafe_allow_html=True)
st.markdown("Provide the customer details below to predict the likelihood of churn.")

# ---------------- User Inputs ----------------
gender = st.selectbox("Gender (Male or Female)", ["Male", "Female"])
contract = st.selectbox("Contract Type (Month-to-month / One year / Two year)", ["Month-to-month", "One year", "Two year"])
tenure = st.slider("Tenure (Number of months with the company)", 0, 72, 6)
monthly_charges = st.number_input("Monthly Charges (Amount customer pays per month)", 0.0, 200.0, 70.0)
internet = st.selectbox("Internet Service (DSL / Fiber optic / No)", ["DSL", "Fiber optic", "No"])
payment = st.selectbox(
    "Payment Method (How the customer pays the bill)", 
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)

# ---------------- Prepare Data ----------------
input_data = {
    "gender": gender,
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": tenure,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": internet,
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": contract,
    "PaperlessBilling": "Yes",
    "PaymentMethod": payment,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": tenure * monthly_charges
}

input_df = pd.DataFrame([input_data])

# Encode categorical features
for col, encoder in encoders.items():
    input_df[col] = encoder.transform(input_df[col])

input_df = input_df[feature_names]

# ---------------- Prediction ----------------
st.markdown("<div class='section'>Prediction Result</div>", unsafe_allow_html=True)
st.markdown("Click the **Predict Churn** button to see if the customer is at risk of leaving.")

if st.button("Predict Churn"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)

    if prediction == 1:
        st.error(
            f"High Risk of Customer Churn\n\n"
            f"Churn Probability: {probability[0][1]*100:.2f}%"
        )
    else:
        st.success(
            f"Low Risk of Customer Churn\n\n"
            f"No-Churn Probability: {probability[0][0]*100:.2f}%"
        )
