import streamlit as st
import pandas as pd
import joblib

# --- App Layout ---
# Moved this to the top, right after imports. This must be the first Streamlit command.
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

# --- Load Model and Preprocessors ---
# Use st.cache_resource to load these only once.
@st.cache_resource
def load_assets():
    """Loads and caches the ML model and preprocessing objects."""
    try:
        model = joblib.load("churn_model.pkl")
        scaler = joblib.load("scaler.pkl")
        label_encoders = joblib.load("label_encoders.pkl")
        model_columns = joblib.load("model_columns.pkl")
        return model, scaler, label_encoders, model_columns
    except FileNotFoundError:
        return None, None, None, None

model, scaler, label_encoders, model_columns = load_assets()

# --- Page Title ---
st.title("Customer Churn Prediction 🔮")
st.markdown("This app predicts whether a customer is likely to churn based on their attributes.")

# Check if model and assets loaded correctly
if not all([model, scaler, label_encoders, model_columns]):
    st.error("🔴 Critical Error: Model or preprocessing files are missing. Please run the `main.ipynb` notebook to generate them.")
else:
    # --- User Input Sidebar ---
    st.sidebar.header("User Input Features")

    def user_input_features():
        """Creates sidebar widgets and returns a DataFrame of user inputs."""
        age = st.sidebar.slider("Age", 18, 100, 30)
        gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
        tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
        usage_frequency = st.sidebar.slider("Usage Frequency (Days/Month)", 0, 30, 15)
        support_calls = st.sidebar.slider("Support Calls", 0, 10, 2)
        payment_delay = st.sidebar.slider("Payment Delay (Days)", 0, 30, 5)
        subs_type = st.sidebar.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
        cont_length = st.sidebar.selectbox("Contract Length", ["Monthly", "Quarterly", "Annual"])
        tot_spend = st.sidebar.number_input("Total Spend ($)", min_value=0.0, max_value=1000.0, value=500.0, step=10.0)
        last_inter = st.sidebar.slider("Last Interaction (Days Ago)", 0, 30, 10)

        data = {
            'Age': age,
            'Gender': gender,
            'Tenure': tenure,
            'Usage Frequency': usage_frequency,
            'Support Calls': support_calls,
            'Payment Delay': payment_delay,
            'Subscription Type': subs_type,
            'Contract Length': cont_length,
            'Total Spend': tot_spend,
            'Last Interaction': last_inter
        }
        features = pd.DataFrame(data, index=[0])
        return features

    input_df = user_input_features()

    st.subheader('User Input')
    st.write(input_df)

    # --- Prediction Logic ---
    if st.sidebar.button("Predict Churn"):
        # Create a copy for preprocessing
        df_processed = input_df.copy()

        # Apply the loaded label encoders
        for col, le in label_encoders.items():
            df_processed[col] = le.transform(df_processed[col])
            
        # Reorder columns to match the model's training order
        df_processed = df_processed[model_columns]

        # Apply the loaded scaler
        scaled_features = scaler.transform(df_processed)

        # Make prediction and get probability
        prediction = model.predict(scaled_features)
        prediction_proba = model.predict_proba(scaled_features)
        churn_probability = prediction_proba[0][1]

        # --- Display Result ---
        st.subheader("Prediction Result")
        
        if prediction[0] == 1:
            st.error(f"🔴 High Risk: Customer is likely to churn (Probability: {churn_probability:.2%})")
        else:
            st.success(f"🟢 Low Risk: Customer is likely to stay (Probability of Churn: {churn_probability:.2%})")