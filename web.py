import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image
import base64
from io import BytesIO
import shap  

import numpy as np
if not hasattr(np, 'bool'):
    np.bool = bool

# Set page title and layout
st.set_page_config(
    page_title="Stroke-Associated Pneumonia Prediction System",
    page_icon="🏥",
    layout="wide"
)

# Load model only, not the explainer
@st.cache_resource
def load_model():
    try:
        model = joblib.load('lgbm_model.joblib')
    except:
        model = joblib.load('lgbm_model.pkl')
    return model

# Load feature names and English translations
feature_names = [
    'gender', 'los', 'resp_rate_mean', 'temperature_mean', 'spo2_mean', 
    'wbc', 'rdw', 'potassium', 'calcium', 'bun', 'mchc', 'mcv', 'inr',
    'hypertension', 'diabetes', 'myocardial_infarct', 'malignant_cancer',
    'severe_liver_disease', 'aids', 'dvt', 'sedatives', 'sirs', 'gcs_min',
    'antibiotic', 'steroid', 'positive_respiratory_culture'
]

feature_names_en = [
    'Gender', 'Length of Stay', 'Mean Respiratory Rate', 'Mean Temperature', 'Mean Oxygen Saturation',
    'White Blood Cell Count', 'Red Cell Distribution Width', 'Potassium', 'Calcium', 'Blood Urea Nitrogen',
    'Mean Corpuscular Hemoglobin Concentration', 'Mean Corpuscular Volume', 'International Normalized Ratio',
    'Hypertension', 'Diabetes', 'Myocardial Infarction', 'Malignant Cancer',
    'Severe Liver Disease', 'AIDS', 'Deep Vein Thrombosis', 'Sedatives Use', 'SIRS', 'Minimum GCS Score',
    'Antibiotic Use', 'Steroid Use', 'Positive Respiratory Culture'
]

feature_dict = dict(zip(feature_names, feature_names_en))

# Variable description dictionary
variable_descriptions = {
    'gender': 'Patient gender (0=Female, 1=Male)',
    'los': 'Length of hospital stay in days',
    'resp_rate_mean': 'Mean respiratory rate, normal range 12-20 breaths/min',
    'temperature_mean': 'Mean body temperature, normal range 36.5-37.5°C',
    'spo2_mean': 'Mean oxygen saturation, normal value ≥95%',
    'wbc': 'White blood cell count, assesses infection and inflammation, normal range 4-10×10^9/L',
    'rdw': 'Red cell distribution width, assesses red blood cell size variation, normal range 11.5-14.5%',
    'potassium': 'Potassium level, normal range 3.5-5.0 mmol/L',
    'calcium': 'Calcium level, normal range 8.5-10.5 mg/dL',
    'bun': 'Blood urea nitrogen, assesses kidney function, normal range 7-20 mg/dL',
    'mchc': 'Mean corpuscular hemoglobin concentration, normal range 32-36 g/dL',
    'mcv': 'Mean corpuscular volume, normal range 80-100 fL',
    'inr': 'International normalized ratio, normal range 0.8-1.2',
    'hypertension': 'Indicates whether the patient has hypertension (0=No, 1=Yes)',
    'diabetes': 'Indicates whether the patient has diabetes (0=No, 1=Yes)',
    'myocardial_infarct': 'Indicates whether the patient has had a myocardial infarction (0=No, 1=Yes)',
    'malignant_cancer': 'Indicates whether the patient has malignant cancer (0=No, 1=Yes)',
    'severe_liver_disease': 'Indicates whether the patient has severe liver disease (0=No, 1=Yes)',
    'aids': 'Indicates whether the patient has AIDS (0=No, 1=Yes)',
    'dvt': 'Indicates whether the patient has deep vein thrombosis (0=No, 1=Yes)',
    'sedatives': 'Indicates whether the patient is using sedatives (0=No, 1=Yes)',
    'sirs': 'Systemic inflammatory response syndrome score (0-4)',
    'gcs_min': 'Minimum Glasgow Coma Scale score (3-15)',
    'antibiotic': 'Indicates whether the patient is using antibiotics (0=No, 1=Yes)',
    'steroid': 'Indicates whether the patient is using steroids (0=No, 1=Yes)',
    'positive_respiratory_culture': 'Indicates whether the patient has a positive respiratory culture (0=No, 1=Yes)'
}

# Main application
def main():
    # Sidebar title
    st.sidebar.title("Stroke-Associated Pneumonia Prediction System")
    
    # Load and display logo
    try:
    # 使用相对路径，适合GitHub部署
        logo_image = Image.open("脑卒中预测系统.png")  
        st.sidebar.image(logo_image, width=200)
    except:
        st.sidebar.image("https://img.freepik.com/free-vector/hospital-logo-design-vector-medical-cross_53876-136743.jpg", width=200)
    
    # Add system description to sidebar
    st.sidebar.markdown("""
    # System Description

    ## About This System
    This is a Stroke-Associated Pneumonia (SAP) prediction system based on LightGBM algorithm, which predicts pneumonia risk in stroke patients by analyzing clinical indicators.

    ## Prediction Results
    The system predicts:
    - Probability of developing pneumonia
    - Probability of not developing pneumonia
    - Risk assessment (low, medium, high risk)

    ## How to Use
    1. Fill in patient clinical indicators in the main interface
    2. Click the prediction button to generate prediction results
    3. View prediction results and feature importance analysis

    ## Important Notes
    - Please ensure accurate patient information input
    - All fields need to be filled
    - Numeric fields require number input
    - Selection fields require choosing from options
    """)
    
    # Add variable descriptions to sidebar
    with st.sidebar.expander("Variable Descriptions"):
        for feature in feature_names:
            st.markdown(f"**{feature_dict[feature]}**: {variable_descriptions[feature]}")
    
    # Main page title
    st.title("Stroke-Associated Pneumonia Prediction System")
    st.markdown("### Based on LightGBM Model")
    
    # Load model
    try:
        model = load_model()
        st.sidebar.success("Model loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Model loading failed: {e}")
        return
    
    # Create input form
    st.sidebar.header("Patient Information Input")
    
    # Create two-column layout for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Basic Information")
        gender = st.selectbox(f"{feature_dict['gender']}", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        los = st.number_input(f"{feature_dict['los']} (days)", min_value=0.1, max_value=50.0, value=3.0, step=0.1)
        resp_rate_mean = st.number_input(f"{feature_dict['resp_rate_mean']} (breaths/min)", min_value=8.0, max_value=40.0, value=18.0, step=0.1)
        temperature_mean = st.number_input(f"{feature_dict['temperature_mean']} (°C)", min_value=35.0, max_value=40.0, value=36.8, step=0.1)
        spo2_mean = st.number_input(f"{feature_dict['spo2_mean']} (%)", min_value=70.0, max_value=100.0, value=96.0, step=0.1)
        wbc = st.number_input(f"{feature_dict['wbc']} (×10^9/L)", min_value=0.1, max_value=50.0, value=10.0, step=0.1)
        rdw = st.number_input(f"{feature_dict['rdw']} (%)", min_value=10.0, max_value=25.0, value=14.0, step=0.1)
        potassium = st.number_input(f"{feature_dict['potassium']} (mmol/L)", min_value=2.0, max_value=7.0, value=4.0, step=0.1)
        calcium = st.number_input(f"{feature_dict['calcium']} (mg/dL)", min_value=6.0, max_value=12.0, value=9.0, step=0.1)
        bun = st.number_input(f"{feature_dict['bun']} (mg/dL)", min_value=5, max_value=150, value=20)
        mchc = st.number_input(f"{feature_dict['mchc']} (g/dL)", min_value=25.0, max_value=40.0, value=33.0, step=0.1)
        mcv = st.number_input(f"{feature_dict['mcv']} (fL)", min_value=60, max_value=120, value=90)
        inr = st.number_input(f"{feature_dict['inr']}", min_value=0.8, max_value=5.0, value=1.2, step=0.1)
    
    with col2:
        st.subheader("Medical Conditions & Treatments")
        hypertension = st.selectbox(f"{feature_dict['hypertension']}", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        diabetes = st.selectbox(f"{feature_dict['diabetes']}", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        myocardial_infarct = st.selectbox(f"{feature_dict['myocardial_infarct']}", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        malignant_cancer = st.selectbox(f"{feature_dict['malignant_cancer']}", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        severe_liver_disease = st.selectbox(f"{feature_dict['severe_liver_disease']}", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        aids = st.selectbox(f"{feature_dict['aids']}", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        dvt = st.selectbox(f"{feature_dict['dvt']}", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        sedatives = st.selectbox(f"{feature_dict['sedatives']}", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        sirs = st.selectbox(f"{feature_dict['sirs']}", options=[0, 1, 2, 3, 4])
        # 将滑块改为数字输入框
        gcs_min = st.number_input(f"{feature_dict['gcs_min']}", min_value=3, max_value=15, value=14)
        antibiotic = st.selectbox(f"{feature_dict['antibiotic']}", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        steroid = st.selectbox(f"{feature_dict['steroid']}", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        positive_respiratory_culture = st.selectbox(f"{feature_dict['positive_respiratory_culture']}", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    
    # Create prediction button
    predict_button = st.button("Predict Pneumonia Risk")
    
    if predict_button:
        # Collect all input features
        features = [gender, los, resp_rate_mean, temperature_mean, spo2_mean, 
                   wbc, rdw, potassium, calcium, bun, mchc, mcv, inr,
                   hypertension, diabetes, myocardial_infarct, malignant_cancer,
                   severe_liver_disease, aids, dvt, sedatives, sirs, gcs_min,
                   antibiotic, steroid, positive_respiratory_culture]
        
        # Convert to DataFrame
        input_df = pd.DataFrame([features], columns=feature_names)
        
        # Make prediction
        prediction = model.predict_proba(input_df)[0]
        no_pneumonia_prob = prediction[0]
        pneumonia_prob = prediction[1]
        
        # Display prediction results
        st.header("Prediction Results")
        
        # Use progress bars to display probabilities
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("No Pneumonia Probability")
            st.progress(float(no_pneumonia_prob))
            st.write(f"{no_pneumonia_prob:.2%}")
        
        with col2:
            st.subheader("Pneumonia Probability")
            st.progress(float(pneumonia_prob))
            st.write(f"{pneumonia_prob:.2%}")
        
        # Risk assessment
        risk_level = "Low Risk" if pneumonia_prob < 0.3 else "Medium Risk" if pneumonia_prob < 0.6 else "High Risk"
        risk_color = "green" if pneumonia_prob < 0.3 else "orange" if pneumonia_prob < 0.6 else "red"
        
        st.markdown(f"### Risk Assessment: <span style='color:{risk_color}'>{risk_level}</span>", unsafe_allow_html=True)
        
        # Clinical recommendations
        st.header("Clinical Recommendations")
        st.write("Based on the model prediction, the following clinical recommendations are provided:")
        
        if pneumonia_prob > 0.5:
            st.warning("This patient has a high risk of developing stroke-associated pneumonia. Close monitoring and preventive measures are recommended.")
            st.markdown("""
            **Recommended preventive measures:**
            - Regular oral care and swallowing assessment
            - Elevation of the head of the bed to 30-45 degrees
            - Consider early mobilization if appropriate
            - Monitor respiratory status closely
            - Consider prophylactic antibiotics based on clinical judgment
            """)
        else:
            st.success("This patient has a relatively low risk of developing stroke-associated pneumonia. Standard care protocol is recommended.")
            st.markdown("""
            **Standard care recommendations:**
            - Regular oral care
            - Routine swallowing assessment
            - Standard positioning protocols
            - Regular monitoring of vital signs
            """)
        
        # Add SHAP value explanation
        st.write("---")
        st.subheader("Model Interpretation")
        
        try:
            # Calculate SHAP values
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_df)

            if isinstance(shap_values, list) and len(shap_values) > 1:
                base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
                shap_value = shap_values[1][0]
            else:
                base_value = explainer.expected_value
                shap_value = shap_values[0]

            # Fix base_value and shap_value structures
            if isinstance(base_value, (list, np.ndarray)):
                base_value = base_value[0]
            if isinstance(shap_value, np.ndarray) and shap_value.ndim == 2 and shap_value.shape[0] == 1:
                shap_value = shap_value.flatten()

            shap_html = shap.force_plot(
                base_value, 
                shap_value, 
                input_df.iloc[0],
                matplotlib=False
            )
            st.components.v1.html(shap.getjs() + shap_html.html(), height=200, scrolling=True)

            # Display feature importance explanation
            st.write("---")
            st.subheader("Feature Contribution Analysis")
            
            # Create a DataFrame with feature names and their SHAP values
            feature_importance = pd.DataFrame({
                'Feature': [feature_dict[f] for f in input_df.columns],
                'SHAP Value': np.abs(shap_value)
            }).sort_values('SHAP Value', ascending=False)
            
            # Display as a table
            st.table(feature_importance)

        except Exception as e:
            st.error(f"Unable to generate SHAP explanation: {str(e)}")
            st.info("Using model's feature importance as an alternative")
            
            # If SHAP fails, use model's feature_importances_ if available
            if hasattr(model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'Feature': [feature_dict[f] for f in input_df.columns],
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                st.table(feature_importance)
            else:
                st.warning("Feature importance information is not available for this model.")

if __name__ == "__main__":
    main()