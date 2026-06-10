import streamlit as pd_st # Common alias or standard import
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Set up page configurations
st.set_page_config(
    page_title="Student Performance Predictive Analytics",
    page_icon="🎓",
    layout="wide"
)

# -------------------------------------------------------------
# 1. LOAD TRAINED ARTIFACTS
# -------------------------------------------------------------
@st.cache_resource
def load_model_artifacts():
    """
    Loads and caches the model and preprocessor to ensure 
    fast page reloads and prevent redundant file I/O.
    """
    model_path = 'models/random_forest_model.pkl'
    preprocessor_path = 'models/preprocessor.pkl'
    
    if os.path.exists(model_path) and os.path.exists(preprocessor_path):
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        return model, preprocessor
    return None, None

model, preprocessor = load_model_artifacts()

# -------------------------------------------------------------
# 2. APP HEADER & LAYOUT
# -------------------------------------------------------------
st.title("🎓 Student Performance Early-Intervention Dashboard")
st.markdown("""
This predictive system assists educators in identifying **at-risk students** early in the academic term. 
Adjust the student parameters in the sidebar to compute real-time pass/fail probabilities.
""")

st.divider()

# Check if models are available before rendering dashboard inputs
if model is None or preprocessor is None:
    st.error("⚠️ Model artifacts not found! Please run `python src/train_model.py` first to generate the required serialization files.")
else:
    # -------------------------------------------------------------
    # 3. SIDEBAR - USER INPUT FEATURES
    # -------------------------------------------------------------
    st.sidebar.header("📊 Student Profile Inputs")
    
    st.sidebar.subheader("Academic Metrics")
    midterm_score = st.sidebar.slider("Midterm Exam Score", min_value=0, max_value=100, value=75, step=1)
    study_time_weekly = st.sidebar.slider("Weekly Study Hours", min_value=0, max_value=40, value=12, step=1)
    absences = st.sidebar.number_input("Number of Absences", min_value=0, max_value=50, value=3, step=1)
    
    st.sidebar.subheader("Institutional & Social Factors")
    school_support = st.sidebar.selectbox("Receiving Extra School Support?", options=['yes', 'no'], index=1)
    parent_engagement = st.sidebar.selectbox("Parental Engagement Level", options=['High', 'Medium', 'Low'], index=1)

    # Convert inputs into a DataFrame matching original feature structure
    input_data = pd.DataFrame([{
        'school_support': school_support,
        'parent_engagement': parent_engagement,
        'study_time_weekly': study_time_weekly,
        'absences': absences,
        'midterm_score': midterm_score
    }])

    # -------------------------------------------------------------
    # 4. PREDICTION PIPELINE EXECUTION
    # -------------------------------------------------------------
    # Transform raw inputs using the standard deviations/rules learned during training
    input_transformed = preprocessor.transform(input_data)
    
    # Generate predictions
    prediction = model.predict(input_transformed)[0]
    prediction_proba = model.predict_proba(input_transformed)[0]
    
    pass_probability = prediction_proba[1]
    fail_probability = prediction_proba[0]

    # -------------------------------------------------------------
    # 5. VISUALIZING RESULTS (MAIN PANEL)
    # -------------------------------------------------------------
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🔮 Prediction Status")
        if prediction == 1:
            st.success("🟢 STATUS: PASSING")
            st.metric(label="Success Probability", value=f"{pass_probability:.1%}")
        else:
            st.error("🔴 STATUS: AT-RISK (FAIL)")
            st.metric(label="Risk Probability", value=f"{fail_probability:.1%}")
            
    with col2:
        st.subheader("📉 Probability Breakdown")
        # Build a simple visualization metric bar using native Streamlit tools
        chart_data = pd.DataFrame({
            'Outcome': ['At-Risk (Fail)', 'Passing'],
            'Probability': [fail_probability, pass_probability]
        })
        st.bar_chart(data=chart_data, x='Outcome', y='Probability', color='Outcome')

    st.divider()

    # -------------------------------------------------------------
    # 6. CONTEXTUAL EDUCATIONAL INSIGHTS
    # -------------------------------------------------------------
    st.subheader("💡 Actionable Intervention Guidelines")
    
    # Conditional logic helping teachers translate model statistics into human tasks
    if prediction == 0:
        st.warning("""
        **Recommended Interventions for this Student:**
        *   **Attendance Review:** The student has accrued a notable volume of absences. Consider scheduling a short check-in meeting to identify structural external hurdles.
        *   **Peer Tutoring:** Given the current mid-term performance trends, enrolling this individual in structured peer study cohorts could alter their final curve trajectory.
        """)
    elif prediction == 1 and absences > 8:
        st.info("""
        **Borderline Warning:** While currently classified as passing due to reasonable exam inputs, the student's high **absence count** makes them vulnerable to a sudden drop. Monitor upcoming evaluations closely.
        """)
    else:
        st.success("✨ Student exhibits stable parameters. Keep encouragement consistent; no direct preventative intervention required.")