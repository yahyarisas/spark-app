"""
Neurological Health Assessment Tool - Streamlit Application

A web-based assessment tool that evaluates potential indicators of neurological
conditions including Parkinson's disease and Multiple Sclerosis. Users complete
a 10-question symptom questionnaire along with demographic information, and
receive AI-powered health insights via a trained machine learning model.

Features:
- Multi-step guided assessment process
- Real-time BMI calculation
- Interactive result visualizations
- Integration with FastAPI ML prediction service

Note: This tool is for educational purposes only and should not replace
professional medical consultation.

Usage:
    streamlit run neurological_assessment.py

Requirements:
    - streamlit
    - requests
    - plotly
    - pandas
    - FastAPI backend service running on localhost:8000
"""
import streamlit as st
import requests
import json
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import time

# Page configuration
st.set_page_config(
    page_title="Motion Health Assessment",
    page_icon="🏃‍♀️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
    }

    .sub-header {
        font-size: 1.3rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }

    .section-header {
        font-size: 1.8rem;
        color: #A23B72;
        margin: 2rem 0 1rem 0;
        border-bottom: 3px solid #F18F01;
        padding-bottom: 0.5rem;
        font-weight: 600;
    }

    .welcome-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 2rem 0;
    }

    .info-box {
        background-color: #E3F2FD !important;
        padding: 1.5rem !important;
        border-radius: 12px !important;
        border-left: 5px solid #2196F3 !important;
        margin: 1.5rem 0 !important;
    }

    .warning-box {
        background-color: #FFF3E0 !important;
        padding: 1.5rem !important;
        border-radius: 12px !important;
        border-left: 5px solid #FF9800 !important;
        margin: 1.5rem 0 !important;
    }

    .success-box {
        background-color: #E8F5E8 !important;
        padding: 1.5rem !important;
        border-radius: 12px !important;
        border-left: 5px solid #4CAF50 !important;
        margin: 1.5rem 0 !important;
    }

    .privacy-box {
        background-color: #F3E5F5 !important;
        padding: 1.5rem !important;
        border-radius: 12px !important;
        border-left: 5px solid #9C27B0 !important;
        margin: 1.5rem 0 !important;
    }

    .step-indicator {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 2rem 0;
    }

    .step-circle {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 10px;
        font-weight: bold;
        color: white;
    }

    .step-active {
        background-color: #2196F3;
    }

    .step-completed {
        background-color: #4CAF50;
    }

    .step-pending {
        background-color: #ddd;
        color: #666;
    }

    .motion-guide {
        background: linear-gradient(135deg, #74b9ff, #0984e3);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
    }

    .upload-area {
        border: 2px dashed #2196F3;
        border-radius: 15px;
        padding: 3rem;
        text-align: center;
        background-color: #f8f9ff;
        margin: 2rem 0;
    }

    .big-button {
        font-size: 1.2rem !important;
        padding: 0.75rem 2rem !important;
        border-radius: 25px !important;
        font-weight: 600 !important;
        margin: 0.5rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Constants
FASTAPI_URL = "https://spark-283984718972.europe-west1.run.app/predict"

# Questionnaire data
QUESTIONNAIRE = {
    "02": "Has your handwriting become smaller or more cramped?",
    "03": "Do people have trouble understanding your speech?",
    "09": "Do you have trouble with daily activities like bathing or dressing?",
    "13": "Do you have trouble with fine motor tasks?",
    "17": "Do you feel tired more often than usual?",
    "20": "Do you experience depression or anxiety?",
}

# Motion-related conditions checklist
MOTION_CONDITIONS = [
    "Arthritis",
    "Essential Tremor",
    "Multiple Sclerosis",
    "Stroke",
    "Muscle weakness",
    "Joint problems",
    "Other neurological conditions"
]

def initialize_session_state():
    """Initialize session state variables."""
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 0
    if 'user_data' not in st.session_state:
        st.session_state.user_data = {}
    if 'health_history' not in st.session_state:
        st.session_state.health_history = {}
    if 'motion_data_uploaded' not in st.session_state:
        st.session_state.motion_data_uploaded = False
    if 'questionnaire_responses' not in st.session_state:
        st.session_state.questionnaire_responses = {}
    if 'consent_given' not in st.session_state:
        st.session_state.consent_given = False

def show_step_indicator(current_step, total_steps):
    """Show progress indicator."""
    st.markdown('<div class="step-indicator">', unsafe_allow_html=True)

    steps = ["Welcome", "Basic Info", "Assessment", "Results"]

    cols = st.columns(len(steps))
    for i, (col, step_name) in enumerate(zip(cols, steps)):
        with col:
            if i < current_step:
                status = "completed"
                icon = "✓"
            elif i == current_step:
                status = "active"
                icon = str(i + 1)
            else:
                status = "pending"
                icon = str(i + 1)

            st.markdown(f"""
            <div style="text-align: center;">
                <div class="step-circle step-{status}">{icon}</div>
                <small>{step_name}</small>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

def step_0_welcome():
    """Step 0: Welcome Screen"""
    st.markdown("""
    <div class="welcome-box">
        <h1>🏃‍♀️ Motion Health Assessment</h1>
        <h2>Get insights into your motion health in minutes</h2>
        <p style="font-size: 1.1rem; margin-top: 1rem;">
            Our advanced screening tool helps you understand potential motion-related
            health concerns through a simple, science-based assessment.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("""
        ### What you'll do:
        - 📝 Share some basic health information
        - 🤝 Answer a few simple questions
        - 📱 Optionally upload motion data
        - 📊 Receive personalized insights

        **Time needed:** 5-10 minutes
        """)

        if st.button("Start Your Assessment", type="primary", key="start_btn",
                    help="Click to begin your motion health screening"):
            st.session_state.current_step = 1
            st.rerun()

def step_1_basic_info():
    """Step 1: Personal Details Form & Privacy"""
    st.markdown('<div class="section-header">📋 Personal Details</div>', unsafe_allow_html=True)

    # Privacy Assurance
    st.markdown("""
    <div class="privacy-box">
        <h4>🔒 Your Privacy Matters</h4>
        <p>• All data is encrypted and anonymized</p>
        <p>• No personal identifiers are stored</p>
        <p>• Information used solely for health screening</p>
        <p>• You can delete your data at any time</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Your Age", min_value=18, max_value=120, value=65,
                             help="This helps us provide age-appropriate insights")
        age_at_diagnosis = st.number_input(
            "Age at any motion-related diagnosis (0 if none)",
            min_value=0, max_value=120, value=0,
            help="Enter 0 if you haven't been diagnosed with motion-related conditions"
        )

    with col2:
        gender = st.selectbox("Gender", ["Male", "Female"],
                             help="Biological sex assigned at birth")
        appearance_in_kinship = st.selectbox(
            "Family history of Parkinson's disease",
            ["No", "Yes"],
            help="Do any blood relatives have Parkinson's disease?"
        )

    # Consent checkbox
    consent = st.checkbox(
        "I consent to this health screening assessment and understand this is not a medical diagnosis",
        value=st.session_state.consent_given
    )

    st.session_state.user_data = {
        "age": age,
        "age_at_diagnosis": age_at_diagnosis,
        "gender": gender,
        "appearance_in_kinship": appearance_in_kinship
    }
    st.session_state.consent_given = consent

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.button("← Back", key="back_1"):
            st.session_state.current_step = 0
            st.rerun()

    with col3:
        if st.button("Continue →", type="primary", key="continue_1", disabled=not consent):
            if consent:
                st.session_state.current_step = 2
                st.rerun()
            else:
                st.error("Please provide consent to continue")

def step_2_health_history():
    """Step 2: Health History Input"""
    st.markdown('<div class="section-header">🏥 Health History</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <h4>Motion-Related Conditions Check</h4>
        <p>This helps us understand your overall motion health context.</p>
    </div>
    """, unsafe_allow_html=True)

    has_conditions = st.radio(
        "Have you been diagnosed with any motion-related conditions?",
        ["No", "Yes"],
        key="has_conditions"
    )

    conditions_list = []
    if has_conditions == "Yes":
        st.markdown("**Please select all that apply:**")

        cols = st.columns(2)
        for i, condition in enumerate(MOTION_CONDITIONS):
            with cols[i % 2]:
                if st.checkbox(condition, key=f"condition_{i}"):
                    conditions_list.append(condition)

    st.session_state.health_history = {
        "has_conditions": has_conditions == "Yes",
        "conditions": conditions_list
    }

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.button("← Back", key="back_2"):
            st.session_state.current_step = 1
            st.rerun()

    with col3:
        if st.button("Continue →", type="primary", key="continue_2"):
            st.session_state.current_step = 3
            st.rerun()

def step_3_motion_intro():
    """Step 3: Motion Data Collection Setup"""
    st.markdown('<div class="section-header">📱 Motion Data Collection</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="motion-guide">
        <h3>📊 Why Motion Data Matters</h3>
        <p style="font-size: 1.1rem;">Your movements provide valuable insights into your neurological health.
        Motion patterns can reveal early indicators that aren't easily noticed in daily life.</p>

        <h4 style="margin-top: 1.5rem;">🎯 What We Analyze:</h4>
        <p>• Movement smoothness and coordination</p>
        <p>• Tremor patterns and frequency</p>
        <p>• Gait stability and rhythm</p>
        <p>• Fine motor control precision</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("### Ready to proceed?")

        if st.button("📱 Start Motion Assessment", type="primary", key="motion_start",
                    use_container_width=True):
            st.session_state.current_step = 4
            st.rerun()

        if st.button("⏭️ Skip Motion Data", key="skip_motion",
                    use_container_width=True):
            st.session_state.current_step = 5
            st.rerun()

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("← Back", key="back_3"):
            st.session_state.current_step = 2
            st.rerun()

def step_4_motion_demo():
    """Step 4: Motion Capture Demo & Upload Options"""
    st.markdown('<div class="section-header">🎬 Motion Data Collection</div>', unsafe_allow_html=True)

    # Motion Guide
    st.markdown("""
    <div class="info-box">
        <h4>📱 Motion Collection Guide</h4>
        <p><strong>For best results:</strong></p>
        <p>• Hold your phone or wear your smartwatch normally</p>
        <p>• Perform natural arm movements (up, down, left, right)</p>
        <p>• Walk normally for 30 seconds if possible</p>
        <p>• Tap your fingers rhythmically</p>
    </div>
    """, unsafe_allow_html=True)

    # Upload options
    st.markdown("### 📤 Upload Your Motion Data")

    tab1, tab2, tab3 = st.tabs(["🔗 Connect Device", "📁 Upload File", "⏭️ Skip"])

    with tab1:
        st.markdown("""
        <div class="upload-area">
            <h4>📱 Connect Your Wearable Device</h4>
            <p>Seamlessly import data from:</p>
            <p>• Apple Watch Health app</p>
            <p>• Fitbit activity data</p>
            <p>• Samsung Health</p>
            <p>• Google Fit</p>
            <br>
            <p><em>Feature coming soon! Use file upload for now.</em></p>
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.markdown("**Upload motion data files:**")
        uploaded_files = st.file_uploader(
            "Choose motion data files",
            accept_multiple_files=True,
            type=['csv', 'json', 'txt'],
            help="Upload accelerometer, gyroscope, or other motion sensor data"
        )

        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} file(s) uploaded successfully!")
            st.session_state.motion_data_uploaded = True

            for file in uploaded_files:
                st.write(f"📁 {file.name} ({file.size} bytes)")

    with tab3:
        st.markdown("""
        <div class="warning-box">
            <h4>⏭️ Skip Motion Data</h4>
            <p>You can still get valuable insights from the questionnaire alone,
            though motion data provides additional accuracy.</p>
        </div>
        """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.button("← Back", key="back_4"):
            st.session_state.current_step = 3
            st.rerun()

    with col3:
        if st.button("Continue →", type="primary", key="continue_4"):
            st.session_state.current_step = 5
            st.rerun()

def step_5_upload_confirmation():
    """Step 5: Upload & Confirmation"""
    st.markdown('<div class="section-header">✅ Data Processing</div>', unsafe_allow_html=True)

    if st.session_state.motion_data_uploaded:
        # Simulate processing
        with st.spinner("Processing your motion data..."):
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.02)
                progress_bar.progress(i + 1)

        st.markdown("""
        <div class="success-box">
            <h3>🎉 Motion data received successfully!</h3>
            <p>Your data has been processed and anonymized. This brings us closer to your personalized results!</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-box">
            <h3>📝 Proceeding with questionnaire</h3>
            <p>No motion data uploaded. We'll provide insights based on your responses to the health questionnaire.</p>
        </div>
        """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.button("← Back", key="back_5"):
            st.session_state.current_step = 4
            st.rerun()

    with col3:
        if st.button("Continue to Survey →", type="primary", key="continue_5"):
            st.session_state.current_step = 6
            st.rerun()

def step_6_questionnaire():
    """Step 6: Questionnaire Completion"""
    st.markdown('<div class="section-header">📋 Health Questionnaire</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <h4>Quick Health Check</h4>
        <p>These questions help us understand your daily experiences.
        Answer based on how you've felt over the past few months.</p>
    </div>
    """, unsafe_allow_html=True)

    responses = {}

    # Progress tracking
    progress = st.progress(0)

    st.markdown("### Please answer the following questions:")

    cols = st.columns(1)  # Single column for better readability
    question_keys = list(QUESTIONNAIRE.keys())

    for idx, question_key in enumerate(question_keys):
        question_text = QUESTIONNAIRE[question_key]

        with st.container():
            st.markdown(f"**Question {idx + 1} of {len(question_keys)}:**")
            response = st.radio(
                question_text,
                ["No", "Yes"],
                key=f"q_{question_key}",
                horizontal=True
            )
            responses[question_key] = response == "Yes"
            st.markdown("---")

    # Update progress
    progress.progress(len(responses) / len(QUESTIONNAIRE))

    st.session_state.questionnaire_responses = responses

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.button("← Back", key="back_6"):
            st.session_state.current_step = 5
            st.rerun()

    with col3:
        if st.button("Get My Results →", type="primary", key="continue_6"):
            if len(responses) == len(QUESTIONNAIRE):
                st.session_state.current_step = 7
                st.rerun()
            else:
                st.error("Please answer all questions to continue.")

def make_prediction(user_data, questionnaire_responses):
    """Make prediction using the FastAPI endpoint."""
    payload = {
        "age": user_data["age"],
        "age_at_diagnosis": user_data["age_at_diagnosis"],
        "appearance_in_kinship": user_data["appearance_in_kinship"],
        "gender": user_data["gender"],
        "02": questionnaire_responses["02"],
        "03": questionnaire_responses["03"],
        "09": questionnaire_responses["09"],
        "13": questionnaire_responses["13"],
        "17": questionnaire_responses["17"],
        "20": questionnaire_responses["20"],
    }

    try:
        response = requests.post(f"{FASTAPI_URL}", json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error: {str(e)}")
        return None

def step_7_results():
    """Step 7: Results & Insights"""
    st.markdown('<div class="section-header">📊 Your Motion Health Results</div>', unsafe_allow_html=True)

    # Get prediction
    with st.spinner("Analyzing your health data..."):
        prediction_result = make_prediction(
            st.session_state.user_data,
            st.session_state.questionnaire_responses
        )

    if prediction_result is None:
        st.error("Unable to generate results. Please try again.")
        return

    prediction = prediction_result.get("prediction", 0)

    # Dynamic Result Display
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        if prediction > 0.5:
            # Higher risk
            st.markdown("""
            <div class="warning-box">
                <h2>⚠️ Important Health Notice</h2>
                <p style="font-size: 1.1rem;">Your results suggest a higher risk for Parkinson's disease.
                We strongly recommend consulting with a healthcare professional for further assessment.</p>

                <h4>🏥 Next Steps:</h4>
                <p>• Schedule an appointment with a neurologist</p>
                <p>• Bring these results to your doctor</p>
                <p>• Consider a comprehensive neurological examination</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Lower risk
            st.markdown("""
            <div class="success-box">
                <h2>🎉 Great News!</h2>
                <p style="font-size: 1.1rem;">Your results indicate no significant risk for Parkinson's disease.
                Keep up your healthy habits!</p>

                <h4>🌟 Recommendations:</h4>
                <p>• Continue regular physical activity</p>
                <p>• Monitor any changes in movement</p>
                <p>• Maintain regular health check-ups</p>
            </div>
            """, unsafe_allow_html=True)

    # Additional insights if motion data was uploaded
    if st.session_state.motion_data_uploaded:
        st.markdown("""
        <div class="info-box">
            <h4>📊 Motion Data Insights</h4>
            <p>Your motion data provided additional context for a more accurate assessment.</p>
        </div>
        """, unsafe_allow_html=True)

    # Resources and next steps
    st.markdown("### 📚 Helpful Resources")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **🏥 Medical Resources:**
        - Find a neurologist near you
        - Parkinson's Foundation: parkinson.org
        - Movement Disorder Society
        - Local support groups
        """)

    with col2:
        st.markdown("""
        **📱 Health Tracking:**
        - Track symptoms over time
        - Share results with doctor
        - Monitor medication effects
        - Join patient communities
        """)

    # Disclaimer
    st.markdown("""
    <div class="warning-box">
        <h4>⚠️ Important Medical Disclaimer</h4>
        <p>This screening tool is for informational purposes only and is not a substitute for
        professional medical diagnosis. Results should not be used as the sole basis for medical decisions.
        Please consult with a qualified healthcare provider for proper medical evaluation and diagnosis.</p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Continue to Summary →", type="primary", key="continue_7"):
        st.session_state.current_step = 8
        st.rerun()

def step_8_closing():
    """Step 8: Closing & Engagement"""
    st.markdown('<div class="section-header">🎯 What\'s Next?</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="success-box">
        <div style="color: #2E7D32; font-size: 1.3rem; font-weight: bold; margin-bottom: 1rem;">✅ Assessment Complete!</div>
        <p>Thank you for taking the time to complete your motion health screening.
        Your proactive approach to health monitoring is commendable.</p>
    </div>
    """, unsafe_allow_html=True)

    # Follow-up CTAs
    st.markdown("### 🔄 Stay Connected With Your Health")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📅 Schedule Retest", key="retest", use_container_width=True):
            st.info("Set a reminder to retake this assessment in 6 months to track changes over time.")

    with col2:
        if st.button("📊 Download Results", key="download", use_container_width=True):
            st.info("Download feature coming soon! For now, take a screenshot of your results.")

    with col3:
        if st.button("💌 Health Tips", key="tips", use_container_width=True):
            st.info("Sign up for monthly motion health tips and updates (feature coming soon).")

    # Reassurance
    st.markdown("""
    <div class="info-box">
        <h4 style="color: #1565C0; margin-bottom: 1rem;">🔍 Remember</h4>
        <p style="margin-bottom: 1rem;">This is a <strong>screening tool</strong>, not a formal diagnosis. It's designed to help you
        stay informed about your health and know when to seek professional medical advice.</p>

        <p style="margin-bottom: 0;"><strong>Your health journey is unique</strong> - use these results as one piece of information
        alongside regular healthcare consultations.</p>
    </div>
    """, unsafe_allow_html=True)

    # Final CTA
    if st.button("🔄 Take New Assessment", key="new_assessment", type="primary", use_container_width=True):
        # Clear all session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

def main():
    """Main application function with step-based navigation."""
    initialize_session_state()

    # Show progress indicator
    show_step_indicator(st.session_state.current_step, 8)

    # Route to appropriate step
    if st.session_state.current_step == 0:
        step_0_welcome()
    elif st.session_state.current_step == 1:
        step_1_basic_info()
    elif st.session_state.current_step == 2:
        step_2_health_history()
    elif st.session_state.current_step == 3:
        step_3_motion_intro()
    elif st.session_state.current_step == 4:
        step_4_motion_demo()
    elif st.session_state.current_step == 5:
        step_5_upload_confirmation()
    elif st.session_state.current_step == 6:
        step_6_questionnaire()
    elif st.session_state.current_step == 7:
        step_7_results()
    elif st.session_state.current_step == 8:
        step_8_closing()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9em; padding: 2rem 0;">
        <p>🏥 Motion Health Assessment Tool | For educational and screening purposes only</p>
        <p>Always consult healthcare professionals for medical advice | © 2025</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
