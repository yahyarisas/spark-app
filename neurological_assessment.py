"""
Neurological Health Assessment Tool - Streamlit Application

A web-based assessment tool that evaluates potential indicators of neurological
conditions including Parkinson's disease and Multiple Sclerosis. Users complete
a 30-question symptom questionnaire along with demographic information, and
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

# Page configuration
st.set_page_config(
    page_title="Parkinson's Health Assessment",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 600;
    }

    .section-header {
        font-size: 1.5rem;
        color: #A23B72;
        margin: 1.5rem 0 1rem 0;
        border-bottom: 2px solid #F18F01;
        padding-bottom: 0.5rem;
    }

    .info-box {
        background-color: #E3F2FD !important;
        padding: 1rem !important;
        border-radius: 10px !important;
        border-left: 4px solid #2196F3 !important;
        margin: 1rem 0 !important;
    }

    .info-box h4 {
        color: #1976D2 !important;
        margin-bottom: 0.5rem !important;
    }

    .info-box p {
        margin-bottom: 0.5rem !important;
        line-height: 1.6 !important;
    }

    .info-box strong {
        color: #1565C0 !important;
        font-weight: 600 !important;
    }

    .warning-box {
        background-color: #FFF3E0 !important;
        padding: 1rem !important;
        border-radius: 10px !important;
        border-left: 4px solid #FF9800 !important;
        margin: 1rem 0 !important;
    }

    .success-box {
        background-color: #E8F5E8 !important;
        padding: 1rem !important;
        border-radius: 10px !important;
        border-left: 4px solid #4CAF50 !important;
        margin: 1rem 0 !important;
    }

    .metric-card {
        background: white !important;
        padding: 1rem !important;
        border-radius: 10px !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
        text-align: center !important;
        margin: 0.5rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Constants
FASTAPI_URL = "https://spark-283984718972.europe-west1.run.app/predict"

# Questionnaire data with actual question text.
QUESTIONNAIRE = {
    "01": "Do you have trouble getting up from a chair?",
    "02": "Has your handwriting become smaller or more cramped?",
    "03": "Do people have trouble understanding your speech?",
    "04": "Do you have trouble with buttons or shoelaces?",
    "05": "Do you drag your feet or take smaller steps when walking?",
    "06": "Do you have trouble with your balance?",
    "07": "Do you fall more often than you used to?",
    "08": "Do you have trouble turning in bed?",
    "09": "Do you have trouble with daily activities like bathing or dressing?",
    "10": "Do you have difficulty swallowing food or drink?",
    "11": "Do you drool during the day or at night?",
    "12": "Have you noticed a change in your voice?",
    "13": "Do you have trouble with fine motor tasks?",
    "14": "Do you have stiffness in your arms or legs?",
    "15": "Do you have tremor or shaking in your hands?",
    "16": "Do you have trouble with coordination?",
    "17": "Do you feel tired more often than usual?",
    "18": "Do you have trouble sleeping at night?",
    "19": "Do you have vivid dreams or nightmares?",
    "20": "Do you experience depression or anxiety?",
    "21": "Do you have trouble concentrating?",
    "22": "Do you have memory problems?",
    "23": "Do you have constipation?",
    "24": "Do you have a decreased sense of smell?",
    "25": "Do you have trouble controlling your bladder?",
    "26": "Do you have sexual problems?",
    "27": "Do you experience dizziness when standing up?",
    "28": "Do you have pain in your muscles or joints?",
    "29": "Do you sweat excessively?",
    "30": "Do you have restless legs at night?"
}

def initialize_session_state():
    """Initialize session state variables."""
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 0
    if 'user_data' not in st.session_state:
        st.session_state.user_data = {}
    if 'questionnaire_responses' not in st.session_state:
        st.session_state.questionnaire_responses = {}
    if 'assessment_complete' not in st.session_state:
        st.session_state.assessment_complete = False

def calculate_bmi(height, weight):
    """Calculate BMI from height (cm) and weight (kg)."""
    height_m = height / 100
    return weight / (height_m ** 2)

def collect_basic_info():
    """Collect basic demographic information."""
    st.markdown('<div class="section-header">📋 Basic Information</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=120, value=65, help="Your current age")
        age_at_diagnosis = st.number_input(
            "Age at diagnosis (if applicable)",
            min_value=0,
            max_value=120,
            value=0,
            help="Enter 0 if not diagnosed"
        )
        height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
        weight = st.number_input("Weight (kg)", min_value=30, max_value=300, value=70)

    with col2:
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        handedness = st.selectbox("Dominant hand", ["Right", "Left", "Ambidextrous"])
        appearance_in_kinship = st.selectbox(
            "Family history of Parkinson's",
            ["No", "Yes - Parent", "Yes - Sibling", "Yes - Other relative"]
        )
        subject_id = st.number_input("Patient ID (optional)", min_value=1, value=1)

    # Calculate BMI
    bmi = calculate_bmi(height, weight)

    # BMI status
    if bmi < 18.5:
        bmi_status = "Underweight"
        bmi_color = "#FFC107"
    elif bmi < 25:
        bmi_status = "Normal"
        bmi_color = "#4CAF50"
    elif bmi < 30:
        bmi_status = "Overweight"
        bmi_color = "#FF9800"
    else:
        bmi_status = "Obese"
        bmi_color = "#F44336"

    st.markdown(f"""
    <div class="info-box">
        <strong>Calculated BMI:</strong> {bmi:.1f} ({bmi_status})
    </div>
    """, unsafe_allow_html=True)

    return {
        "age": age,
        "age_at_diagnosis": age_at_diagnosis,
        "height": height,
        "weight": weight,
        "bmi": bmi,
        "gender": gender,
        "handedness": handedness,
        "appearance_in_kinship": appearance_in_kinship,
        "subject_id": subject_id
    }

def collect_questionnaire_responses():
    """Collect questionnaire responses with progress tracking."""
    st.markdown('<div class="section-header">📝 Health Assessment Questionnaire</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <strong>Instructions:</strong> Please answer the following questions based on your experiences
        over the past few months. Select "Yes" if you experience the symptom regularly,
        "No" if you rarely or never experience it.
    </div>
    """, unsafe_allow_html=True)

    responses = {}

    # Progress bar
    progress = st.progress(0)

    # Group questions for better organization
    question_groups = [
        (1, 10, "🚶 Movement and Mobility"),
        (11, 20, "🗣️ Speech and Communication"),
        (21, 30, "🧠 Cognitive and Other Symptoms")
    ]

    total_questions = len(QUESTIONNAIRE)
    answered_questions = 0

    for start, end, group_title in question_groups:
        with st.expander(group_title, expanded=True):
            cols = st.columns(2)
            col_idx = 0

            for i in range(start, end + 1):
                question_key = f"{i:02d}"
                question_text = QUESTIONNAIRE[question_key]

                with cols[col_idx]:
                    response = st.radio(
                        f"Q{i}: {question_text}",
                        ["No", "Yes"],
                        key=f"q_{question_key}",
                        horizontal=True
                    )
                    responses[question_key] = response == "Yes"
                    answered_questions += 1

                col_idx = 1 - col_idx  # Alternate between columns

    # Update progress
    progress.progress(answered_questions / total_questions)

    return responses

def make_prediction(user_data, questionnaire_responses):
    """Make prediction using the FastAPI endpoint."""
    # Combine all data
    payload = {**user_data, **questionnaire_responses}

    try:
        response = requests.post(f"{FASTAPI_URL}", json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error: Unable to connect to the prediction service. ({str(e)})")
        return None

def display_results(prediction_result):
    """Display prediction results in a user-friendly manner."""
    st.markdown('<div class="section-header">📊 Assessment Results</div>', unsafe_allow_html=True)

    if prediction_result is None:
        st.error("Unable to generate results. Please try again.")
        return

    if "error" in prediction_result:
        st.error(f"Error: {prediction_result['error']}")
        return

    prediction = prediction_result.get("prediction", 0)

    # Main result display
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        if prediction > 0.5:
            st.markdown("""
            <div class="warning-box">
                <h3>⚠️ Assessment Result</h3>
                <p>Based on your responses, there are indicators that suggest you may benefit
                from a consultation with a healthcare professional specializing in movement disorders.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="success-box">
                <h3>✅ Assessment Result</h3>
                <p>Based on your responses, the indicators for Parkinson's disease appear to be low.
                However, if you have concerns, it's always best to consult with a healthcare professional.</p>
            </div>
            """, unsafe_allow_html=True)

    # Probability visualization if available
    if "prob_class_0" in prediction_result and "prob_class_1" in prediction_result:
        prob_no = prediction_result["prob_class_0"]
        prob_yes = prediction_result["prob_class_1"]

        st.markdown("### Confidence Levels")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("No indicators", f"{prob_no:.1%}")
        with col2:
            st.metric("Indicators present", f"{prob_yes:.1%}")

        # Visualization
        fig = go.Figure(data=[
            go.Bar(
                x=['No Indicators', 'Indicators Present'],
                y=[prob_no, prob_yes],
                marker_color=['#4CAF50', '#FF9800'],
                text=[f'{prob_no:.1%}', f'{prob_yes:.1%}'],
                textposition='auto',
            )
        ])

        fig.update_layout(
            title="Assessment Confidence Levels",
            yaxis_title="Probability",
            showlegend=False,
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    # Important disclaimer
    st.markdown("""
    <div class="warning-box">
        <strong>⚠️ Important Medical Disclaimer:</strong>
        <br><br>
        This assessment tool is for informational purposes only and is not a substitute for
        professional medical diagnosis. The results should not be used as the sole basis for
        medical decisions. Please consult with a qualified healthcare provider for proper
        medical evaluation and diagnosis.
        <br><br>
        If you have concerns about Parkinson's disease or any neurological symptoms,
        please schedule an appointment with a neurologist or movement disorder specialist.
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main application function."""
    initialize_session_state()

    # Header
    st.markdown('<div class="main-header">🏥 Parkinson\'s Health Assessment Tool</div>', unsafe_allow_html=True)

    # Sidebar navigation
    st.sidebar.title("Assessment Progress")

    steps = ["Basic Information", "Health Questionnaire", "Results"]
    current_step = st.sidebar.radio("Navigation", steps, index=st.session_state.current_step)

    # Update current step
    st.session_state.current_step = steps.index(current_step)

    # Introduction
    if st.session_state.current_step == 0:

        st.info("""
        **Welcome to the Neurological Health Assessment Tool**

        This tool uses a comprehensive questionnaire to assess potential indicators
        of neurological conditions including Parkinson's disease and Multiple Sclerosis
        based on your symptoms and experiences. The assessment takes about 10-15 minutes to complete.

        **Please note:** This tool is for educational and informational
        purposes only and should not replace professional medical consultation.
        """)

        # Collect basic information
        user_data = collect_basic_info()
        st.session_state.user_data = user_data

        if st.button("Continue to Questionnaire", type="primary"):
            st.session_state.current_step = 1
            st.rerun()

    elif st.session_state.current_step == 1:
        # Show collected basic info summary
        if st.session_state.user_data:
            with st.expander("Basic Information Summary", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Age:** {st.session_state.user_data['age']}")
                    st.write(f"**Gender:** {st.session_state.user_data['gender']}")
                    st.write(f"**Height:** {st.session_state.user_data['height']} cm")
                with col2:
                    st.write(f"**Weight:** {st.session_state.user_data['weight']} kg")
                    st.write(f"**BMI:** {st.session_state.user_data['bmi']:.1f}")
                    st.write(f"**Handedness:** {st.session_state.user_data['handedness']}")

        # Collect questionnaire responses
        questionnaire_responses = collect_questionnaire_responses()
        st.session_state.questionnaire_responses = questionnaire_responses

        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            if st.button("← Back to Basic Info"):
                st.session_state.current_step = 0
                st.rerun()

        with col3:
            if st.button("Complete Assessment →", type="primary"):
                if len(questionnaire_responses) == 30:
                    st.session_state.current_step = 2
                    st.rerun()
                else:
                    st.error("Please answer all questions before proceeding.")

    elif st.session_state.current_step == 2:
        # Show assessment summary
        with st.expander("Assessment Summary", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Basic Information:**")
                for key, value in st.session_state.user_data.items():
                    if key not in ['subject_id']:
                        st.write(f"- {key.replace('_', ' ').title()}: {value}")

            with col2:
                positive_responses = sum(1 for v in st.session_state.questionnaire_responses.values() if v)
                st.write(f"**Questionnaire Responses:**")
                st.write(f"- Total questions answered: {len(st.session_state.questionnaire_responses)}")
                st.write(f"- Positive responses: {positive_responses}")
                st.write(f"- Negative responses: {30 - positive_responses}")

        # Make prediction
        with st.spinner("Processing your assessment..."):
            prediction_result = make_prediction(
                st.session_state.user_data,
                st.session_state.questionnaire_responses
            )

        # Display results
        display_results(prediction_result)

        # Option to start over
        if st.button("Start New Assessment", type="secondary"):
            # Clear session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9em;">
        <p>This assessment tool is developed for educational purposes.
        Always consult healthcare professionals for medical advice.</p>
        <p>© 2025 Neurological Health Assessment Tool</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
