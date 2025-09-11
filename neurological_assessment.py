"""
Neurological Health Assessment Tool - Streamlit Application (Optimized)

A web-based assessment tool that evaluates potential indicators of neurological
conditions including Parkinson's disease and Multiple Sclerosis. Users complete
a streamlined assessment process with optional user ID lookup for existing data.

Features:
- Streamlined 4-step assessment process
- Optional user ID lookup for existing data
- Real-time questionnaire pre-population
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



csv_path = "./data/questionnaire_test.csv"

def load_questionnaire_data():
    df = pd.read_csv("data/questionnaire_test.csv")
    df['id_index'] = df['Unnamed: 0']
    return df

questionnaire_df = load_questionnaire_data()

# Page configuration
st.set_page_config(
    page_title="Personal Motor Health Assessment",
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
        padding: 1.5rem !important;
        border-radius: 12px !important;
        border-left: 5px solid #2196F3 !important;
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

    .questionnaire-section {
        background-color: #f8f9ff;
        padding: 2rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        border: 2px solid #e3f2fd;
    }
    
    .metric-card {
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        font-size: 1.1rem;
        font-weight: 500;
        background: #f8f9ff;
        border: 2px solid #e3f2fd;
        overflow: visible;
        word-break: break-word;
    }
    .healthy-card { background: #e8f5e9; border-left: 6px solid #43a047; }
    .parkinson-card { background: #ffebee; border-left: 6px solid #e53935; }
    .motor-card { background: #fffde7; border-left: 6px solid #fbc02d; }

    .result-title {
        font-size: 1.8rem;
        color: #A23B72;
        font-weight: 600;
        margin-bottom: 0.5rem;
        margin-top: 0;
    }
    

    .big-button {
        font-size: 1.2rem !important;
        padding: 0.75rem 2rem !important;
        border-radius: 25px !important;
        font-weight: 600 !important;
        margin: 0.5rem !important;
    }
    
    .disclaimer-box {
        background-color: #f5f5f5 !important;
        border-left: 3px solid #bdbdbd !important;
        color: #555 !important;
        font-size: 0.95rem !important;
        padding: 0.7rem 1rem !important;
        margin: 1.5rem 0 0.5rem 0 !important;
        border-radius: 7px !important;
    }
      
    
</style>
""", unsafe_allow_html=True)


# Constants
FASTAPI_URL = "https://spark-283984718972.europe-west1.run.app/predict_by_qn"
FASTAPI_URL_USER = 'https://spark-283984718972.europe-west1.run.app/predict_by_user_id'

# Complete questionnaire data (all 10 questions)
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
    "Other conditions"
]

# Sample user data (you can replace this with actual dataset later)
SAMPLE_USER_DATA = {
    "user001": {
        "age": 65,
        "gender": "Male",
        "appearance_in_kinship": "Yes",
        "age_at_diagnosis": 62,
        "responses": {
            "02": True, "03": False, "04": True, "05": False, "06": True,
            "07": False, "08": True, "09": False, "10": False, "11": True,
            "12": False, "13": True, "14": True, "15": False, "16": True,
            "17": True, "18": False, "19": False, "20": True
        }
    },
    "user002": {
        "age": 45,
        "gender": "Female",
        "appearance_in_kinship": "No",
        "age_at_diagnosis": 0,
        "responses": {
            "02": False, "03": False, "04": False, "05": True, "06": False,
            "07": False, "08": False, "09": False, "10": False, "11": False,
            "12": False, "13": False, "14": False, "15": False, "16": False,
            "17": True, "18": False, "19": False, "20": False
        }
    }
}

def initialize_session_state():
    """Initialize session state variables."""
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 0
    if 'user_data' not in st.session_state:
        st.session_state.user_data = {}
    if 'questionnaire_responses' not in st.session_state:
        st.session_state.questionnaire_responses = {}
    if 'consent_given' not in st.session_state:
        st.session_state.consent_given = False
    if 'user_id' not in st.session_state:
        st.session_state.user_id = ""
    if 'existing_user_data' not in st.session_state:
        st.session_state.existing_user_data = None
    if "id_index" not in st.session_state:
        st.session_state.id_index = None  # FIXED: Initialize as None instead of undefined


def show_step_indicator(current_step, total_steps):
    """Show progress indicator."""
    st.markdown('<div class="step-indicator">', unsafe_allow_html=True)

    steps = ["Welcome", "Basic info", "Assessment", "Results"]

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

            # Use flexbox to align the circle and the text
            st.markdown(f"""
            <div style="display: flex; flex-direction: column; align-items: center; justify-content: flex-start;">
                <div class="step-circle step-{status}">{icon}</div>
                <small style="margin-top: 0.5rem; text-align: center;">{step_name}</small>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

def step_0_welcome():
    """Step 0: Welcome Screen"""
    st.markdown("""
    <div class="welcome-box">
        <h1>🏃‍♀️ Motor Health Assessment</h1>
        <h2>Get insights into your motor health in minutes</h2>
        <p style="font-size: 1.1rem; margin-top: 1rem;">
            Our advanced screening tool helps you understand potential motor-related
            health concerns through a simple, science-based assessment.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("""
        ### What you'll do:
        - 📝 Share some basic health information
        - 🤝 Answer a comprehensive health questionnaire
        - 📊 Receive personalized insights

        **Time needed:** 5-10 minutes
        """)

        if st.button("Start Your Assessment", type="primary", key="start_btn",
                    help="Click to begin your motor health screening"):
            st.session_state.current_step = 1
            st.rerun()

def step_1_basic_info():
    """Step 1: Personal Details Form & Health History"""
    st.markdown('<div class="section-header">📋 Personal Information & Health History</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box" style="background-color: #F3E5F5; border-left: 5px solid #9C27B0;">
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
        gender = st.selectbox("Gender", ["Male", "Female"],
                             help="Biological sex assigned at birth")

    with col2:
        appearance_in_kinship = st.selectbox(
            "Family history of Parkinson's disease",
            ["No", "Yes"],
            help="Do any blood relatives have Parkinson's disease?"
        )

    # Health History Section
    st.markdown("---")
    st.markdown("### 🏥 Motor-Related Health History")

    has_conditions = st.radio(
        "Have you been diagnosed with any motor-related conditions?",
        ["No", "Yes"],
        key="has_conditions"
    )

    age_at_diagnosis = 0
    conditions_list = []

    if has_conditions == "Yes":
        age_at_diagnosis = st.number_input(
            "Age at diagnosis",
            min_value=1, max_value=120, value=50,
            help="Age when you were first diagnosed with a motor-related condition"
        )

        st.markdown("**Please select all conditions that apply:**")
        cols = st.columns(2)
        for i, condition in enumerate(MOTION_CONDITIONS):
            with cols[i % 2]:
                if st.checkbox(condition, key=f"condition_{i}"):
                    conditions_list.append(condition)

    # Consent checkbox
    consent = st.checkbox(
        "I consent to this health screening assessment and understand this is not a medical diagnosis",
        value=st.session_state.consent_given
    )

    st.session_state.user_data = {
        "age": age,
        "age_at_diagnosis": age_at_diagnosis,
        "gender": gender,
        "appearance_in_kinship": appearance_in_kinship,
        "has_conditions": has_conditions == "Yes",
        "conditions": conditions_list
    }
    st.session_state.consent_given = consent

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.button("← Back", key="back_1"):
            st.session_state.current_step = 0
            st.rerun()

    with col3:
        if st.button("Continue to Assessment →", type="primary", key="continue_1", disabled=not consent):
            if consent:
                st.session_state.current_step = 2
                st.rerun()
            else:
                st.error("Please provide consent to continue")
                

def get_user_answers_by_id(id_index):
    """Return user questionnaire answers for a given id_index from CSV."""
    questionnaire_df = pd.read_csv(csv_path)
    questionnaire_df['id_index'] = questionnaire_df['Unnamed: 0']  # <-- Add this line
    row = questionnaire_df[questionnaire_df["id_index"] == id_index]
    if row.empty:
        return None
    # Drop id_index column, keep only question responses
    answers = row.drop(columns=["id_index"]).iloc[0].to_dict()
    return answers

def fetch_user_data(user_id):
    """
    Fetch existing user data based on user ID from the FastAPI backend.

    Args:
        user_id (str): The ID of the user to fetch.

    Returns:
        dict or None: The user's data if found, otherwise None.
    """
    try:
        response = requests.get(f"{FASTAPI_URL_USER}/{user_id}")
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            return None
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error: {str(e)}")
        return None

def step_2_assessment():
    """Step 2: Motor Data Collection & Health Questionnaire"""
    st.markdown('<div class="section-header">📊 Health Assessment & Questionnaire</div>', unsafe_allow_html=True)
    

    # User ID Section (Optional)
    st.markdown("""
    <div class="info-box" style="background-color: #E8F5E8; border-left: 5px solid #4CAF50;">
        <h4>👤 Optional: Input User ID</h4>
        <p>If you have an existing user ID, enter it below to pre-populate your previous responses.</p>
    </div>
    """, unsafe_allow_html=True)

    
    col1, col2 = st.columns([2, 1])
    with col1:
        id_index = st.number_input(
            "Enter ID Index (optional)",
            min_value=0,
            value=st.session_state.id_index if st.session_state.id_index else 0,
            help="Load questionnaire data by ID"
        )
        
    with col2:
        if st.button("Load Data", key="load_csv_data"):
            if id_index > 0:
                # Load data from CSV
                answers = get_user_answers_by_id(id_index)
                if answers:
                    st.session_state.id_index = id_index
                    st.session_state.user_id = str(id_index)  # <-- ADD THIS LINE
                    st.session_state.existing_user_data = None  # Clear user data
                    st.success(f"✅ data loaded for ID Index: {id_index}")
                    st.rerun()
                else:
                    st.error(f"❌ No data found for ID Index: {id_index}")
            else:
                st.session_state.id_index = None
                st.session_state.user_id = ""  # <-- ADD THIS LINE (clear user_id)
                st.info("Starting fresh assessment")    
            
    
    


    # Display loaded user info if available
    if st.session_state.existing_user_data:
        st.markdown("""
        <div class="info-box" style="background-color: #E3F2FD; border-left: 5px solid #2196F3;">
            <h4>📋 Loaded User Information</h4>
        </div>
        """, unsafe_allow_html=True)

        existing = st.session_state.existing_user_data
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Age", existing["age"])
        with col2:
            st.metric("Gender", existing["gender"])
        with col3:
            st.metric("Family History", existing["appearance_in_kinship"])
        with col4:
            st.metric("Age at Diagnosis", existing["age_at_diagnosis"] if existing["age_at_diagnosis"] > 0 else "N/A")

    # Show loaded CSV data if available
    if st.session_state.id_index:
        st.markdown(f"""
        <div class="info-box" style="background-color: #FFF3E0; border-left: 5px solid #FF9800;">
            <h4>📄 Loaded Your Apple Watch Data based on your ID: {st.session_state.id_index}</h4>
            <p>Questionnaire responses will be pre-filled.</p>
        </div>
        """, unsafe_allow_html=True)

    # Health Questionnaire Section
    st.markdown("---")
    st.markdown("""
    <div class="questionnaire-section">
        <h3>🔍 Health Symptom Questionnaire</h3>
        <p>Please answer all questions based on your experience over the past few months.</p>
    </div>
    """, unsafe_allow_html=True)

    responses = {}

    # Get existing responses if user data is loaded
    existing_responses = {}
    if st.session_state.existing_user_data:
        # From User ID
        existing_responses = st.session_state.existing_user_data.get("responses", {})
    elif st.session_state.id_index:
        # From CSV ID Index
        csv_answers = get_user_answers_by_id(st.session_state.id_index)
        if csv_answers:
            # Convert CSV answers to boolean format expected by questionnaire
            for key in QUESTIONNAIRE.keys():
                if key in csv_answers:
                    existing_responses[key] = bool(csv_answers[key])
    
    
   

    # Create questionnaire in a more compact format
    question_keys = list(QUESTIONNAIRE.keys())

    # Progress indicator
    progress_placeholder = st.empty()

    # Display questions in a grid format
    for i in range(0, len(question_keys), 2):
        col1, col2 = st.columns(2)

        # First question in the pair
        question_key = question_keys[i]
        question_text = QUESTIONNAIRE[question_key]

        with col1:
            st.markdown(f"**Q{i+1}:** {question_text}")
            default_value = existing_responses.get(question_key, False)
            response = st.radio(
                "Select your answer:",
                ["No", "Yes"],
                index=1 if default_value else 0,
                key=f"q_{question_key}",
                horizontal=True
            )
            responses[question_key] = response == "Yes"

        # Second question in the pair (if exists)
        if i + 1 < len(question_keys):
            question_key = question_keys[i + 1]
            question_text = QUESTIONNAIRE[question_key]

            with col2:
                st.markdown(f"**Q{i+2}:** {question_text}")
                default_value = existing_responses.get(question_key, False)
                response = st.radio(
                    "Select your answer:",
                    ["No", "Yes"],
                    index=1 if default_value else 0,
                    key=f"q_{question_key}",
                    horizontal=True
                )
                responses[question_key] = response == "Yes"

        st.markdown("---")

    # Update progress
    completed_questions = len(responses)
    total_questions = len(QUESTIONNAIRE)
    progress_placeholder.progress(completed_questions / total_questions)

    st.session_state.questionnaire_responses = responses

    # Navigation buttons
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.button("← Back", key="back_2"):
            st.session_state.current_step = 1
            st.rerun()

    with col3:
        if st.button("Get My Results →", type="primary", key="continue_2"):
            if len(responses) == len(QUESTIONNAIRE):
                st.session_state.current_step = 3
                st.rerun()
            else:
                st.error(f"Please answer all {total_questions} questions to continue. ({completed_questions}/{total_questions} completed)")

def make_prediction(user_data, questionnaire_responses):
    required_questions = ["02", "03", "09", "13", "17", "20"]

    payload = {
        "age": int(user_data["age"]),
        "age_at_diagnosis": int(user_data["age_at_diagnosis"]),
        "appearance_in_kinship": user_data["appearance_in_kinship"],  # "Yes"/"No"
        "gender": user_data["gender"],  # "Male"/"Female"
    }

    for q in required_questions:
        payload[q] = 1 if questionnaire_responses.get(q, False) else 0

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
    
def predict_by_user_id(user_id):
    """Call FastAPI endpoint to predict using user_id - uses FASTAPI_URL_USER."""
    try:
        payload = {"user_id": int(user_id)}
        response = requests.post(FASTAPI_URL_USER, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error: {str(e)}")
        return None
    

def display_classification_result(prediction, confidence=None, probabilities=None):
    """Display the main classification result with styling"""
    # Use a single column for the result banner
    col = st.container()
    with col:
        if prediction == 'Healthy':
            st.markdown(f"""
            <div class="metric-card healthy-card">
                <div class="result-title">🟢 Classification Result: {prediction}</div>
                <p>Our result shows you are healthy! Please keep it up!</p>
            </div>
            """, unsafe_allow_html=True)
        elif prediction == "Parkinson's Disease":
            st.markdown(f"""
            <div class="metric-card parkinson-card">
                <div class="result-title">🔴 Classification Result: {prediction}</div>
                <p>Our result shows you have a higher risk for Parkinson’s disease symptoms. Please seek professional assistance!</p>
            </div>
            """, unsafe_allow_html=True)
        else:  # Other Motor Disease
            st.markdown(f"""
            <div class="metric-card motor-card">
                <div class="result-title">🟡 Classification Result: {prediction}</div>
                <p>Our result shows you have a lower risk for Parkinsons's disease but higher risk for other motor disease symptoms. Please seek professional assistance!</p>
            </div>
            """, unsafe_allow_html=True)
   

    

def step_3_results():
     # Insert an anchor at the top
    st.markdown('<a id="top"></a>', unsafe_allow_html=True)
    
    #"""Step 3: Results & Insights - FIXED: Updated prediction logic"""
    st.markdown('<div class="section-header">📊 Your Motor Health Results</div>', unsafe_allow_html=True)

    # Show user identifier
    if st.session_state.user_id:
        st.markdown(f"""
        <div class="info-box" style="background-color: #E3F2FD; border-left: 5px solid #2196F3;">
            <h4>👤 Assessment for User ID: {st.session_state.user_id}</h4>
        </div>
        """, unsafe_allow_html=True)
    elif st.session_state.id_index:
        st.markdown(f"""
        <div class="info-box" style="background-color: #FFF3E0; border-left: 5px solid #FF9800;">
            <h4>📄 Assessment for ID Index: {st.session_state.id_index}</h4>
        </div>
        """, unsafe_allow_html=True)

    result_container = st.container()

    with st.spinner("Analyzing your health data..."):
        prediction_result = None

        if st.session_state.user_id:
            prediction_result = predict_by_user_id(st.session_state.user_id)
        else:
            prediction_result = make_prediction(
                st.session_state.user_data,
                st.session_state.questionnaire_responses
            )

        with result_container:
            if prediction_result is None:
                st.error("Unable to generate results. Please try again.")
                return

            prediction_class = prediction_result.get("prediction", None)
            confidence = prediction_result.get("confidence", None)
            probabilities = prediction_result.get("probabilities", None)

            if prediction_class is None:
                st.error("Invalid response from prediction service.")
                return

            # Map prediction_class to label
            if prediction_class == 0:
                label = "Healthy"
            elif prediction_class == 1:
                label = "Parkinson's Disease"
            else:
                label = "Other Motor Disease"

            display_classification_result(label, confidence, probabilities)
            
        

        # Rest of the results display remains the same...
        st.markdown("### 📚 Helpful Resources")
        # ... (resources section unchanged)

        # Action buttons
        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔄 New Assessment", key="new_assessment"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

        with col2:
            if st.button("← Back to Edit", key="back_to_edit"):
                st.session_state.current_step = 2
                st.rerun()

        # Disclaimer
        # Footer
        st.markdown("""
        <div class="disclaimer-box">
            <strong>⚠️ Medical Disclaimer:</strong>
            This screening tool is for informational purposes only and is not a substitute for
            professional medical diagnosis. Results should not be used as the sole basis for medical decisions.
            Please consult with a qualified healthcare provider for proper medical evaluation and diagnosis.
        </div>
        """, unsafe_allow_html=True)


def main():
    """Main application function with step-based navigation."""
    initialize_session_state()

    # Show progress indicator
    show_step_indicator(st.session_state.current_step, 4)

    # Route to appropriate step
    if st.session_state.current_step == 0:
        step_0_welcome()
    elif st.session_state.current_step == 1:
        step_1_basic_info()
    elif st.session_state.current_step == 2:
        step_2_assessment()
    elif st.session_state.current_step == 3:
        step_3_results()
        
    

if __name__ == "__main__":
    main()
