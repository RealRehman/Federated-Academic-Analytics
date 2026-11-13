import os

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Student Academic Outcome Predictor",
    page_icon="🎓",
    layout="wide",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "random_forest_model.pkl")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "..", "models", "preprocessor.pkl")
CSS_PATH = os.path.join(BASE_DIR, "styles.css")

APP_NAME = "ACADEMIC ANALYTICS"


@st.cache_resource
def load_artifacts():
    return joblib.load(MODEL_PATH), joblib.load(PREPROCESSOR_PATH)


model, preprocessor = load_artifacts()

# ---------------------------------------------------------------- styling
DARK_VARS = """
:root {
  --ink: #ecf4e9; --muted: #a2b2a9; --cream: #101714; --paper: #17221e;
  --line: rgba(233, 245, 233, .13); --glass: rgba(35, 51, 44, .52);
  --glass-border: rgba(255, 255, 255, .08); --shadow: 0 24px 80px rgba(0, 0, 0, .25);
}
.liquid-orb { opacity: .28; }
"""

dark = st.session_state.get("dark", False)
with open(CSS_PATH, encoding="utf-8") as f:
    css = f.read()
st.markdown(f"<style>{css}{DARK_VARS if dark else ''}</style>", unsafe_allow_html=True)

st.markdown(
    '<div class="liquid-orb orb-one"></div>'
    '<div class="liquid-orb orb-two"></div>'
    '<div class="liquid-orb orb-three"></div>'
    '<div class="page-noise"></div>',
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------- header
head_left, head_right = st.columns([6, 1.3])
with head_left:
    st.markdown(
        f'<div class="brand"><span class="brand-mark">A</span>'
        f'<span class="brand-name">{APP_NAME}</span></div>',
        unsafe_allow_html=True,
    )
with head_right:
    st.toggle("Dark mode", key="dark")

# ------------------------------------------------------------------- hero
hero_left, hero_right = st.columns([1.15, 1], gap="large")

with hero_left:
    st.markdown(
        '<div style="height:30px"></div>'
        '<div class="eyebrow"><span class="pulse-dot"></span>'
        "Student success intelligence, without the guesswork</div>"
        '<h1 class="hero-title">Student outcomes.<br><em>In focus.</em></h1>'
        '<p class="hero-lede">Enter a few simple student details and the model predicts one of '
        "three outcomes: Dropout, Enrolled, or Graduate.</p>",
        unsafe_allow_html=True,
    )

with hero_right:
    st.markdown(
        '<div class="glass-panel">'
        '<div class="window-bar"><span class="mono">PREDICT / EXAMPLE</span>'
        '<span class="mono window-status"><b></b>RANDOM FOREST</span></div>'
        '<div class="chat-content">'
        '<div class="chat-question">Will this student graduate?</div>'
        '<div class="chat-answer">'
        '<div class="answer-top"><span class="answer-avatar">AI</span><span>PREDICTION</span>'
        '<span class="answer-line"></span><span class="mono">SAMPLE</span></div>'
        "<p>Results appear here with the predicted outcome and the model's "
        "probability for each class.</p>"
        '<div class="citation"><span>Trained on the UCI student dataset</span></div>'
        "</div></div></div>",
        unsafe_allow_html=True,
    )

st.markdown(
    '<div class="signal-strip">'
    '<div><span class="mono">01</span><strong>Simple inputs</strong>'
    "<span>Just the details that matter most.</span></div>"
    '<div><span class="mono">02</span><strong>Instant prediction</strong>'
    "<span>A pre-trained Random Forest does the work.</span></div>"
    '<div><span class="mono">03</span><strong>Clear confidence</strong>'
    "<span>See the probability behind every outcome.</span></div>"
    "</div>",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------- form
st.markdown(
    '<div class="section-kicker">STUDENT INFORMATION</div>'
    '<h2 class="section-title">Tell us about the student.<br>'
    "<span>We'll handle the rest.</span></h2>",
    unsafe_allow_html=True,
)

with st.container(border=True, key="form_card"):
    col1, col2 = st.columns(2, gap="large")

    with col1:
        age = st.number_input("Age at Enrollment", min_value=15, max_value=80, value=20)

        gender_label = st.selectbox("Gender", ["Female", "Male"])
        gender = 0 if gender_label == "Female" else 1

        attendance_label = st.selectbox("Attendance", ["Daytime", "Evening"])
        attendance = 1 if attendance_label == "Daytime" else 0

        admission_grade = st.number_input(
            "Admission Grade", min_value=0.0, max_value=200.0, value=120.0
        )

        international_label = st.selectbox("International Student", ["No", "Yes"])
        international = 1 if international_label == "Yes" else 0

    with col2:
        first_sem_grade = st.number_input(
            "1st Semester Average Grade", min_value=0.0, max_value=20.0, value=12.0
        )
        second_sem_grade = st.number_input(
            "2nd Semester Average Grade", min_value=0.0, max_value=20.0, value=12.0
        )

        tuition_label = st.selectbox("Tuition Fees Paid", ["Yes", "No"])
        tuition_fees = 1 if tuition_label == "Yes" else 0

        scholarship_label = st.selectbox("Scholarship Holder", ["No", "Yes"])
        scholarship = 1 if scholarship_label == "Yes" else 0

        debtor_label = st.selectbox("Has Outstanding Debt", ["No", "Yes"])
        debtor = 1 if debtor_label == "Yes" else 0

    st.caption("Technical dataset features are handled automatically by the application.")
    predict_clicked = st.button("Predict Student Outcome  ↗", use_container_width=True)

# ----------------------------------------------------------------- result
OUTCOME_TEXT = {
    "Dropout": ("dropout", "Elevated dropout risk. This profile resembles students who left before completing."),
    "Enrolled": ("enrolled", "Likely to remain enrolled. This profile resembles students still studying."),
    "Graduate": ("graduate", "Likely to graduate. This profile resembles students who completed their course."),
}

if predict_clicked:
    # The trained model expects all original UCI features.
    # Advanced features are kept internally with standard default values.
    input_data = pd.DataFrame([{
        "Marital status": 1,
        "Application mode": 1,
        "Application order": 1,
        "Course": 171,
        "Daytime/evening attendance\t": attendance,
        "Previous qualification": 1,
        "Previous qualification (grade)": 120.0,
        "Nacionality": 1,
        "Mother's qualification": 1,
        "Father's qualification": 1,
        "Mother's occupation": 0,
        "Father's occupation": 0,
        "Admission grade": admission_grade,
        "Displaced": 0,
        "Educational special needs": 0,
        "Debtor": debtor,
        "Tuition fees up to date": tuition_fees,
        "Gender": gender,
        "Scholarship holder": scholarship,
        "Age at enrollment": age,
        "International": international,
        "Curricular units 1st sem (credited)": 0,
        "Curricular units 1st sem (enrolled)": 6,
        "Curricular units 1st sem (evaluations)": 6,
        "Curricular units 1st sem (approved)": 5,
        "Curricular units 1st sem (grade)": first_sem_grade,
        "Curricular units 1st sem (without evaluations)": 0,
        "Curricular units 2nd sem (credited)": 0,
        "Curricular units 2nd sem (enrolled)": 6,
        "Curricular units 2nd sem (evaluations)": 6,
        "Curricular units 2nd sem (approved)": 5,
        "Curricular units 2nd sem (grade)": second_sem_grade,
        "Curricular units 2nd sem (without evaluations)": 0,
        "Unemployment rate": 10.0,
        "Inflation rate": 1.0,
        "GDP": 1.5,
    }])

    processed_input = preprocessor.transform(input_data)
    prediction = model.predict(processed_input)[0]
    css_class, description = OUTCOME_TEXT.get(prediction, ("graduate", ""))

    bars = ""
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(processed_input)[0]
        for label, p in sorted(zip(model.classes_, probs), key=lambda x: -x[1]):
            cls = OUTCOME_TEXT.get(label, ("graduate", ""))[0]
            bars += (
                f'<div class="prob-row"><span class="mono">{str(label).upper()}</span>'
                f'<div class="prob-track"><div class="prob-fill {cls}" style="width:{p * 100:.1f}%"></div></div>'
                f'<span class="mono">{p * 100:.1f}%</span></div>'
            )

    st.markdown(
        f'<div class="glass-panel result-card {css_class}">'
        '<div class="section-kicker">PREDICTION RESULT</div>'
        f'<div class="result-title">{prediction}</div>'
        f'<p class="result-desc">{description}</p>'
        f"{bars}</div>",
        unsafe_allow_html=True,
    )

# ----------------------------------------------------------------- footer
st.markdown(
    '<div class="site-footer"><span class="brand-name">ACADEMIC ANALYTICS</span>'
    "<span>Student outcomes, clarified.</span>"
    '<span class="mono">RANDOM FOREST · UCI DATASET</span></div>',
    unsafe_allow_html=True,
)