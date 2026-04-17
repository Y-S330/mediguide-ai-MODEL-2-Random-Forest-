import os
import re
from html import escape
from typing import Dict, List, Tuple, Optional, Union

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ==============================
# 1) PAGE SETUP
# ==============================
st.set_page_config(page_title="MediGuide AI", page_icon="🩺", layout="wide")

# ==============================
# 2) STYLE
# ==============================
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top, #0f172a, #020617);
    color: #e2e8f0;
    font-family: 'Inter', sans-serif;
}

[data-testid="block-container"] {
    padding: 2rem 3rem;
}

.hero {
    text-align: center;
    margin-bottom: 1.5rem;
}
.hero h1 {
    font-size: 3.2rem;
    font-weight: 900;
    background: linear-gradient(135deg, #38bdf8, #6366f1, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero p {
    color: #94a3b8;
    font-size: 1.05rem;
}

.input-label {
    font-size: 1.2rem;
    font-weight: 700;
    color: #38bdf8;
    margin-bottom: 0.5rem;
}

.stMultiSelect div,
.stTextArea textarea {
    background: #020617 !important;
    border: 1.5px solid #1e293b !important;
    border-radius: 16px !important;
}

.stTextArea textarea:focus {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 0 3px rgba(56, 189, 248, .15) !important;
}

.stButton button {
    background: linear-gradient(135deg, #0ea5e9, #6366f1) !important;
    border-radius: 14px !important;
    font-weight: 600 !important;
    border: none !important;
    padding: 0.6rem 1.2rem !important;
}
.stButton button:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 30px rgba(14, 165, 233, .35);
}

.result-card {
    background: #020617;
    border: 1px solid #1e293b;
    border-radius: 18px;
    padding: 1.4rem;
    margin-bottom: 1rem;
    transition: 0.25s ease;
}
.result-card:hover {
    transform: translateY(-4px) scale(1.01);
    box-shadow: 0 15px 40px rgba(0, 0, 0, .45);
}

.result-card.top {
    border: 1.5px solid #38bdf8;
    background: linear-gradient(135deg, rgba(56, 189, 248, .20), rgba(99, 102, 241, .08));
    box-shadow: 0 25px 80px rgba(56, 189, 248, .25);
}

.disease-name {
    font-size: 1.7rem;
    font-weight: 900;
    margin-bottom: 0.6rem;
}

.bar-bg {
    background: #1e293b;
    height: 8px;
    border-radius: 99px;
    overflow: hidden;
    margin: 0.7rem 0;
}
.bar {
    height: 8px;
    border-radius: 99px;
    background: linear-gradient(90deg, #38bdf8, #6366f1);
}

.warn-box {
    background: rgba(251, 191, 36, .10);
    border: 1px solid rgba(251, 191, 36, .30);
    padding: 12px;
    border-radius: 14px;
    margin-bottom: 10px;
}

.low-conf {
    background: rgba(239, 68, 68, .08);
    border: 1px solid rgba(239, 68, 68, .30);
    padding: 12px;
    border-radius: 14px;
    margin: 0.8rem 0;
}

.med-conf {
    background: rgba(251, 191, 36, .10);
    border: 1px solid rgba(251, 191, 36, .30);
    padding: 12px;
    border-radius: 14px;
    margin: 0.8rem 0;
}

.good-conf {
    background: rgba(34, 197, 94, .08);
    border: 1px solid rgba(34, 197, 94, .25);
    padding: 12px;
    border-radius: 14px;
    margin: 0.8rem 0;
}

.small-note {
    color: #94a3b8;
    font-size: 0.95rem;
}

.symptom-pill {
    display: inline-block;
    background: rgba(52, 211, 153, .12);
    border: 1px solid rgba(52, 211, 153, .30);
    border-radius: 999px;
    padding: 4px 10px;
    font-size: 0.78rem;
    color: #6ee7b7;
    margin: 3px 4px 0 0;
}

.unknown-box {
    background: rgba(251, 191, 36, .08);
    border: 1px solid rgba(251, 191, 36, .22);
    border-radius: 12px;
    padding: 10px 12px;
    color: #fbbf24;
    margin-top: 10px;
    font-size: 0.85rem;
}

.footer {
    text-align: center;
    color: #64748b;
    font-size: .80rem;
    margin-top: 30px;
}
</style>
""", unsafe_allow_html=True)

BASE = os.path.dirname(os.path.abspath(__file__))

# ==============================
# 3) CORE HELPERS
# ==============================
def find_existing_file(candidates: List[str]) -> Optional[str]:
    for name in candidates:
        path = os.path.join(BASE, name)
        if os.path.exists(path):
            return path
    return None

def clean_text_for_match(text: str) -> str:
    text = str(text).lower().strip()
    text = text.replace("_", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def normalize_disease_key(disease_name: str) -> str:
    return (
        str(disease_name)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
    )

def confidence_message(top_conf: float, second_conf: float) -> Tuple[str, str]:
    gap = top_conf - second_conf

    if top_conf >= 0.50:
        return "good", "Strong confidence prediction."
    if top_conf >= 0.35 and gap >= 0.08:
        return "good", "Reasonable confidence prediction."
    if top_conf >= 0.20:
        return "medium", "Moderate confidence. Adding more symptoms may improve the result."
    return "low", "Low confidence. Add more relevant symptoms for a better prediction."

def render_symptom_pills(symptoms: List[str], prefix_check: bool = False) -> str:
    pills = []
    for sym in symptoms:
        label = escape(str(sym).replace("_", " ").title())
        if prefix_check:
            label = f"✓ {label}"
        pills.append(f'<span class="symptom-pill">{label}</span>')
    return "".join(pills)

# ==============================
# 4) FILE DISCOVERY
# ==============================
model_file = os.path.join(BASE, "rf_model.pkl")
label_encoder_file = os.path.join(BASE, "label_encoder.pkl")
feature_columns_file = os.path.join(BASE, "feature_columns.pkl")
display_features_file = os.path.join(BASE, "display_features.pkl")

required_files = [model_file, label_encoder_file, feature_columns_file]
missing_files = [os.path.basename(f) for f in required_files if not os.path.exists(f)]

if missing_files:
    st.error("Missing required files:")
    for f in missing_files:
        st.write(f"- {f}")
    st.stop()

description_file = find_existing_file([
    "symptom_description.csv",
    "symptom_Description.csv",
])

precaution_file = find_existing_file([
    "disease_precaution.csv",
    "Disease precaution.csv",
])

# ==============================
# 5) LOAD MODELS
# ==============================
@st.cache_resource
def load_models():
    rf_model = joblib.load(model_file)
    label_encoder = joblib.load(label_encoder_file)

    model_feature_cols = joblib.load(feature_columns_file)
    model_feature_cols = [
        str(x).strip()
        for x in model_feature_cols
        if str(x).strip() and str(x).strip().lower() != "none"
    ]

    if os.path.exists(display_features_file):
        raw_display_feature_cols = joblib.load(display_features_file)
        raw_display_feature_cols = [
            str(x).strip()
            for x in raw_display_feature_cols
            if str(x).strip() and clean_text_for_match(x) != "none"
        ]
    else:
        raw_display_feature_cols = [x.replace("_", " ") for x in model_feature_cols]

    if len(model_feature_cols) != len(raw_display_feature_cols):
        raise ValueError("Mismatch between feature_columns.pkl and display_features.pkl lengths.")

    if not hasattr(rf_model, "predict_proba"):
        raise ValueError("Loaded model does not support predict_proba().")

    return rf_model, label_encoder, model_feature_cols, raw_display_feature_cols

# ==============================
# 6) LOAD METADATA MAPS
# ==============================
@st.cache_data
def load_maps():
    desc_map: Dict[str, str] = {}
    prec_map: Dict[str, List[str]] = {}

    if description_file is not None:
        desc_df = pd.read_csv(description_file)
        if {"Disease", "Description"}.issubset(desc_df.columns):
            desc_df["Disease"] = desc_df["Disease"].astype(str).apply(normalize_disease_key)
            desc_df["Description"] = desc_df["Description"].astype(str).str.strip()
            desc_map = dict(zip(desc_df["Disease"], desc_df["Description"]))

    if precaution_file is not None:
        prec_df = pd.read_csv(precaution_file)
        if "Disease" in prec_df.columns:
            prec_df["Disease"] = prec_df["Disease"].astype(str).apply(normalize_disease_key)
            for _, row in prec_df.iterrows():
                disease = row["Disease"]
                precautions = row[1:].dropna().astype(str).str.strip().tolist()
                prec_map[disease] = precautions

    return desc_map, prec_map

try:
    rf, le, model_features, display_features = load_models()
    desc_map, prec_map = load_maps()
except Exception as e:
    st.error(f"Failed to load app resources: {e}")
    st.stop()

# ==============================
# 7) FEATURE MAPPING
# ==============================
display_to_model: Dict[str, str] = {}
for disp, model in zip(display_features, model_features):
    display_to_model[clean_text_for_match(disp)] = model

feature_index = {model: i for i, model in enumerate(model_features)}

# ==============================
# 8) ALIASES / NORMALIZATION RULES
# ==============================
MANUAL_ALIASES = {
    "frequent urination": "polyuria",
    "urinating frequently": "polyuria",
    "pee a lot": "polyuria",
    "excessive urination": "polyuria",

    "burning urination": "burning micturition",
    "pain urinating": "burning micturition",
    "painful urination": "burning micturition",
    "burning while urinating": "burning micturition",

    "fever": "high fever",
    "high fever": "high fever",
    "very high fever": "high fever",
    "mild fever": "mild fever",
    "low fever": "mild fever",
    "slight fever": "mild fever",

    "sweat": "sweating",
    "night sweats": "sweating",
    "chill": "chills",
    "chills": "chills",
    "shivering": "chills",
    "shivers": "chills",

    "cold hands": "cold hands and feets",
    "cold feet": "cold hands and feets",
    "cold hands and feet": "cold hands and feets",

    "light sensitivity": "visual disturbances",
    "sensitivity to light": "visual disturbances",
    "blurred vision": "blurred and distorted vision",
    "blurry vision": "blurred and distorted vision",

    "runny nose": "runny nose",
    "blocked nose": "congestion",
    "nasal congestion": "congestion",
    "sore throat": "throat irritation",
    "throat pain": "throat irritation",

    "joint pain": "joint pain",
    "joint ache": "joint pain",
    "muscle pain": "muscle pain",
    "body ache": "muscle pain",
    "body pain": "muscle pain",
    "chest pain": "chest pain",
    "chest ache": "chest pain",
    "chest discomfort": "chest pain",
    "stomach pain": "stomach pain",
    "stomach ache": "stomach pain",
    "belly pain": "abdominal pain",
    "belly ache": "abdominal pain",
    "abdominal pain": "abdominal pain",
    "back pain": "back pain",
    "lower back pain": "back pain",
    "hip pain": "hip joint pain",

    "headache": "headache",
    "head ache": "headache",
    "head pain": "headache",
    "migraine": "headache",

    "fatigue": "fatigue",
    "tiredness": "fatigue",
    "tired": "fatigue",
    "weakness": "fatigue",
    "lack of energy": "fatigue",

    "diarrhea": "diarrhoea",
    "diarrhoea": "diarrhoea",

    "nausea": "nausea",
    "queasiness": "nausea",
    "vomiting": "vomiting",
    "throwing up": "vomiting",

    "cough": "cough",
    "dry cough": "cough",
    "wet cough": "cough",
    "coughing": "cough",
    "breathlessness": "breathlessness",
    "shortness of breath": "breathlessness",
    "difficulty breathing": "breathlessness",
    "can t breathe": "breathlessness",

    "sneezing": "continuous sneezing",
    "continuous sneezing": "continuous sneezing",

    "dizziness": "dizziness",
    "dizzy": "dizziness",
    "lightheaded": "dizziness",
    "lightheadedness": "dizziness",
}

alias_to_display: Dict[str, str] = {}

for disp in display_features:
    cleaned = clean_text_for_match(disp)
    alias_to_display[cleaned] = cleaned
    alias_to_display[cleaned.replace(" ", "")] = cleaned
    alias_to_display[cleaned.replace(" ", "_")] = cleaned

for alias, target in MANUAL_ALIASES.items():
    alias_clean = clean_text_for_match(alias)
    target_clean = clean_text_for_match(target)

    if target_clean in display_to_model:
        alias_to_display[alias_clean] = target_clean
    elif target_clean.replace(" ", "") in display_to_model:
        alias_to_display[alias_clean] = target_clean.replace(" ", "")

sorted_aliases = sorted(alias_to_display.keys(), key=len, reverse=True)

# ==============================
# 9) TEXT MATCHING (FINAL FIXED)
# ==============================
def extract_symptoms_from_text(text: str) -> Tuple[List[str], str]:
    cleaned = clean_text_for_match(text)
    if not cleaned:
        return [], ""

    detected_model: List[str] = []
    remaining = cleaned

    for alias in sorted_aliases:
        display_key = alias_to_display[alias]

        if display_key not in display_to_model:
            continue

        model_symptom = display_to_model[display_key]
        pattern = r"\b" + re.escape(alias) + r"\b"

        if re.search(pattern, remaining):
            if model_symptom not in detected_model:
                detected_model.append(model_symptom)

            # 🔴 FIX: remove ONLY one match (not all)
            remaining = re.sub(pattern, " ", remaining, count=1)

    remaining = re.sub(r"\s+", " ", remaining).strip()
    return detected_model, remaining

# ==============================
# 10) VECTOR BUILDING
# ==============================
def build_input_vector(selected_model_symptoms: List[str]) -> np.ndarray:
    input_vector = np.zeros((1, len(model_features)), dtype=np.float32)

    for symptom in selected_model_symptoms:
        if symptom in feature_index:
            input_vector[0, feature_index[symptom]] = 1.0

    return input_vector

# ==============================
# 11) MODEL PREDICTION (FINAL FIXED)
# ==============================
def predict_rf_core(selected_model_symptoms_tuple: Tuple[str, ...], k: int = 5) -> Union[Dict[str, str], List[Tuple[str, float]]]:
    selected_model_symptoms = list(selected_model_symptoms_tuple)
    input_vector = build_input_vector(selected_model_symptoms)

    # 🔴 empty vector protection (critical)
    if input_vector.sum() == 0:
        return {"error": "No valid symptoms matched the system vocabulary."}

    try:
        probabilities = rf.predict_proba(input_vector)[0]
    except Exception as e:
        return {"error": f"Prediction failed: {e}"}

    if probabilities is None or len(probabilities) == 0:
        return {"error": "Prediction failed. Empty probability output."}

    k = min(k, len(probabilities))
    top_indices = np.argsort(probabilities)[::-1][:k]

    results: List[Tuple[str, float]] = []
    for idx in top_indices:
        disease_name = le.inverse_transform([idx])[0]
        confidence = float(probabilities[idx])
        results.append((disease_name, confidence))

    if not results:
        return {"error": "Prediction failed. Please try different symptoms."}

    return results

# ==============================
# 12) DECISION LAYER
# ==============================
def evaluate_prediction(results: List[Tuple[str, float]], matched_count: int) -> Dict[str, Union[str, float, List[Tuple[str, float]]]]:
    if not results:
        return {"error": "Prediction failed."}

    top_conf = results[0][1]
    second_conf = results[1][1] if len(results) > 1 else 0.0
    margin = top_conf - second_conf

    if matched_count < 2:
        return {
            "warning": "Too few valid symptoms detected. Please provide more specific symptoms.",
            "results": results,
            "confidence": top_conf,
            "margin": margin
        }

    if top_conf < 0.35:
        return {
            "warning": "Low confidence prediction. Try adding clearer symptoms.",
            "results": results,
            "confidence": top_conf,
            "margin": margin
        }

    if margin < 0.08:
        return {
            "warning": "Symptoms overlap across diseases. Add more details.",
            "results": results,
            "confidence": top_conf,
            "margin": margin
        }

    return {
        "results": results,
        "confidence": top_conf,
        "margin": margin
    }

# ==============================
# 13) SESSION STATE
# ==============================
if "selected_display" not in st.session_state:
    st.session_state["selected_display"] = []
if "free_text" not in st.session_state:
    st.session_state["free_text"] = ""
if "results" not in st.session_state:
    st.session_state["results"] = None
if "used_symptoms" not in st.session_state:
    st.session_state["used_symptoms"] = []
if "decision_warning" not in st.session_state:
    st.session_state["decision_warning"] = None
if "decision_margin" not in st.session_state:
    st.session_state["decision_margin"] = None

# ==============================
# 14) UI HEADER
# ==============================
st.markdown("""
<div class="hero">
    <h1>🩺 MediGuide AI</h1>
    <p>AI-powered disease prediction using structured symptoms</p>
</div>
""", unsafe_allow_html=True)

# ==============================
# 15) MAIN UI (FINAL FIXED)
# ==============================
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="input-label">Select or Describe Your Symptoms</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="warn-box">
        ⚠️ You can use the dropdown, type symptoms naturally, or use both.<br>
        Try to enter symptoms from the same illness for better results.
    </div>
    """, unsafe_allow_html=True)

    selected_display = st.multiselect(
        "Symptoms",
        display_features,
        placeholder="Choose symptoms from the list...",
        help="Start typing to search and select symptoms from the list",
        max_selections=10,
        label_visibility="collapsed",
        key="selected_display"
    )

    free_text = st.text_area(
        "Or type symptoms naturally",
        placeholder="e.g. high fever, headache, blurred vision, frequent urination",
        height=110,
        key="free_text"
    )

    detected_model, leftover_text = extract_symptoms_from_text(free_text)

    if free_text.strip():
        if detected_model:
            st.markdown(
                f'<div style="margin-top:.5rem"><div class="small-note">Recognized from text</div>{render_symptom_pills(detected_model, prefix_check=True)}</div>',
                unsafe_allow_html=True
            )
        if leftover_text:
            st.markdown(
                f'<div class="unknown-box">Some text was not recognized: <b>{escape(leftover_text)}</b></div>',
                unsafe_allow_html=True
            )

    b1, b2 = st.columns([3, 1])

    with b1:
        diagnose_clicked = st.button("Diagnose", use_container_width=True)

    with b2:
        clear_clicked = st.button("Clear", use_container_width=True)

    # 🔴 FIXED CLEAR BUTTON (SAFE VERSION)
    if clear_clicked:
        for key in [
            "selected_display",
            "free_text",
            "results",
            "used_symptoms",
            "decision_warning",
            "decision_margin"
        ]:
            if key in st.session_state:
                del st.session_state[key]

        st.rerun()

    if diagnose_clicked:
        selected_model = []
        for s in selected_display:
            display_key = clean_text_for_match(s)
            if display_key in display_to_model:
                selected_model.append(display_to_model[display_key])

        combined_symptoms = list(dict.fromkeys(selected_model)) + [
            s for s in detected_model if s not in selected_model
        ]

        if not combined_symptoms:
            st.warning("Please select symptoms or type symptoms that the system can recognize.")
        else:
            with st.spinner("Analyzing symptoms..."):
                prediction_output = predict_rf_core(tuple(combined_symptoms), k=5)

            if isinstance(prediction_output, dict) and "error" in prediction_output:
                st.error(prediction_output["error"])
                st.session_state["results"] = None
                st.session_state["used_symptoms"] = []
                st.session_state["decision_warning"] = None
                st.session_state["decision_margin"] = None
            else:
                decision = evaluate_prediction(prediction_output, len(combined_symptoms))

                if "error" in decision:
                    st.error(decision["error"])
                    st.session_state["results"] = None
                    st.session_state["used_symptoms"] = []
                    st.session_state["decision_warning"] = None
                    st.session_state["decision_margin"] = None
                else:
                    st.session_state["results"] = decision["results"]
                    st.session_state["used_symptoms"] = combined_symptoms
                    st.session_state["decision_warning"] = decision.get("warning")
                    st.session_state["decision_margin"] = decision.get("margin")

    if st.session_state.get("results"):
        results = st.session_state["results"]
        combined_symptoms = st.session_state["used_symptoms"]
        decision_warning = st.session_state.get("decision_warning")

        top_disease, top_conf = results[0]
        second_conf = results[1][1] if len(results) > 1 else 0.0
        level, msg = confidence_message(top_conf, second_conf)

        st.markdown(f"""
        <div class="result-card top">
            <div class="disease-name">{escape(top_disease)}</div>
            <div class="bar-bg">
                <div class="bar" style="width:{top_conf * 100:.1f}%"></div>
            </div>
            <p>{top_conf * 100:.1f}% confidence</p>
        </div>
        """, unsafe_allow_html=True)

        if decision_warning:
            st.warning(decision_warning)

        if level == "good":
            st.markdown(f'<div class="good-conf">{escape(msg)}</div>', unsafe_allow_html=True)
        elif level == "medium":
            st.markdown(f'<div class="med-conf">{escape(msg)}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="low-conf">{escape(msg)}</div>', unsafe_allow_html=True)

        if len(combined_symptoms) < 2:
            st.warning("Only one symptom was recognized. Results may be weak.")
        elif len(combined_symptoms) > 8:
            st.warning("Many symptoms were entered. If they belong to multiple illnesses, accuracy may decrease.")

        disease_key = normalize_disease_key(top_disease)

        st.subheader("Recognized Symptoms Used")
        st.markdown(render_symptom_pills(combined_symptoms), unsafe_allow_html=True)

        st.subheader("Description")
        desc = desc_map.get(disease_key, "No description available.")
        st.markdown(f"**{escape(desc)}**")

        st.subheader("Precautions")
        precautions = prec_map.get(disease_key, [])
        if precautions:
            for precaution in precautions:
                st.success(precaution)
        else:
            st.warning("No precautions available.")

# ==============================
# 16) FOOTER
# ==============================
st.markdown(
    '<div class="footer">Educational use only — not medical advice</div>',
    unsafe_allow_html=True
)