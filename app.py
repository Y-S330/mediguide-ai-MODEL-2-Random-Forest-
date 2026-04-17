import os
import re
from html import escape

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MediGuide AI", layout="wide")

# ================== STYLE ==================
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

BASE = os.path.dirname(__file__)

# ================== REQUIRED FILES ==================
required_files = [
    "rf_model.pkl",
    "label_encoder.pkl",
    "feature_columns.pkl",
]

missing_files = [f for f in required_files if not os.path.exists(os.path.join(BASE, f))]

if missing_files:
    st.error("Missing required files:")
    for f in missing_files:
        st.write(f"- {f}")
    st.stop()

# ================== HELPERS ==================
def find_existing_file(candidates):
    for name in candidates:
        path = os.path.join(BASE, name)
        if os.path.exists(path):
            return path
    return None

def normalize_disease_key(disease_name):
    return str(disease_name).strip().lower().replace(" ", "")

def normalize_symptom_text(text):
    return str(text).strip().lower().replace("_", " ")

def confidence_message(top_conf, second_conf):
    gap = top_conf - second_conf

    if top_conf >= 0.50:
        return "good", "Strong confidence prediction."
    if top_conf >= 0.35 and gap >= 0.10:
        return "good", "Reasonable confidence prediction."
    if top_conf >= 0.20:
        return "medium", "Moderate confidence. Adding more symptoms may improve the result."
    return "low", "Low confidence. Add more relevant symptoms for a better prediction."

def clean_text_for_match(text):
    text = str(text).lower().strip()
    text = text.replace("_", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ================== OPTIONAL CSV FILES ==================
description_file = find_existing_file([
    "symptom_description.csv",
    "symptom_Description.csv",
])

precaution_file = find_existing_file([
    "disease_precaution.csv",
    "Disease precaution.csv",
])

# ================== LOAD MODELS ==================
@st.cache_resource
def load_models():
    rf_model = joblib.load(os.path.join(BASE, "rf_model.pkl"))
    label_encoder = joblib.load(os.path.join(BASE, "label_encoder.pkl"))
    feature_cols = joblib.load(os.path.join(BASE, "feature_columns.pkl"))
    return rf_model, label_encoder, feature_cols

# ================== LOAD MAPS ==================
@st.cache_data
def load_maps():
    desc_map = {}
    prec_map = {}

    if description_file is not None:
        desc_df = pd.read_csv(description_file)

        if {"Disease", "Description"}.issubset(desc_df.columns):
            desc_df["Disease"] = (
                desc_df["Disease"]
                .astype(str)
                .str.strip()
                .str.lower()
                .str.replace(" ", "", regex=False)
            )
            desc_df["Description"] = desc_df["Description"].astype(str).str.strip()
            desc_map = dict(zip(desc_df["Disease"], desc_df["Description"]))

    if precaution_file is not None:
        prec_df = pd.read_csv(precaution_file)

        if "Disease" in prec_df.columns:
            prec_df["Disease"] = (
                prec_df["Disease"]
                .astype(str)
                .str.strip()
                .str.lower()
                .str.replace(" ", "", regex=False)
            )

            for _, row in prec_df.iterrows():
                disease = row["Disease"]
                precautions = row[1:].dropna().astype(str).str.strip().tolist()
                prec_map[disease] = precautions

    return desc_map, prec_map

rf, le, feature_columns = load_models()
desc_map, prec_map = load_maps()

# ================== SYMPTOMS LIST ==================
# In your final pipeline, feature_columns.pkl stores all_symptoms,
# so each item is already a symptom name like "headache", "high_fever", etc.
feature_columns = [str(x).strip() for x in feature_columns if str(x).strip()]
symptoms_list = sorted(
    [normalize_symptom_text(s) for s in feature_columns if normalize_symptom_text(s) != "none"]
)

feature_index = {normalize_symptom_text(sym): i for i, sym in enumerate(feature_columns)}

# ================== AUTO-GENERATED ALIAS MAP ==================
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
    "sweating": "sweating",
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
    "blurred and distorted vision": "blurred and distorted vision",

    "runny nose": "runny nose",
    "blocked nose": "congestion",
    "nasal congestion": "congestion",
    "sore throat": "throat irritation",
    "throat pain": "throat irritation",
    "throat irritation": "throat irritation",

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
    "can't breathe": "breathlessness",

    "sneezing": "continuous sneezing",
    "continuous sneezing": "continuous sneezing",

    "dizziness": "dizziness",
    "dizzy": "dizziness",
    "lightheaded": "dizziness",
    "lightheadedness": "dizziness",
}

@st.cache_data
def build_alias_map(symptoms):
    alias_map = {}

    for symptom in symptoms:
        s = clean_text_for_match(symptom)

        alias_map[s] = symptom
        alias_map[s.replace(" ", "_")] = symptom

        if s.endswith("s"):
            alias_map[s[:-1]] = symptom
        else:
            alias_map[s + "s"] = symptom

        alias_map[s.replace(" and ", " ")] = symptom
        alias_map[s.replace(" ", "")] = symptom

    for alias, target in MANUAL_ALIASES.items():
        alias_clean = clean_text_for_match(alias)
        target_clean = clean_text_for_match(target)
        if target_clean in symptoms:
            alias_map[alias_clean] = target_clean

    return alias_map

alias_map = build_alias_map(symptoms_list)
sorted_aliases = sorted(alias_map.keys(), key=len, reverse=True)

def extract_symptoms_from_text(text):
    cleaned = clean_text_for_match(text)

    if not cleaned:
        return [], ""

    detected = []
    remaining = cleaned

    for symptom in feature_columns:

        # normalize both sides
        normalized = clean_text_for_match(symptom)

        # 🔥 flexible matching (handles small variations)
        words = normalized.split()
        pattern = r"\b" + r"\s+".join(words) + r"\b"

        if re.search(pattern, remaining):
            if symptom not in detected:
                detected.append(symptom)

            remaining = re.sub(pattern, " ", remaining, count=1)

    remaining = re.sub(r"\s+", " ", remaining).strip()

    return detected, remaining

# ================== PREDICTION ==================
def predict_topk_rf(selected_symptoms, k=5):
    input_vector = np.zeros((1, len(feature_columns)), dtype=np.float32)

    matched_count = 0

    for symptom in selected_symptoms:
        clean_symptom = normalize_symptom_text(symptom)

        if clean_symptom in feature_index:
            idx = feature_index[clean_symptom]
            input_vector[0, idx] = 1.0
            matched_count += 1

    probabilities = rf.predict_proba(input_vector)[0]

    k = min(k, len(probabilities))
    top_indices = np.argsort(probabilities)[::-1][:k]

    results = []
    for idx in top_indices:
        disease_name = le.inverse_transform([idx])[0]
        confidence = float(probabilities[idx])
        results.append((disease_name, confidence))

    top_conf = results[0][1]
    second_conf = results[1][1] if len(results) > 1 else 0.0

    if matched_count <= 1:
        return {
            "warning": "Too few valid symptoms detected. Please provide more specific symptoms.",
            "results": results
        }

    if top_conf < 0.35 or (top_conf - second_conf) < 0.10:
        return {
            "warning": "Low confidence prediction. Try adding clearer symptoms.",
            "results": results
        }

    return results

# ================== UI ==================
st.markdown("""
<div class="hero">
    <h1>🩺 MediGuide AI</h1>
    <p>AI-powered disease prediction using structured symptoms</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="input-label">Select or Describe Your Symptoms</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="warn-box">
        ⚠️ You can use the dropdown, type symptoms naturally, or use both.<br>
        Try to enter symptoms from the same illness for better results.
    </div>
    """, unsafe_allow_html=True)

    selected = st.multiselect(
        "Symptoms",
        symptoms_list,
        placeholder="Choose symptoms from the list...",
        help="Start typing to search and select symptoms from the list",
        max_selections=10,
        label_visibility="collapsed"
    )

    free_text = st.text_area(
        "Or type symptoms naturally",
        placeholder="e.g. high fever, headache, blurred vision, frequent urination",
        height=110
    )

    detected_from_text, leftover_text = extract_symptoms_from_text(free_text)

    if free_text.strip():
        if detected_from_text:
            pills_html = "".join(
                f'<span class="symptom-pill">✓ {escape(symptom.title())}</span>'
                for symptom in detected_from_text
            )
            st.markdown(
                f'<div style="margin-top:.5rem"><div class="small-note">Recognized from text</div>{pills_html}</div>',
                unsafe_allow_html=True
            )
        if leftover_text:
            st.markdown(
                f'<div class="unknown-box">Some text was not recognized: <b>{escape(leftover_text)}</b></div>',
                unsafe_allow_html=True
            )

    diagnose_clicked = st.button("Diagnose")

    if diagnose_clicked:
        combined_symptoms = list(dict.fromkeys(selected + detected_from_text))

        if not combined_symptoms:
            st.warning("Please select symptoms or type symptoms that the system can recognize.")
        else:
            with st.spinner("Analyzing symptoms..."):
                output = predict_topk_rf(combined_symptoms, k=5)

            if isinstance(output, dict) and "warning" in output:
                st.warning(output["warning"])
                results = output["results"]
            else:
                results = output

            st.session_state["results"] = results
            st.session_state["used_symptoms"] = combined_symptoms

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
            st.write(", ".join(combined_symptoms))

            st.subheader("Description")
            st.info(desc_map.get(disease_key, "No description available."))

            st.subheader("Precautions")
            precautions = prec_map.get(disease_key, [])
            if precautions:
                for precaution in precautions:
                    st.success(precaution)
            else:
                st.warning("No precautions available.")

with col2:
    st.subheader("Other Possibilities")

    if "results" in st.session_state:
        results = st.session_state["results"]

        if len(results) > 1:
            for disease, conf in results[1:]:
                st.markdown(f"""
                <div class="result-card">
                    <b>{escape(disease)}</b><br>
                    <span class="small-note">{conf * 100:.1f}% confidence</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No additional predictions available yet.")
    else:
        st.info("Your top alternative predictions will appear here after diagnosis.")

st.markdown(
    '<div class="footer">Educational use only — not medical advice</div>',
    unsafe_allow_html=True
)