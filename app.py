import os
import re
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

.stMultiSelect div {
    background: #020617 !important;
    border: 1.5px solid #1e293b !important;
    border-radius: 16px !important;
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

.footer {
    text-align: center;
    color: #64748b;
    font-size: .80rem;
    margin-top: 30px;
}

.small-note {
    color: #94a3b8;
    font-size: 0.95rem;
}
</style>
""", unsafe_allow_html=True)

BASE = os.path.dirname(__file__)

# ================== REQUIRED FILES ==================
required_files = [
    "rf_model.pkl",
    "label_encoder.pkl",
    "feature_columns.pkl"
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

def extract_clean_symptom(feature_name):
    feature_name = str(feature_name).strip()
    parts = feature_name.split("_")

    if len(parts) >= 3 and parts[0].lower() == "symptom" and parts[1].isdigit():
        return " ".join(parts[2:]).strip().lower()

    return feature_name.replace("_", " ").strip().lower()

def confidence_message(top_conf, second_conf):
    gap = top_conf - second_conf

    if top_conf >= 0.60:
        return "good", "Strong confidence prediction."
    if top_conf >= 0.35 and gap >= 0.10:
        return "good", "Reasonable confidence prediction."
    if top_conf >= 0.20:
        return "medium", "Moderate confidence. Adding more symptoms may improve the result."
    return "low", "Low confidence. Add more symptoms for a better prediction."

# ================== OPTIONAL CSV FILES ==================
description_file = find_existing_file([
    "symptom_description.csv",
    "symptom_Description.csv"
])

precaution_file = find_existing_file([
    "disease_precaution.csv",
    "Disease precaution.csv"
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

# ================== SYMPTOM MAP ==================
symptom_to_columns = {}
for col in feature_columns:
    clean_symptom = extract_clean_symptom(col)
    if clean_symptom and clean_symptom != "none":
        symptom_to_columns.setdefault(clean_symptom, []).append(col)

symptoms_list = sorted(symptom_to_columns.keys())

# ================== PREDICTION ==================
def predict_topk_rf(selected_symptoms, k=5):
    input_dict = {col: 0 for col in feature_columns}

    for symptom in selected_symptoms:
        clean_symptom = normalize_symptom_text(symptom)
        matching_columns = symptom_to_columns.get(clean_symptom, [])
        for col in matching_columns:
            input_dict[col] = 1

    input_df = pd.DataFrame([input_dict])

    probabilities = rf.predict_proba(input_df)[0]
    top_indices = np.argsort(probabilities)[::-1][:k]

    results = []
    for idx in top_indices:
        disease_name = le.inverse_transform([idx])[0]
        confidence = float(probabilities[idx])
        results.append((disease_name, confidence))

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
    st.markdown('<div class="input-label">Select Your Symptoms</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="warn-box">
        ⚠️ Choose symptoms from the list below.<br>
        Do not type random text. Select symptoms from the suggestions only.
    </div>
    """, unsafe_allow_html=True)

    selected = st.multiselect(
        "Symptoms",
        symptoms_list,
        placeholder="Click and choose symptoms...",
        help="Start typing to search, then select from the list",
        max_selections=10,
        label_visibility="collapsed"
    )

    diagnose_clicked = st.button("Diagnose")

    if diagnose_clicked:
        if not selected:
            st.warning("Please select at least one symptom.")
        else:
            with st.spinner("Analyzing symptoms..."):
                results = predict_topk_rf(selected, k=5)

            st.session_state["results"] = results

            top_disease, top_conf = results[0]
            second_conf = results[1][1] if len(results) > 1 else 0.0
            level, msg = confidence_message(top_conf, second_conf)

            st.markdown(f"""
            <div class="result-card top">
                <div class="disease-name">{top_disease}</div>
                <div class="bar-bg">
                    <div class="bar" style="width:{top_conf * 100:.1f}%"></div>
                </div>
                <p>{top_conf * 100:.1f}% confidence</p>
            </div>
            """, unsafe_allow_html=True)

            if level == "good":
                st.markdown(f'<div class="good-conf">{msg}</div>', unsafe_allow_html=True)
            elif level == "medium":
                st.markdown(f'<div class="med-conf">{msg}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="low-conf">{msg}</div>', unsafe_allow_html=True)

            disease_key = normalize_disease_key(top_disease)

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
                    <b>{disease}</b><br>
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