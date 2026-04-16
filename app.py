import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="MediGuide AI", layout="wide")

# ================== STYLE SYSTEM ==================
st.markdown("""
<style>

/* GLOBAL */
.stApp {
    background: radial-gradient(circle at top, #0f172a, #020617);
    color: #e2e8f0;
    font-family: 'Inter', sans-serif;
}

/* SPACING */
[data-testid="block-container"] {
    padding: 2rem 3rem;
}

/* HERO */
.hero {
    text-align:center;
    margin-bottom:1.5rem;
}
.hero h1 {
    font-size:3.2rem;
    font-weight:900;
    background:linear-gradient(135deg,#38bdf8,#6366f1,#34d399);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}
.hero p {
    color:#94a3b8;
}

/* INPUT LABEL */
.input-label {
    font-size:1.2rem;
    font-weight:700;
    color:#38bdf8;
    margin-bottom:0.5rem;
}

/* MULTISELECT */
.stMultiSelect div {
    background:#020617!important;
    border:1.5px solid #1e293b!important;
    border-radius:16px!important;
}

/* BUTTON */
.stButton button {
    background:linear-gradient(135deg,#0ea5e9,#6366f1)!important;
    border-radius:14px!important;
    font-weight:600!important;
}
.stButton button:hover {
    transform:translateY(-2px);
    box-shadow:0 12px 30px rgba(14,165,233,.35);
}

/* CARDS */
.result-card {
    background:#020617;
    border:1px solid #1e293b;
    border-radius:18px;
    padding:1.6rem;
    margin-bottom:1.4rem;
    transition:0.25s ease;
}
.result-card:hover {
    transform:translateY(-5px) scale(1.01);
    box-shadow:0 15px 40px rgba(0,0,0,.6);
}

/* TOP RESULT */
.result-card.top {
    border:1.5px solid #38bdf8;
    background:linear-gradient(135deg, rgba(56,189,248,.20), rgba(99,102,241,.08));
    box-shadow:0 25px 80px rgba(56,189,248,.30);
    transform:scale(1.03);
}

/* TEXT */
.disease-name {
    font-size:1.6rem;
    font-weight:900;
}

/* BAR */
.bar-bg {
    background:#1e293b;
    height:7px;
    border-radius:99px;
}
.bar {
    height:7px;
    border-radius:99px;
    background:linear-gradient(90deg,#38bdf8,#6366f1);
}

/* WARNING BOX */
.warn-box {
    background:rgba(251,191,36,.1);
    border:1px solid rgba(251,191,36,.3);
    padding:12px;
    border-radius:14px;
    margin-bottom:10px;
}

/* LOW CONF */
.low-conf {
    background:rgba(239,68,68,.08);
    border:1px solid rgba(239,68,68,.3);
    padding:12px;
    border-radius:14px;
}

/* FOOTER */
.footer {
    text-align:center;
    color:#334155;
    font-size:.75rem;
    margin-top:30px;
}

</style>
""", unsafe_allow_html=True)

# ================== LOAD ==================
BASE = os.path.dirname(__file__)

@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(BASE, "dataset.csv"))
    sym_cols = [c for c in df.columns if "Symptom" in c]

    df[sym_cols] = df[sym_cols].fillna("none")
    X = pd.get_dummies(df[sym_cols])
    y = df["Disease"]

    le = LabelEncoder()
    y = le.fit_transform(y)

    return X, y, le

@st.cache_resource
def train_model(X, y):
    model = RandomForestClassifier(n_estimators=150)
    model.fit(X, y)
    return model

@st.cache_data
def load_maps():
    desc = pd.read_csv(os.path.join(BASE, "symptom_Description.csv"))
    desc_map = dict(zip(desc["Disease"].str.lower(), desc["Description"]))

    prec = pd.read_csv(os.path.join(BASE, "Disease precaution.csv"))
    prec_map = {
        row["Disease"].lower(): row[1:].dropna().tolist()
        for _, row in prec.iterrows()
    }

    return desc_map, prec_map

X, y, le = load_data()
model = train_model(X, y)
desc_map, prec_map = load_maps()

symptoms_list = sorted(set([c.split("_")[-1] for c in X.columns if "none" not in c]))

# ================== PREDICT ==================
def predict_topk_rf(selected, k=5):
    vec = [0]*len(X.columns)

    for i, col in enumerate(X.columns):
        for s in selected:
            if s.replace(" ","_") in col:
                vec[i] = 1

    proba = model.predict_proba([vec])[0]
    idx = np.argsort(proba)[::-1][:k]

    return [(le.inverse_transform([i])[0], float(proba[i])) for i in idx]

# ================== UI ==================
st.markdown("""
<div class="hero">
<h1>🩺 MediGuide AI</h1>
<p>AI-powered disease prediction using structured symptoms</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1,1])

# ================== INPUT ==================
with col1:

    st.markdown('<div class="input-label">Step 1: Select Your Symptoms</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="warn-box">
    ⚠️ Choose symptoms from the list below.<br>
    Do NOT type random text — only select from suggestions.
    </div>
    """, unsafe_allow_html=True)

    selected = st.multiselect(
        "Symptoms",
        symptoms_list,
        placeholder="Click and choose symptoms...",
        help="Start typing to search, then select from list",
        max_selections=10
    )

    if st.button("Diagnose"):
        if not selected:
            st.warning("Please select at least one symptom")
        else:
            with st.spinner("Analyzing..."):
                results = predict_topk_rf(selected)

            top, conf = results[0]

            st.markdown(f"""
            <div class="result-card top">
                <div class="disease-name">{top}</div>
                <div class="bar-bg">
                    <div class="bar" style="width:{conf*100}%"></div>
                </div>
                <p>{conf*100:.1f}% confidence</p>
            </div>
            """, unsafe_allow_html=True)

            if conf < 0.2:
                st.markdown('<div class="low-conf">Low confidence — add more symptoms</div>', unsafe_allow_html=True)

            st.subheader("Description")
            st.info(desc_map.get(top.lower(),"No description"))

            st.subheader("Precautions")
            for p in prec_map.get(top.lower(),[]):
                st.success(p)

# ================== RESULTS ==================
with col2:
    st.subheader("Other Possibilities")

    if "results" in locals():
        for d,c in results[1:]:
            st.markdown(f"""
            <div class="result-card">
                <b>{d}</b><br>
                {c*100:.1f}%
            </div>
            """, unsafe_allow_html=True)

st.markdown('<div class="footer">Educational use only — not medical advice</div>', unsafe_allow_html=True)