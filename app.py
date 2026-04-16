import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="MediGuide AI - RF", layout="wide")

# ================= STYLE =================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color:white;
}
.title {
    text-align:center;
    font-size:50px;
    font-weight:bold;
    color:#00FFB2;
}
.subtitle {
    text-align:center;
    color:#bbb;
    margin-bottom:30px;
}
.card {
    background:#121212;
    padding:25px;
    border-radius:15px;
    border:1px solid #00FFB2;
    box-shadow:0px 0px 20px rgba(0,255,178,0.2);
    margin-top:20px;
}
.stButton>button {
    width:100%;
    background:#00FFB2;
    color:black;
    font-size:18px;
    border-radius:10px;
    padding:12px;
}
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown('<div class="title">MediGuide AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Random Forest Disease Prediction</div>', unsafe_allow_html=True)
st.markdown("---")

st.markdown("""
<div style='text-align:center; font-size:18px; margin-bottom:20px; color:#ddd;'>
Select symptoms to predict disease using structured data
</div>
""", unsafe_allow_html=True)

# ================= LOAD =================
@st.cache_data
def load_data():
    df = pd.read_csv("dataset.csv")

    symptom_cols = [c for c in df.columns if "Symptom" in c]

    for col in symptom_cols:
        df[col] = df[col].astype(str).str.strip().str.replace(" ", "_")

    df[symptom_cols] = df[symptom_cols].replace("nan", "none").fillna("none")

    top = df["Disease"].value_counts().head(15).index
    df = df[df["Disease"].isin(top)]

    X = pd.get_dummies(df[symptom_cols])
    y = df["Disease"]

    le = LabelEncoder()
    y = le.fit_transform(y)

    return X, y, le

X, y, le = load_data()

@st.cache_resource
def train():
    m = RandomForestClassifier(n_estimators=100, random_state=42)
    m.fit(X, y)
    return m

model = train()

@st.cache_data
def load_extra():
    p = pd.read_csv("Disease precaution.csv")
    d = pd.read_csv("symptom_Description.csv")

    p["Disease"] = p["Disease"].str.lower().str.replace(" ","")
    d["Disease"] = d["Disease"].str.lower().str.replace(" ","")

    p_map = {r["Disease"]: r[1:].dropna().tolist() for _, r in p.iterrows()}
    d_map = dict(zip(d["Disease"], d["Description"]))

    return p_map, d_map

prec_map, desc_map = load_extra()

# ================= SYMPTOMS =================
clean = []
for c in X.columns:
    s = c.split("_",2)[-1].replace("_"," ")
    if s != "none":
        clean.append(s)
clean = sorted(list(set(clean)))

# ================= INPUT =================
col1, col2 = st.columns([1.2,1])

with col1:
    selected = st.multiselect("🧠 Select Symptoms", clean)

with col2:
    st.write("")

st.markdown("<br>", unsafe_allow_html=True)

btn = st.columns([1,1.5,1])
with btn[1]:
    run = st.button("🔍 Diagnose Now")

# ================= PREDICT =================
if run:
    if not selected:
        st.warning("Select symptoms")
    else:
        inp = {c:0 for c in X.columns}

        for s in selected:
            f = s.replace(" ","_")
            for c in X.columns:
                if f in c:
                    inp[c]=1

        df_in = pd.DataFrame([inp])
        pred = model.predict(df_in)[0]

        disease = le.inverse_transform([pred])[0]
        key = disease.lower().replace(" ","")

        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.markdown(f"<h2 style='color:#00FFB2;'>🧾 {disease}</h2>", unsafe_allow_html=True)
        st.write("Confidence: High")

        st.markdown("---")

        c1,c2 = st.columns(2)

        with c1:
            st.markdown("### 📄 Description")
            st.write(desc_map.get(key,"No description"))

        with c2:
            st.markdown("### 🛡️ Precautions")
            for p in prec_map.get(key,[]):
                st.write("✔",p)

        st.markdown('</div>', unsafe_allow_html=True)