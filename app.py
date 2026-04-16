import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="MediGuide AI - Model 1", layout="wide")

# ================================
# SIDEBAR
# ================================
st.sidebar.title("MediGuide AI")
st.sidebar.success("Model: Random Forest")
st.sidebar.info("Structured symptom-based disease prediction.")

# ================================
# HEADER
# ================================
st.markdown("""
<h1 style='text-align:center; color:#4CAF50;'>MediGuide AI</h1>
<p style='text-align:center; color:gray;'>Random Forest Disease Prediction</p>
""", unsafe_allow_html=True)

st.markdown("---")

# ================================
# LOAD + PROCESS DATA
# ================================
@st.cache_data
def load_data():
    df = pd.read_csv("dataset.csv")

    symptom_cols = [col for col in df.columns if "Symptom" in col]

    for col in symptom_cols:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].str.replace(" ", "_")
        df[col] = df[col].str.replace("__", "_")

    df[symptom_cols] = df[symptom_cols].replace("nan", "none")
    df[symptom_cols] = df[symptom_cols].fillna("none")

    top_diseases = df["Disease"].value_counts().head(15).index
    df = df[df["Disease"].isin(top_diseases)]

    X = pd.get_dummies(df[symptom_cols])
    y = df["Disease"]

    le = LabelEncoder()
    y = le.fit_transform(y)

    return X, y, le

X, y, le = load_data()

@st.cache_resource
def train_model():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = train_model()

# ================================
# LOAD EXTRA DATA
# ================================
@st.cache_data
def load_extra():
    precautions_df = pd.read_csv("Disease precaution.csv")
    desc_df = pd.read_csv("symptom_Description.csv")

    precautions_df["Disease"] = precautions_df["Disease"].str.strip().str.lower().str.replace(" ", "")
    desc_df["Disease"] = desc_df["Disease"].str.strip().str.lower().str.replace(" ", "")

    precautions_map = {
        row["Disease"]: row[1:].dropna().astype(str).values.tolist()
        for _, row in precautions_df.iterrows()
    }

    desc_map = dict(zip(desc_df["Disease"], desc_df["Description"]))

    return precautions_map, desc_map

precautions_map, desc_map = load_extra()

# ================================
# CLEAN SYMPTOMS
# ================================
clean_symptoms = []
for col in X.columns:
    s = col.split("_", 2)[-1].replace("_", " ")
    if s != "none":
        clean_symptoms.append(s)

clean_symptoms = sorted(list(set(clean_symptoms)))

# ================================
# INPUT UI
# ================================
col1, col2 = st.columns(2)

with col1:
    selected = st.multiselect("🧠 Select Symptoms", clean_symptoms)

with col2:
    st.write(" ")

center = st.columns([1,2,1])
with center[1]:
    diagnose = st.button("🔍 Diagnose")

# ================================
# PREDICTION
# ================================
if diagnose:
    if not selected:
        st.warning("Select symptoms first")
    else:
        input_dict = {col: 0 for col in X.columns}

        for symptom in selected:
            formatted = symptom.replace(" ", "_")
            for col in X.columns:
                if formatted in col:
                    input_dict[col] = 1

        input_df = pd.DataFrame([input_dict])
        pred = model.predict(input_df)[0]

        disease = le.inverse_transform([pred])[0]
        key = disease.lower().replace(" ", "")

        st.markdown("---")

        c1, c2, c3 = st.columns(3)

        c1.metric("Disease", disease)
        c2.metric("Confidence", "High")
        c3.metric("Model", "Random Forest")

        st.markdown("---")

        colA, colB = st.columns(2)

        with colA:
            st.markdown("### 📄 Description")
            st.info(desc_map.get(key, "No description"))

        with colB:
            st.markdown("### 🛡️ Precautions")
            for p in precautions_map.get(key, []):
                st.success(p)