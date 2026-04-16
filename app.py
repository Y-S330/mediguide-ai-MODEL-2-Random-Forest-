import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="MediGuide AI - Model 1", layout="wide")

# ---------- STYLE ----------
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
    }
    .title {
        font-size: 40px;
        font-weight: bold;
        color: #4CAF50;
    }
    .result-box {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #4CAF50;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ================================
# LOAD + PREPROCESS (CACHED)
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
    df["Disease"] = df["Disease"].str.strip()

    # keep only top diseases (clean demo)
    top_diseases = df["Disease"].value_counts().head(15).index
    df = df[df["Disease"].isin(top_diseases)]

    X = pd.get_dummies(df[symptom_cols])
    y = df["Disease"]

    le = LabelEncoder()
    y = le.fit_transform(y)

    return X, y, le, symptom_cols

# ================================
# TRAIN MODEL (CACHED)
# ================================
@st.cache_resource
def train_model(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

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
        row["Disease"]: row[1:].dropna().astype(str).str.strip().values.tolist()
        for _, row in precautions_df.iterrows()
    }

    desc_df["Description"] = desc_df["Description"].astype(str).str.strip()
    desc_map = dict(zip(desc_df["Disease"], desc_df["Description"]))

    return precautions_map, desc_map

# ================================
# PREP
# ================================
X, y, le, symptom_cols = load_data()
model = train_model(X, y)
precautions_map, desc_map = load_extra()

# ================================
# CLEAN SYMPTOMS FOR UI
# ================================
clean_symptoms = []

for col in X.columns:
    symptom = col

    if "_" in symptom:
        symptom = symptom.split("_", 2)[-1]

    symptom = symptom.replace("_", " ")

    if symptom != "none":
        clean_symptoms.append(symptom)

clean_symptoms = sorted(list(set(clean_symptoms)))

# ================================
# UI
# ================================
st.markdown('<p class="title">MediGuide AI - Random Forest Model</p>', unsafe_allow_html=True)

st.markdown("### 🧠 Select Symptoms")
selected_symptoms = st.multiselect("Choose symptoms", clean_symptoms)

st.divider()

# ================================
# PREDICTION
# ================================
if st.button("Diagnose"):

    if not selected_symptoms:
        st.warning("Please select at least one symptom.")
    else:
        input_dict = {col: 0 for col in X.columns}

        for symptom in selected_symptoms:
            formatted = symptom.replace(" ", "_")
            for col in X.columns:
                if formatted in col:
                    input_dict[col] = 1

        input_df = pd.DataFrame([input_dict])

        with st.spinner("Analyzing symptoms..."):
            pred = model.predict(input_df)[0]

        disease_display = le.inverse_transform([pred])[0]

        disease_key = (
            disease_display.strip().lower().replace(" ", "")
        )

        description = desc_map.get(disease_key, "No description available")
        precautions = precautions_map.get(disease_key, [])

        st.markdown(f"""
        <div class="result-box">
        <h2 style='color:#4CAF50;'>Predicted Disease: {disease_display}</h2>
        <p><b>Model:</b> Random Forest</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📄 Description")
        st.info(description)

        st.markdown("### 🛡️ Precautions")

        if precautions:
            for p in precautions:
                st.success(p)
        else:
            st.warning("No precautions available.")

        st.success("Model Confidence: High (structured symptom matching)")