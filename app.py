import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq

st.set_page_config(page_title="Fraud Detection & Investigation Assistant", layout="wide")

@st.cache_resource
def load_resources():
    models = joblib.load('models/trained_models.pkl')
    encoders = joblib.load('models/label_encoders.pkl')
    best_model = models['random_forest']
    explainer = shap.TreeExplainer(best_model)
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Rebuild the vector store fresh from the text knowledge base file
    # (more reliable than persisting/uploading Chroma's binary files across environments)
    with open('data/fraud_indicators_knowledge_base.txt') as f:
        raw_text = f.read()
    documents = [doc.strip() for doc in raw_text.split("DOCUMENT:") if doc.strip()]
    doc_texts = ["DOCUMENT:" + doc for doc in documents]

    embeddings = embed_model.encode(doc_texts).tolist()

    chroma_client = chromadb.Client()  # in-memory, rebuilt fresh each app start
    collection = chroma_client.create_collection(name="fraud_indicators")
    collection.add(
        documents=doc_texts,
        embeddings=embeddings,
        ids=[f"doc_{i}" for i in range(len(doc_texts))]
    )

    feature_columns = list(best_model.feature_names_in_)
    return models, encoders, best_model, explainer, embed_model, collection, feature_columns

models, encoders, best_model, explainer, embed_model, collection, FEATURE_COLUMNS = load_resources()

groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])

st.title("🔍 AI-Powered Insurance Claims Fraud Detection & Investigation Assistant")
st.caption("Predictive fraud scoring + SHAP explainability + RAG-grounded investigation notes")

# --- Sidebar: claim input form ---
st.sidebar.header("Claim Details")

incident_type = st.sidebar.selectbox("Incident Type", encoders['incident_type'].classes_)
incident_severity = st.sidebar.selectbox("Incident Severity", encoders['incident_severity'].classes_)
collision_type = st.sidebar.selectbox("Collision Type", encoders['collision_type'].classes_)
total_claim_amount = st.sidebar.number_input("Total Claim Amount (£)", min_value=0, value=15000)
age = st.sidebar.slider("Claimant Age", 18, 90, 35)
witnesses = st.sidebar.slider("Number of Witnesses", 0, 5, 1)
police_report = st.sidebar.selectbox("Police Report Available", encoders['police_report_available'].classes_)
authorities_contacted = st.sidebar.selectbox("Authorities Contacted", encoders['authorities_contacted'].classes_)

run_button = st.sidebar.button("Score This Claim", type="primary")

def encode_input(raw_values: dict) -> pd.DataFrame:
    row = {}
    for col in FEATURE_COLUMNS:
        val = raw_values.get(col, 0)
        if col in encoders and isinstance(val, str):
            try:
                val = encoders[col].transform([val])[0]
            except Exception:
                val = 0
        row[col] = val
    return pd.DataFrame([row])[FEATURE_COLUMNS]

def retrieve_guidance(query_text, n_results=2):
    query_embedding = embed_model.encode([query_text]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=n_results)
    return results['documents'][0]

if run_button:
    raw_input = {
        'incident_type': incident_type,
        'incident_severity': incident_severity,
        'collision_type': collision_type,
        'total_claim_amount': total_claim_amount,
        'age': age,
        'witnesses': witnesses,
        'police_report_available': police_report,
        'authorities_contacted': authorities_contacted,
    }
    X = encode_input(raw_input)

    proba = float(best_model.predict_proba(X)[0][1])
    prediction = "⚠️ Fraud Risk" if proba >= 0.5 else "✅ Low Risk"

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Fraud Probability", f"{proba:.1%}")
        st.metric("Assessment", prediction)

    # --- SHAP explanation ---
    shap_values_raw = explainer.shap_values(X)
    if isinstance(shap_values_raw, list):
        shap_vals = shap_values_raw[1][0]
    elif np.array(shap_values_raw).ndim == 3:
        shap_vals = shap_values_raw[0, :, 1]
    else:
        shap_vals = shap_values_raw[0]

    top_factors = pd.DataFrame({
        'feature': FEATURE_COLUMNS,
        'value': X.iloc[0].values,
        'shap_value': shap_vals
    }).sort_values('shap_value', key=abs, ascending=False).head(6)

    with col2:
        st.subheader("Top Factors Driving This Score")
        st.bar_chart(top_factors.set_index('feature')['shap_value'])

    # --- RAG-grounded GenAI investigation summary ---
    st.subheader("🕵️ Investigation Summary")
    with st.spinner("Generating investigation note..."):
        factors_text = "\n".join([
            f"- {row['feature']}: SHAP impact = {row['shap_value']:.3f}"
            for _, row in top_factors.iterrows()
        ])
        query = f"{incident_type} incident, severity {incident_severity}"
        retrieved_docs = retrieve_guidance(query)
        guidance_text = "\n\n".join(retrieved_docs)

        prompt = f"""You are assisting a UK insurance fraud investigator reviewing a flagged claim.

Claim details:
- Predicted fraud probability: {proba:.1%}
- Incident type: {incident_type}
- Incident severity: {incident_severity}
- Total claim amount: £{total_claim_amount}

Top factors driving the model's fraud score:
{factors_text}

Relevant fraud investigation guidance:
{guidance_text}

Write a brief, professional investigation note (4-5 sentences) explaining why this claim was flagged.
Do not make definitive fraud accusations — frame it as "warrants review" language."""

        response = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            max_tokens=350,
            messages=[{"role": "user", "content": prompt}]
        )
        st.write(response.choices[0].message.content)
else:
    st.info("👈 Fill in claim details in the sidebar and click **Score This Claim** to run the full pipeline: ML scoring → SHAP explainability → RAG-grounded investigation note.")
