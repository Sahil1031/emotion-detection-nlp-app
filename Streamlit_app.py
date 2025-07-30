import streamlit as st
import joblib
import pandas as pd

# -----------------------------------------------------------------------------
# App Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Emotion Detection — NLP Demo",
    page_icon="🎭",
    layout="centered",
    initial_sidebar_state="expanded",
)
st.title("🎭 Emotion Detection from Text")
st.markdown(
    """
    <span style='font-size:1.1em;'>A professional NLP demo web app that automatically predicts <b>emotions</b> from user text.
    Designed as a technical showcase for interviews, portfolios, and rapid prototyping, powered by scikit-learn and Streamlit.</span>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Model and Vectorizer Loading
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model & vectorizer...")
def load_artifacts():
    vectorizer = joblib.load("bow_vectorizer.pkl")
    model = joblib.load("logistic_model.pkl")
    return vectorizer, model

vectorizer, model = load_artifacts()

# -----------------------------------------------------------------------------
# Label Mapping & Color Scheme
# -----------------------------------------------------------------------------
label_to_emotion = {
    0: 'sadness', 1: 'anger', 2: 'love',
    3: 'surprise', 4: 'fear', 5: 'joy'
}
emotion_colors = {
    'sadness': '#1f77b4', 'anger': '#d62728', 'love': '#e377c2',
    'surprise': '#9467bd', 'fear': '#8c564b', 'joy': '#ff7f0e'
}

# -----------------------------------------------------------------------------
# Sidebar: App Details and Features
# -----------------------------------------------------------------------------
st.sidebar.header("About this App")
st.sidebar.info(
    """
    - **Purpose:** Automatic emotion detection using NLP.
    - **Model:** Logistic Regression (Bag-of-Words, scikit-learn).
    - **Technologies:** Python, Streamlit, sklearn, pandas.
    - **Features:** 
        - Clean  UI.
        - Single sentence prediction with model probabilities.
        - Interactive sample dataset explainer.
        - Download label mappings for integration.
    """
)
st.sidebar.subheader("Emotion Labels")
sidebar_df = pd.DataFrame({
    "Label": list(label_to_emotion.keys()),
    "Emotion": list(label_to_emotion.values())
})
st.sidebar.table(sidebar_df)

# -----------------------------------------------------------------------------
# Main: User Input (Single Prediction)
# -----------------------------------------------------------------------------
st.header("🔍 Predict Emotion")
with st.form("predict_form"):
    user_text = st.text_area("Type a sentence:", "", placeholder="Type something...", height=80)
    submit = st.form_submit_button("Predict Emotion")

if submit and user_text.strip():
    X_new = vectorizer.transform([user_text])
    pred_label = int(model.predict(X_new)[0])
    pred_emotion = label_to_emotion.get(pred_label, "Unknown")
    proba = model.predict_proba(X_new)[0]

    st.markdown(
        f"<h3 style='color:{emotion_colors.get(pred_emotion, 'black')}'>{pred_emotion.upper()}</h3>"
        f"<span>Model label: <b>{pred_label}</b></span>",
        unsafe_allow_html=True,
    )
    proba_df = pd.DataFrame({
        'Emotion': [label_to_emotion[i] for i in range(len(proba))],
        'Probability': proba
    }).set_index('Emotion').sort_values('Probability', ascending=False)
    st.bar_chart(proba_df)
    st.caption("Model confidence scores for each emotion.")

elif submit:
    st.warning("Please enter some text to get predictions.")

# -----------------------------------------------------------------------------
# Example Sentences for Recruiters
# -----------------------------------------------------------------------------
st.markdown("**Try these example sentences:**")
examples = [
    "I feel amazing today.",
    "I'm so angry about what happened!",
    "Nothing makes me more sad than this.",
    "What a wonderful surprise!",
    "I'm really afraid of failing.",
    "She means so much to me."
]
ex_idx = st.selectbox(
    "Choose an example:",
    options=list(range(len(examples))),
    format_func=lambda i: examples[i],
)
if st.button("Autofill Example Sentence"):
    st.experimental_set_query_params(text=examples[ex_idx])
    st.experimental_rerun()

# -----------------------------------------------------------------------------
# Interactive: Sample Dataset Explorer
# -----------------------------------------------------------------------------
st.header("🔬 Sample Datasets Gallery")

# Sample datasets dictionary — Add more datasets here for demonstration!
sample_datasets = {
    "Short Example (6 samples)": pd.DataFrame({
        "text": examples,
        "emotion": [5, 1, 0, 3, 4, 2]
    }),
    "IMDB Movie Reviews Subset": pd.read_csv("sample_imdb_emotions.csv") if "sample_imdb_emotions.csv" in st.session_state else pd.DataFrame(),
    "Custom Demo (processed.csv)": pd.read_csv("processed.csv") if "processed.csv" in st.session_state else pd.DataFrame(),
}

chosen_dataset = st.selectbox("Select a sample dataset to view:", list(sample_datasets.keys()))
df_demo = sample_datasets[chosen_dataset]

if not df_demo.empty:
    filter_val = st.selectbox(
        "Filter by Emotion:", ["All"] + list(label_to_emotion.values()),
        key="filter_"+chosen_dataset
    )
    if filter_val != "All" and "emotion" in df_demo.columns:
        df_demo["emotion_name"] = df_demo["emotion"].apply(lambda x: label_to_emotion[int(x)] if pd.notnull(x) and str(x).isdigit() else x)
        filtered_df = df_demo[df_demo.emotion_name == filter_val]
    else:
        filtered_df = df_demo
    show_cols = ["text", "emotion_name"] if "emotion_name" in filtered_df.columns else ["text", "emotion"]
    st.write(filtered_df[show_cols].head(10) if not filtered_df.empty else "No data for this filter.")
else:
    st.info("No sample data found for this dataset (please add file to app directory if needed).")

# ====== Professional Footer ======
st.markdown("""
---
*Developed with ❤️ by Sahil Bagde — 2025*
""")

