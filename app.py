import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="Movie Review Sentiment Analysis",
    layout="wide"
)

# ----------------------------
# LOAD MODELS
# ----------------------------
@st.cache_resource
def load_models():

    rnn_model = tf.keras.models.load_model("simplernn.h5")
    lstm_model = tf.keras.models.load_model("lstm_model.h5")
    gru_model = tf.keras.models.load_model("gru_model.h5")

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    return rnn_model, lstm_model, gru_model, tokenizer

rnn_model, lstm_model, gru_model, tokenizer = load_models()

MAX_LEN = 200

# ----------------------------
# PREPROCESS FUNCTION
# ----------------------------
def preprocess_text(text):

    seq = tokenizer.texts_to_sequences([text])

    padded = pad_sequences(
        seq,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    return padded


# ----------------------------
# PREDICTION FUNCTION
# ----------------------------
def predict_sentiment(model, review):

    processed = preprocess_text(review)

    prediction = model.predict(processed, verbose=0)[0][0]

    positive_prob = float(prediction)
    negative_prob = 1 - positive_prob

    sentiment = "Positive" if positive_prob >= 0.5 else "Negative"

    confidence = max(positive_prob, negative_prob) * 100

    return sentiment, confidence, positive_prob, negative_prob


# ----------------------------
# HEADER
# ----------------------------
st.title("🎬 Movie Review Sentiment Analysis System")

st.subheader(
    "Deep Learning Based Sentiment Classification"
)

st.markdown("---")

# ----------------------------
# MODEL SELECTION
# ----------------------------
selected_model = st.selectbox(
    "Select Model",
    ["SimpleRNN", "LSTM", "GRU"]
)

# ----------------------------
# REVIEW INPUT
# ----------------------------
review = st.text_area(
    "Enter your movie review here...",
    height=200
)

# ----------------------------
# BUTTON
# ----------------------------
if st.button("Analyze Review"):

    if review.strip() == "":
        st.warning("Please enter a review.")
        st.stop()

    # ------------------------
    # SELECT MODEL
    # ------------------------
    if selected_model == "SimpleRNN":
        model = rnn_model

    elif selected_model == "LSTM":
        model = lstm_model

    else:
        model = gru_model

    sentiment, confidence, pos_prob, neg_prob = predict_sentiment(
        model,
        review
    )

    # ------------------------
    # OUTPUT
    # ------------------------
    st.markdown("## Prediction Result")

    st.success(f"Sentiment: {sentiment}")

    st.info(f"Confidence: {confidence:.2f}%")

    st.markdown("---")

    # ------------------------
    # PROBABILITY CHART
    # ------------------------
    st.markdown("## Probability Distribution")

    chart_data = pd.DataFrame({
        "Class": ["Positive", "Negative"],
        "Probability": [pos_prob, neg_prob]
    })

    fig, ax = plt.subplots(figsize=(5,3))
    ax.bar(
        chart_data["Class"],
        chart_data["Probability"]
    )

    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")

    st.pyplot(fig)

    # ------------------------
    # MODEL COMPARISON
    # ------------------------
    st.markdown("---")
    st.markdown("## Compare All Models")

    models = {
        "SimpleRNN": rnn_model,
        "LSTM": lstm_model,
        "GRU": gru_model
    }

    results = []

    for name, mdl in models.items():

        sentiment, confidence, pos, neg = predict_sentiment(
            mdl,
            review
        )

        results.append(
            [
                name,
                sentiment,
                round(confidence, 2),
                round(pos * 100, 2),
                round(neg * 100, 2)
            ]
        )

    comparison_df = pd.DataFrame(
        results,
        columns=[
            "Model",
            "Sentiment",
            "Confidence (%)",
            "Positive Probability (%)",
            "Negative Probability (%)"
        ]
    )

    st.dataframe(
        comparison_df,
        use_container_width=True
    )

    # ------------------------
    # CONFIDENCE CHART
    # ------------------------
    st.markdown("## Confidence Comparison")

    fig2, ax2 = plt.subplots(figsize=(7,4))

    ax2.bar(
        comparison_df["Model"],
        comparison_df["Confidence (%)"]
    )

    ax2.set_ylabel("Confidence (%)")
    ax2.set_ylim(0, 100)

    st.pyplot(fig2)