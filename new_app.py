import streamlit as st
import pandas as pd
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# ✅ Set Streamlit Page Config First
st.set_page_config(page_title="Emotion Detector (RoBERTa)", page_icon="🧬")

# ✅ Load Emotion Model
@st.cache_resource
def load_model():
    model_name = "bhadresh-savani/distilbert-base-uncased-emotion"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)

classifier = load_model()

# ✅ Emoji Map for Emotions
EMOJI_MAP = {
    "anger": "😡 🔥",
    "joy": "😄 😊",
    "optimism": "🌈 ☀️",
    "sadness": "😢 💔",
    "love": "❤️ 💖",
    "admiration": "👏 😍",
    "gratitude": "🙏 💝",
    "approval": "👍 ✅",
    "realization": "💡 🤯",
    "annoyance": "😤 😑",
    "disappointment": "😞 😔",
    "relief": "😌 🍃",
    "neutral": "😐 🤔",
    "uncertain": "⚠️"
}

# ✅ Profanity Detection
PROFANE_WORDS = [
    "fuck", "f*ck", "f***", "shit", "bitch", "asshole", "dick", "bastard",
    "wtf", "mf", "motherfucker", "a-hole"
]

def is_profane(text):
    return any(re.search(rf'\b{re.escape(word)}\b', text.lower()) for word in PROFANE_WORDS)

# ✅ App Title and Description
st.title("🎭 Emotion Detection from Text")
st.markdown("Analyze sentences or upload a CSV to detect emotions using a BERT-based model fine-tuned on emotional text.")

# === Input Tabs ===
tab1, tab2 = st.tabs(["📝 Type Text", "📁 Upload CSV"])

# === Tab 1: Text Input ===
with tab1:
    st.subheader("🔤 Enter Your Text")
    text_input = st.text_area("Write something here:", height=150)

    if st.button("🔍 Detect Emotion"):
        if not text_input.strip():
            st.warning("Please enter some text.")
        elif is_profane(text_input):
            st.warning("⚠️ This input contains offensive language. Emotion prediction is skipped.")
            st.error("**Top Prediction:** UNCERTAIN ⚠️")
        else:
            result = classifier(text_input)[0]
            sorted_result = sorted(result, key=lambda x: x["score"], reverse=True)
            top_label = sorted_result[0]["label"]
            top_score = sorted_result[0]["score"]
            emoji = EMOJI_MAP.get(top_label.lower(), "🤔")
            st.success(f"**Top Prediction:** {top_label.upper()} {emoji} ({top_score:.2%})")

            st.subheader("🧠 Mixed Emotion Analysis:")
            for item in sorted_result[:3]:
                label = item["label"]
                score = item["score"]
                emoji = EMOJI_MAP.get(label.lower(), "🤔")
                st.write(f"{label.upper()} {emoji} — {score:.2%}")

# === Tab 2: CSV Upload ===
with tab2:
    st.subheader("📄 Upload CSV File")
    uploaded_file = st.file_uploader("Upload a CSV with a 'text' column", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)

            if 'text' not in df.columns:
                st.error("❌ The CSV must contain a 'text' column.")
            elif df.empty:
                st.warning("⚠️ The uploaded file is empty.")
            else:
                def get_top_emotion(text):
                    if is_profane(text):
                        return "uncertain", 0.0
                    result = classifier(text)[0]
                    top = max(result, key=lambda x: x["score"])
                    return top["label"], round(top["score"] * 100, 2)

                df[['predicted_emotion', 'confidence']] = df['text'].apply(
                    lambda x: pd.Series(get_top_emotion(x))
                )
                df['emoji'] = df['predicted_emotion'].map(lambda x: EMOJI_MAP.get(x.lower(), "🤔"))

                st.dataframe(df[['text', 'predicted_emotion', 'confidence', 'emoji']])

                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Results", csv, "emotion_predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Error processing file: {e}")