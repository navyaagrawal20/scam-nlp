import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import streamlit as st
import joblib
from pathlib import Path

# Download required NLTK data (no-op if already present)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Helper: safe model loading
MODEL_PATH = Path("spam_classifier.pkl")
VECT_PATH = Path("tfidf_vectorizer.pkl")

clf = None
vectorizer = None
try:
    if MODEL_PATH.exists() and VECT_PATH.exists():
        clf = joblib.load(str(MODEL_PATH))
        vectorizer = joblib.load(str(VECT_PATH))
    else:
        st.warning("Model or vectorizer not found. Please ensure 'spam_classifier.pkl' and 'tfidf_vectorizer.pkl' are present in the app directory.")
except Exception as e:
    st.error(f"Error loading model files: {e}")

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess(text: str) -> str:
    """Basic preprocessing: tokenize, lowercase, remove stopwords/punctuation, lemmatize."""
    tokens = word_tokenize(str(text).lower())
    tokens = [t for t in tokens if t not in stop_words and t not in string.punctuation]
    lemmatized = []
    for t in tokens:
        try:
            lm = lemmatizer.lemmatize(t)
        except LookupError:
            # WordNet data not available, skip lemmatization
            lm = t
        except Exception:
            lm = t
        lemmatized.append(lm)
    return " ".join(lemmatized)


# ---- Streamlit UI ----
st.set_page_config(page_title="Spam Email Detector", page_icon="📧", layout="wide")

st.title("📧 Spam Email Detection")
st.markdown("Detect whether an email is Spam or Ham (not spam). Enter text, upload a file, or try one of the example messages.")

# Theme toggle (light/dark)
with st.sidebar.expander("Appearance"):
        theme_choice = st.radio("Theme", ["Auto", "Light", "Dark"], index=0)

# CSS for styling; we'll switch classes based on theme_choice
base_css = """
/* Base styles for app */
.header-gradient {
    background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
    color: white;
    padding: 18px;
    border-radius: 8px;
    margin-bottom: 12px;
}
.result-card {
    border-radius: 10px;
    padding: 14px;
    box-shadow: 0 4px 18px rgba(0,0,0,0.08);
}
.spam {
    background: linear-gradient(90deg,#ff8a8a,#ff6b6b);
    color: #5a0000;
}
.ham {
    background: linear-gradient(90deg,#a8ffdf,#69f0ae);
    color: #00331a;
}
.muted {
    color: #6c757d;
}
.prob-bar {
    height: 14px;
    border-radius: 8px;
    background: linear-gradient(90deg,#e9ecef,#f8f9fa);
    overflow: hidden;
}
.prob-fill {
    height: 100%;
    background: linear-gradient(90deg,#007bff,#00d4ff);
}
"""

dark_overrides = """
body {
    background-color: #0f1724;
    color: #e6eef8;
}
.muted { color: #9aa6bb; }
"""

if theme_choice == "Dark":
        st.markdown(f"<style>{base_css}{dark_overrides}</style>", unsafe_allow_html=True)
elif theme_choice == "Light":
        st.markdown(f"<style>{base_css}</style>", unsafe_allow_html=True)
else:
        # Auto: just use base styles and let Streamlit theme control base colors
        st.markdown(f"<style>{base_css}</style>", unsafe_allow_html=True)

# Sidebar for examples and options
with st.sidebar:
    st.header("Try examples")
    example = st.selectbox("Choose an example", ["-- pick an example --",
                                                    "Win a free iPhone now! Click here",
                                                    "Meeting agenda for tomorrow",
                                                    "Your account has been suspended, verify now",
                                                    "Monthly newsletter from our team"]) 
    st.markdown("---")
    st.header("Input options")
    uploaded_file = st.file_uploader("Upload a .txt email file", type=["txt"])
    uploaded_text = ""
    if uploaded_file is not None:
        try:
            raw = uploaded_file.read()
            # try utf-8 then fall back to latin-1
            try:
                uploaded_text = raw.decode('utf-8')
            except Exception:
                uploaded_text = raw.decode('latin-1')
            st.info(f"Loaded file: {uploaded_file.name} ({len(raw)} bytes)")
        except Exception as e:
            st.error(f"Failed to read uploaded file: {e}")
    st.markdown("---")
    st.caption("Model files should be in the app folder: spam_classifier.pkl and tfidf_vectorizer.pkl")


col1, col2 = st.columns([3, 2])

with col1:
    with st.form(key="email_form"):
        # Determine initial value: uploaded file takes precedence, then selected example
        initial_value = uploaded_text if uploaded_text else (example if example and example != "-- pick an example --" else "")
        email_text = st.text_area("Email text", height=250, value=initial_value)
        submitted = st.form_submit_button("Predict")
        clear = st.form_submit_button("Clear")

with col2:
    st.markdown("### Result")
    result_placeholder = st.empty()
    st.markdown("### Details")
    details = st.empty()

if 'email_text' not in locals():
    email_text = ""

if clear:
    # simple clear behaviour
    email_text = ""
    result_placeholder.info("Input cleared. Enter text or select an example.")

if submitted:
    if not email_text:
        result_placeholder.warning("Please enter some email text or choose/upload an example.")
    elif clf is None or vectorizer is None:
        result_placeholder.error("Model not loaded. Prediction not available.")
    else:
        processed_email = preprocess(email_text)
        try:
            email_vec = vectorizer.transform([processed_email])
            pred = clf.predict(email_vec)[0]
            probs = None
            # Try to get probability if available
            if hasattr(clf, "predict_proba"):
                probs = clf.predict_proba(email_vec)[0]
            label = "Spam" if int(pred) == 1 else "Ham"
            spam_prob = None
            if probs is not None:
                try:
                    spam_prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
                except Exception:
                    spam_prob = None

            # Build a styled result card
            card_class = "spam" if label == "Spam" else "ham"
            prob_display = f"{spam_prob*100:.2f}%" if spam_prob is not None else "N/A"

            card_html = f"""
            <div class='result-card {card_class}'>
              <div style='display:flex;justify-content:space-between;align-items:center'>
                <div style='font-size:18px;font-weight:800'>Prediction: {label}</div>
                <div style='text-align:right'>
                  <div class='muted'>Confidence</div>
                  <div style='font-weight:700'>{prob_display}</div>
                </div>
              </div>
              <div style='margin-top:10px'>
                <div class='prob-bar'>
                  <div class='prob-fill' style='width:{(spam_prob*100) if spam_prob is not None else 0}%'></div>
                </div>
              </div>
            </div>
            """

            result_placeholder.markdown(card_html, unsafe_allow_html=True)

            # Details pane
            if spam_prob is not None:
                details.metric("Spam probability", f"{spam_prob*100:.2f}%")
            else:
                details.info("Model does not expose probability scores.")

            st.markdown("---")
            st.subheader("Preprocessed text")
            st.code(processed_email)
        except Exception as e:
            result_placeholder.error(f"Prediction failed: {e}")

