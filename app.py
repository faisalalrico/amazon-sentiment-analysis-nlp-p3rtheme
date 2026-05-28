import streamlit as st
import joblib
import re
import time
from pathlib import Path

model = joblib.load("sentiment_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def sentiment_pred(text):
    cleaned = clean_text(text)
    vectorized = tfidf.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    return "POSITIVE" if prediction == 1 else "NEGATIVE"

st.set_page_config(
    page_title="Sentiment Analysis NLP",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Anton&family=Oswald:wght@400;600;700&display=swap');

.stApp {
    background:
        radial-gradient(circle at 80% 20%, rgba(0,217,255,0.35), transparent 22%),
        radial-gradient(circle at 20% 75%, rgba(0,60,255,0.55), transparent 28%),
        linear-gradient(135deg, #02123f 0%, #004bff 48%, #13d9ff 100%);
    color: white;
}

[data-testid="stHeader"] {background: rgba(0,0,0,0);}
[data-testid="stToolbar"] {display: none;}
[data-testid="stDecoration"] {display: none;}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

.block-container {
    padding-top: 2rem;
    max-width: 1400px;
}

.bg-word {
    position: fixed;
    right: -80px;
    top: 140px;
    font-family: 'Anton', sans-serif;
    font-size: 190px;
    color: rgba(255,255,255,0.10);
    transform: rotate(-62deg);
    z-index: 0;
    pointer-events: none;
}

.bg-word-2 {
    position: fixed;
    left: -120px;
    bottom: -20px;
    font-family: 'Anton', sans-serif;
    font-size: 150px;
    color: rgba(0,0,0,0.22);
    transform: rotate(-90deg);
    z-index: 0;
    pointer-events: none;
}

.diagonal-white {
    position: fixed;
    top: -120px;
    left: 34%;
    width: 180px;
    height: 120vh;
    background: rgba(255,255,255,0.94);
    transform: rotate(12deg);
    z-index: 0;
    pointer-events: none;
}

.diagonal-black {
    position: fixed;
    top: -120px;
    left: -100px;
    width: 240px;
    height: 130vh;
    background: rgba(0,0,0,0.70);
    transform: rotate(-18deg);
    z-index: 0;
    pointer-events: none;
}

.triangle-1 {
    position: fixed;
    right: 180px;
    bottom: 70px;
    width: 0;
    height: 0;
    border-left: 90px solid transparent;
    border-right: 90px solid transparent;
    border-bottom: 150px solid rgba(255,43,191,0.75);
    transform: rotate(25deg);
    z-index: 0;
    pointer-events: none;
}

.triangle-2 {
    position: fixed;
    right: 90px;
    bottom: 160px;
    width: 0;
    height: 0;
    border-left: 55px solid transparent;
    border-right: 55px solid transparent;
    border-bottom: 95px solid rgba(0,217,255,0.75);
    transform: rotate(-25deg);
    z-index: 0;
    pointer-events: none;
}

.top-hud {
    position: fixed;
    top: 20px;
    right: 35px;
    z-index: 5;
    text-align: right;
    font-family: 'Anton', sans-serif;
}

.hud-date {
    font-size: 56px;
    line-height: 0.9;
    color: white;
    text-shadow: 5px 5px 0 #003CFF;
}

.hud-small {
    font-family: 'Oswald', sans-serif;
    font-size: 18px;
    color: #DDFBFF;
    font-weight: 700;
}

.hud-mode {
    display: inline-block;
    margin-top: 6px;
    padding: 4px 18px;
    background: rgba(0,0,0,0.55);
    border-right: 8px solid #00D9FF;
    font-family: 'Oswald', sans-serif;
    font-size: 15px;
}

.logo-box img {
    filter: drop-shadow(0px 0px 12px rgba(0,217,255,0.8))
            drop-shadow(5px 5px 0px rgba(0,0,0,0.7));
}

.tag-label {
    display: inline-block;
    background: black;
    color: white;
    font-family: 'Oswald', sans-serif;
    font-weight: 700;
    font-size: 18px;
    padding: 4px 18px;
    transform: rotate(-8deg);
    border-left: 8px solid #00D9FF;
    margin-bottom: 12px;
}

.big-title {
    font-family: 'Anton', sans-serif;
    font-size: 92px;
    line-height: 0.86;
    letter-spacing: 1px;
    color: white;
    text-transform: uppercase;
    text-shadow:
        8px 8px 0 #003CFF,
        13px 13px 0 rgba(0,0,0,0.7);
}

.sub-title {
    margin-top: 10px;
    font-family: 'Oswald', sans-serif;
    font-size: 24px;
    font-weight: 600;
    color: white;
    background: rgba(0,0,0,0.65);
    padding: 8px 18px;
    width: fit-content;
    border-right: 12px solid white;
    box-shadow: 8px 8px 0 #003CFF;
}

.system-text {
    margin-top: 28px;
    font-family: 'Oswald', sans-serif;
    font-size: 16px;
    color: #E7FCFF;
    line-height: 1.7;
    background: rgba(0,0,0,0.45);
    padding: 18px;
    border-left: 8px solid #00D9FF;
    max-width: 520px;
}

.panel {
    background: rgba(0, 8, 42, 0.88);
    border: 3px solid white;
    padding: 28px;
    transform: skew(-3deg);
    box-shadow:
        16px 16px 0 rgba(0,0,0,0.75),
        -10px -10px 0 rgba(0,217,255,0.35);
}

.panel-inner {
    transform: skew(3deg);
}

.panel-title {
    font-family: 'Anton', sans-serif;
    font-size: 44px;
    line-height: 0.95;
    color: white;
}

.panel-badge {
    font-family: 'Oswald', sans-serif;
    font-size: 14px;
    background: white;
    color: #003CFF;
    padding: 6px 14px;
    font-weight: 700;
    box-shadow: 5px 5px 0 #00D9FF;
    width: fit-content;
    margin-bottom: 20px;
}

label {
    color: white !important;
    font-family: 'Oswald', sans-serif !important;
    font-size: 18px !important;
    font-weight: 700 !important;
}

textarea {
    background: rgba(255,255,255,0.96) !important;
    color: #020B3A !important;
    border: 3px solid white !important;
    border-radius: 0px !important;
    font-family: 'Oswald', sans-serif !important;
    font-size: 18px !important;
    font-weight: 600 !important;
    box-shadow: 8px 8px 0 #003CFF !important;
}

.stButton > button {
    margin-top: 12px;
    background: white;
    color: #003CFF;
    border: 3px solid white;
    border-radius: 0px;
    font-family: 'Anton', sans-serif;
    font-size: 24px;
    text-transform: uppercase;
    width: 100%;
    padding: 0.45rem 1rem;
    box-shadow: 8px 8px 0 #003CFF;
}

.stButton > button:hover {
    background: #00D9FF;
    color: white;
    transform: translate(-3px, -3px);
    box-shadow: 12px 12px 0 white;
}

.result-card {
    margin-top: 22px;
    padding: 18px 22px;
    border: 3px solid white;
    font-family: 'Anton', sans-serif;
    font-size: 42px;
    text-transform: uppercase;
    box-shadow: 10px 10px 0 rgba(0,0,0,0.75);
}

.result-positive {
    background: linear-gradient(90deg, white 0%, #DDFBFF 60%, #00D9FF 100%);
    color: #003CFF;
    border-left: 18px solid #70FF00;
}

.result-negative {
    background: linear-gradient(90deg, #03051f 0%, #061B65 60%, #FF263A 100%);
    color: white;
    border-left: 18px solid #FF263A;
}

.result-small {
    font-family: 'Oswald', sans-serif;
    font-size: 15px;
    margin-top: 6px;
    text-transform: none;
}

.command-bar {
    position: fixed;
    left: 30px;
    bottom: 22px;
    z-index: 4;
    display: flex;
    gap: 18px;
    align-items: center;
    font-family: 'Oswald', sans-serif;
    font-weight: 700;
    color: white;
}

.key {
    display: inline-flex;
    width: 31px;
    height: 31px;
    border-radius: 999px;
    border: 3px solid white;
    align-items: center;
    justify-content: center;
    background: #003CFF;
    color: white;
    margin-right: 6px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="diagonal-black"></div>
<div class="diagonal-white"></div>
<div class="bg-word">NLP</div>
<div class="bg-word-2">SENTIMENT</div>
<div class="triangle-1"></div>
<div class="triangle-2"></div>

<div class="top-hud">
    <div class="hud-date">NLP<br>2026</div>
    <div class="hud-small">Natural Language Processing</div>
    <div class="hud-mode">Amazon Sentiment Classifier</div>
</div>
""", unsafe_allow_html=True)

logo_path = Path("Persona_3_Reload_logo.png")
if logo_path.exists():
    st.markdown('<div class="logo-box">', unsafe_allow_html=True)
    st.image(str(logo_path), width=180)
    st.markdown('</div>', unsafe_allow_html=True)

left, right = st.columns([0.95, 1.2], gap="large")

with left:
    st.markdown('<div class="tag-label">NATURAL LANGUAGE PROCESSING</div>', unsafe_allow_html=True)
    st.markdown('<div class="big-title">SENTIMENT<br>ANALYSIS</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">What does the review feel?</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="system-text">
        <b>Nama:</b> Faisal Alrico<br>
        <b>NIM:</b> 23416255201156<br>
        <b>Kelas:</b> IF23G<br>
        <b>Mata Kuliah:</b> Natural Language Processing<br><br>

        Sistem ini melakukan klasifikasi sentimen review produk Amazon
        menjadi dua kategori utama: <b>Positive</b> dan <b>Negative</b>.<br><br>

        Model menggunakan metode ekstraksi fitur <b>TF-IDF</b>
        dan algoritma <b>Logistic Regression</b>.
    </div>
    """, unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel"><div class="panel-inner">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Review<br>Scanner</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-badge">STATUS : READY</div>', unsafe_allow_html=True)

    user_input = st.text_area(
        "Masukkan teks ulasan produk:",
        placeholder="Example: I love this phone, battery lasts forever!",
        height=170
    )

    analyze = st.button("Analyze Sentiment")

    if analyze:
        if user_input.strip() == "":
            st.warning("Silakan masukkan teks terlebih dahulu.")
        else:
            with st.spinner("Analyzing review pattern..."):
                time.sleep(0.5)

            result = sentiment_pred(user_input)

            if result == "POSITIVE":
                st.markdown("""
                <div class="result-card result-positive">
                    POSITIVE
                    <div class="result-small">
                        The review indicates a good user experience.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-card result-negative">
                    NEGATIVE
                    <div class="result-small">
                        The review indicates dissatisfaction or complaint.
                    </div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown('</div></div>', unsafe_allow_html=True)

st.markdown("""
<div class="command-bar">
    <div><span class="key">A</span>Analyze</div>
    <div><span class="key">B</span>Review</div>
    <div><span class="key">Y</span>NLP</div>
</div>
""", unsafe_allow_html=True)