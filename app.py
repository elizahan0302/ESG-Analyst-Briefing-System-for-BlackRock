# app.py
# BlackRock ESG Risk Analyzer
# ISOM5240 Educational Prototype

from __future__ import annotations

import re
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import torch
from transformers import pipeline

# Optional document extraction libraries
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    from docx import Document
except ImportError:
    Document = None


# =========================================================
# Page Configuration
# =========================================================

st.set_page_config(
    page_title="BlackRock ESG Risk Analyzer",
    page_icon="◌",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# Design System
# Muted / Pastel / Monochromatic
# Neo-minimalism + Card-based Layered UI
# =========================================================

st.markdown(
    """
    <style>
    :root {
        --bg: #f6f4f0;
        --bg-soft: #fbfaf7;
        --card: #ffffff;
        --card-muted: #f2f0ea;
        --ink: #2f302d;
        --ink-soft: #5f625b;
        --muted: #8b8e86;
        --line: #e5e1d8;
        --line-strong: #d7d2c6;
        --sage: #aeb8a2;
        --sage-deep: #526459;
        --sage-dark: #536a5f;
        --sage-soft: #e9eee4;
        --clay: #c8b8a6;
        --clay-soft: #efe7dd;
        --stone: #d7d4cd;
        --stone-soft: #f1efeb;
        --green: #8fa891;
        --green-soft: #e6eee5;
        --amber: #b9a16b;
        --amber-soft: #f4eedc;
        --red: #b7837f;
        --red-soft: #f2e5e3;
        --shadow: 0 18px 45px rgba(47, 48, 45, 0.08);
        --shadow-soft: 0 10px 25px rgba(47, 48, 45, 0.05);
        --radius-lg: 28px;
        --radius-md: 20px;
        --radius-sm: 14px;
    }

    .stApp {
        background: radial-gradient(circle at top left, #f0ede6 0, #f6f4f0 32%, #fbfaf7 100%);
        color: var(--ink);
    }

    html, body, [class*="css"] {
        font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    h1, h2, h3, h4 {
        color: var(--ink);
        letter-spacing: -0.03em;
    }

    p, li, label, span {
        color: var(--ink-soft);
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 3rem;
        max-width: 1220px;
    }

    /* Prevent sidebar from being visually used */
    section[data-testid="stSidebar"] {
        display: none;
    }

    /* Buttons */
    div.stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #536a5f 0%, #6f8178 100%);
        color: #fbfaf7;
        border: 1px solid rgba(83, 106, 95, 0.9);
        border-radius: 16px;
        padding: 0.92rem 1.2rem;
        font-weight: 680;
        letter-spacing: 0.01em;
        box-shadow: 0 10px 24px rgba(47, 48, 45, 0.12);
        transition: all 0.18s ease;
    }

    div.stButton > button:hover {
        background: linear-gradient(135deg, #465a51 0%, #61736a 100%);
        border-color: #465a51;
        color: #ffffff;
        transform: translateY(-1px);
        box-shadow: 0 14px 30px rgba(47, 48, 45, 0.16);
    }

    div.stDownloadButton > button {
        width: 100%;
        background: #f8f6f1;
        color: var(--ink);
        border: 1px solid var(--line-strong);
        border-radius: 16px;
        padding: 0.82rem 1.1rem;
        font-weight: 620;
        transition: all 0.18s ease;
    }

    div.stDownloadButton > button:hover {
        background: #efebe2;
        border-color: var(--ink-soft);
        color: var(--ink);
    }

    /* Inputs */
    textarea {
        min-height: 175px !important;
        border-radius: 22px !important;
        border: 1px solid var(--line) !important;
        background: #ffffff !important;
        color: var(--ink) !important;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.7);
    }

    textarea:focus {
        border-color: #aaa596 !important;
        box-shadow: 0 0 0 3px rgba(174, 184, 162, 0.22) !important;
    }

    [data-testid="stFileUploader"] section {
        border-radius: 22px;
        border: 1.5px dashed #cfc9bd;
        background: #ffffff;
        padding: 1.2rem;
    }

    /* Hero */
    .top-brand {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.8rem;
    }

    .brand-lockup {
        display: flex;
        align-items: center;
        gap: 0.8rem;
    }

    .brand-mark {
        width: 2.55rem;
        height: 2.55rem;
        border-radius: 999px;
        background: var(--sage-soft);
        display: flex;
        align-items: center;
        justify-content: center;
        color: var(--sage-dark);
        font-size: 1.25rem;
        border: 1px solid rgba(215,210,198,0.8);
    }

    .brand-text-main {
        color: var(--ink);
        font-size: 1.05rem;
        font-weight: 760;
        line-height: 1.05;
    }

    .brand-text-sub {
        color: var(--ink-soft);
        font-size: 0.92rem;
        line-height: 1.1;
    }

    .prototype-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: rgba(255,255,255,0.58);
        border: 1px solid var(--line);
        border-radius: 999px;
        padding: 0.55rem 0.85rem;
        font-size: 0.86rem;
        color: var(--ink-soft);
        font-weight: 620;
    }

    .hero-shell {
        position: relative;
        background: linear-gradient(135deg, #efebe2 0%, #f8f6f1 52%, #e9eee4 100%);
        border: 1px solid rgba(215, 210, 198, 0.9);
        border-radius: 30px;
        padding: 2.1rem;
        margin-bottom: 1rem;
        box-shadow: var(--shadow);
        overflow: hidden;
    }

    .hero-shell:before {
        content: "";
        position: absolute;
        width: 360px;
        height: 360px;
        right: -105px;
        top: -115px;
        background: rgba(174, 184, 162, 0.22);
        border-radius: 50%;
    }

    .hero-shell:after {
        content: "";
        position: absolute;
        width: 260px;
        height: 260px;
        right: 130px;
        bottom: -145px;
        background: rgba(200, 184, 166, 0.20);
        border-radius: 50%;
    }

    .hero-content {
        position: relative;
        z-index: 1;
    }

    .eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.35rem 0.75rem;
        border: 1px solid rgba(47,48,45,0.10);
        border-radius: 999px;
        background: rgba(255,255,255,0.52);
        color: var(--ink-soft);
        font-size: 0.8rem;
        font-weight: 680;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        margin-bottom: 0.95rem;
    }

    .hero-title {
        font-size: clamp(2.25rem, 4.8vw, 4.5rem);
        line-height: 0.96;
        font-weight: 780;
        letter-spacing: -0.07em;
        color: var(--ink);
        margin-bottom: 0.9rem;
    }

    .hero-subtitle {
        font-size: 1.14rem;
        line-height: 1.55;
        color: var(--sage-deep);
        max-width: 720px;
        font-weight: 550;
        margin-bottom: 0.9rem;
    }

    .hero-description {
        font-size: 0.98rem;
        line-height: 1.6;
        color: var(--ink-soft);
        max-width: 720px;
    }

    .workflow-card {
        position: relative;
        z-index: 1;
        background: rgba(255,255,255,0.76);
        border: 1px solid rgba(215,210,198,0.95);
        border-radius: 24px;
        padding: 1.25rem;
        box-shadow: var(--shadow-soft);
        backdrop-filter: blur(10px);
    }

    .workflow-title {
        color: var(--ink);
        font-size: 1rem;
        font-weight: 740;
        margin-bottom: 0.9rem;
    }

    .workflow-step {
        display: flex;
        gap: 0.72rem;
        align-items: flex-start;
        padding: 0.65rem 0;
        border-bottom: 1px solid rgba(215,210,198,0.55);
        color: var(--ink-soft);
        font-size: 0.92rem;
    }

    .workflow-step:last-child {
        border-bottom: none;
    }

    .step-index {
        flex: 0 0 auto;
        width: 1.6rem;
        height: 1.6rem;
        border-radius: 999px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: var(--sage-soft);
        color: var(--ink);
        font-size: 0.78rem;
        font-weight: 720;
    }

    /* Context cards */
    .context-card {
        min-height: 132px;
        background: rgba(255,255,255,0.78);
        border: 1px solid var(--line);
        border-radius: 24px;
        padding: 1.1rem;
        box-shadow: var(--shadow-soft);
        display: flex;
        gap: 1rem;
        align-items: flex-start;
    }

    .context-icon {
        width: 3.2rem;
        height: 3.2rem;
        border-radius: 999px;
        background: var(--stone-soft);
        display: flex;
        align-items: center;
        justify-content: center;
        color: var(--sage-dark);
        font-size: 1.2rem;
        flex: 0 0 auto;
        border: 1px solid rgba(215,210,198,0.7);
    }

    .context-title {
        font-size: 1rem;
        font-weight: 740;
        color: var(--ink);
        margin-bottom: 0.35rem;
    }

    .context-text {
        font-size: 0.88rem;
        line-height: 1.52;
        color: var(--ink-soft);
    }

    /* Input Studio */
    .input-studio {
        background: rgba(255, 255, 255, 0.84);
        border: 1px solid #e5e1d8;
        border-radius: 30px;
        padding: 1.35rem;
        box-shadow: 0 18px 45px rgba(47, 48, 45, 0.07);
        margin-top: 1rem;
        margin-bottom: 1.2rem;
    }

    .input-studio-title {
        font-size: 1.45rem;
        font-weight: 760;
        letter-spacing: -0.04em;
        color: #2f302d;
        margin-bottom: 0.25rem;
    }

    .input-studio-subtitle {
        font-size: 0.96rem;
        color: #6b6f66;
        line-height: 1.55;
        max-width: 860px;
    }

    .mode-caption {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #8b8e86;
        font-weight: 760;
        margin-top: 1.05rem;
        margin-bottom: 0.45rem;
    }

    div[role="radiogroup"] {
        background: #f1efeb;
        border: 1px solid #e2ded5;
        border-radius: 18px;
        padding: 0.28rem;
        width: fit-content;
        gap: 0.2rem;
        margin-bottom: 1rem;
    }

    div[role="radiogroup"] label {
        background: transparent;
        border-radius: 14px;
        padding: 0.55rem 1.1rem;
        border: 1px solid transparent;
        transition: all 0.18s ease;
        color: var(--ink-soft);
    }

    div[role="radiogroup"] label:hover {
        background: rgba(255,255,255,0.72);
        border-color: #ddd8cd;
    }

    .module-card {
        background: #fbfaf7;
        border: 1px solid #e5e1d8;
        border-radius: 26px;
        padding: 1.25rem;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.7);
    }

    .module-title-row {
        display: flex;
        align-items: center;
        gap: 0.72rem;
        margin-bottom: 0.25rem;
    }

    .module-icon {
        width: 2.25rem;
        height: 2.25rem;
        border-radius: 999px;
        background: #e9eee4;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #526459;
        font-size: 1rem;
        flex: 0 0 auto;
    }

    .module-title {
        font-size: 1.15rem;
        font-weight: 740;
        letter-spacing: -0.03em;
        color: #2f302d;
    }

    .module-subtitle {
        color: #6b6f66;
        font-size: 0.93rem;
        line-height: 1.55;
        margin-bottom: 1rem;
    }

    .input-help-note {
        font-size: 0.85rem;
        color: #7a7d75;
        line-height: 1.45;
        margin-top: 0.65rem;
        background: var(--stone-soft);
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 0.75rem 0.85rem;
    }

    /* Results */
    .result-shell {
        background: rgba(255,255,255,0.92);
        border: 1px solid var(--line);
        border-radius: 30px;
        padding: 1.35rem;
        box-shadow: var(--shadow);
        margin-top: 1.25rem;
    }

    .result-title {
        font-size: 1.55rem;
        font-weight: 760;
        letter-spacing: -0.04em;
        color: var(--ink);
        margin-bottom: 1rem;
    }

    .metric-card {
        background: var(--bg-soft);
        border: 1px solid var(--line);
        border-radius: 20px;
        padding: 1rem;
        min-height: 125px;
    }

    .metric-label {
        color: var(--muted);
        font-size: 0.78rem;
        font-weight: 720;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.55rem;
    }

    .metric-value {
        color: var(--ink);
        font-size: 1.35rem;
        font-weight: 760;
        letter-spacing: -0.04em;
        margin-bottom: 0.35rem;
        word-break: break-word;
    }

    .metric-caption {
        color: var(--ink-soft);
        font-size: 0.86rem;
        line-height: 1.4;
    }

    .risk-card {
        border-radius: 22px;
        padding: 1.05rem 1.1rem;
        margin-top: 1.05rem;
        border: 1px solid var(--line);
    }

    .risk-low {
        background: var(--green-soft);
        border-color: rgba(143,168,145,0.45);
    }

    .risk-medium {
        background: var(--amber-soft);
        border-color: rgba(185,161,107,0.45);
    }

    .risk-high {
        background: var(--red-soft);
        border-color: rgba(183,131,127,0.45);
    }

    .risk-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--muted);
        font-weight: 780;
        margin-bottom: 0.35rem;
    }

    .risk-value {
        color: var(--ink);
        font-size: 1.85rem;
        font-weight: 780;
        letter-spacing: -0.05em;
        margin-bottom: 0.2rem;
    }

    .interpretation-card {
        background: var(--bg-soft);
        border: 1px solid var(--line);
        border-radius: 22px;
        padding: 1.05rem;
        margin-top: 1rem;
    }

    .interpretation-title {
        color: var(--ink);
        font-weight: 730;
        margin-bottom: 0.45rem;
        font-size: 1rem;
    }

    .interpretation-text {
        color: var(--ink-soft);
        line-height: 1.62;
        font-size: 0.94rem;
    }

    .footer-note {
        color: var(--muted);
        font-size: 0.88rem;
        text-align: center;
        margin-top: 1.8rem;
        padding-top: 1.1rem;
        border-top: 1px solid var(--line);
    }

    @media (max-width: 900px) {
        div[role="radiogroup"] {
            width: 100%;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# Path Handling
# =========================================================

def get_project_root() -> Path:
    """
    Locate the project root robustly.
    app.py is expected to be inside 03_app, but the app should also work
    when launched from project root or inside 03_app.
    """
    try:
        file_path = Path(__file__).resolve()
        if file_path.parent.name == "03_app":
            return file_path.parent.parent
    except NameError:
        pass

    cwd = Path.cwd().resolve()

    if (cwd / "03_app").exists() and (cwd / "04_models").exists():
        return cwd

    if cwd.name == "03_app":
        return cwd.parent

    return cwd


PROJECT_ROOT = get_project_root()
LOCAL_ESG_MODEL_DIR = PROJECT_ROOT / "04_models" / "blackrock_esg_classifier"


# =========================================================
# Utility Functions
# =========================================================

def get_device() -> Tuple[int, str]:
    """Return Hugging Face pipeline device ID and readable device name."""
    if torch.cuda.is_available():
        return 0, "GPU"
    return -1, "CPU"


def clean_input_text(text: str) -> str:
    """Clean user input by removing repeated spaces and line breaks."""
    if text is None:
        return ""
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_esg_label(label: str) -> str:
    """Normalize ESG model output labels."""
    label = str(label).strip()

    mapping = {
        "Environmental": "Environmental",
        "environmental": "Environmental",
        "Social": "Social",
        "social": "Social",
        "Governance": "Governance",
        "governance": "Governance",
        "None": "Non-ESG",
        "none": "Non-ESG",
        "Non-ESG": "Non-ESG",
        "non-esg": "Non-ESG",
        "Non ESG": "Non-ESG",
        "non esg": "Non-ESG",
    }

    return mapping.get(label, label)


def normalize_sentiment_label(label: str) -> str:
    """Normalize sentiment model output labels."""
    label = str(label).strip().lower()

    mapping = {
        "positive": "Positive",
        "neutral": "Neutral",
        "negative": "Negative",
    }

    return mapping.get(label, label.capitalize())


def calculate_risk_level(esg_label: str, sentiment_label: str) -> str:
    """
    Rule-based ESG risk scoring.

    Rule:
    - Non-ESG -> Low
    - ESG + Negative -> High
    - ESG + Neutral -> Medium
    - ESG + Positive -> Low
    """
    esg_label = normalize_esg_label(esg_label)
    sentiment_label = normalize_sentiment_label(sentiment_label)

    if esg_label == "Non-ESG":
        return "Low"

    if sentiment_label == "Negative":
        return "High"
    if sentiment_label == "Neutral":
        return "Medium"
    if sentiment_label == "Positive":
        return "Low"

    return "Medium"


def format_pct(value: float) -> str:
    """Format confidence score as percentage."""
    try:
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return "N/A"


def risk_css_class(risk_level: str) -> str:
    """Map risk level to CSS class."""
    if risk_level == "High":
        return "risk-high"
    if risk_level == "Medium":
        return "risk-medium"
    return "risk-low"


def generate_risk_explanation(
    esg_label: str,
    sentiment_label: str,
    risk_level: str,
) -> str:
    """Generate business-oriented interpretation for short text."""
    if esg_label == "Non-ESG":
        return (
            "This text is classified as Non-ESG. Based on the screening logic, "
            "it is not treated as an ESG-related risk signal. The final risk level "
            "is Low for this input."
        )

    return (
        f"This text is classified as {esg_label} and contains {sentiment_label} "
        f"financial sentiment. Based on the project’s rule-based screening logic, "
        f"it is flagged as {risk_level} ESG risk. The result should be used as an "
        f"early-stage screening signal and reviewed together with supporting evidence."
    )


def get_recommended_action(risk_level: str) -> str:
    """Return practical action statement without investment advice."""
    if risk_level == "High":
        return (
            "Further analyst review is recommended before using this information "
            "in ESG investment research."
        )

    if risk_level == "Medium":
        return (
            "Monitor the issue and review supporting company disclosures or external evidence."
        )

    return "No immediate ESG risk escalation is required based on this text."


def get_document_recommended_action(overall_risk_level: str) -> str:
    """Generate document-level recommended action without investment advice."""
    if overall_risk_level == "High":
        return (
            "Further analyst review is recommended before using this information in ESG "
            "investment research. Analysts should examine the original document sections "
            "related to the dominant ESG topic and negative sentiment."
        )

    if overall_risk_level == "Medium":
        return (
            "Monitor the issue and review supporting company disclosures or external evidence."
        )

    return "No immediate ESG risk escalation is required based on this document."


def safe_pipeline_predict(pipe, text: str, max_length: int = 512) -> Dict[str, Any]:
    """
    Run a Hugging Face text-classification pipeline safely.
    Token-level truncation prevents transformer input length errors.
    """
    return pipe(
        text,
        truncation=True,
        max_length=max_length,
    )[0]


# =========================================================
# Document Extraction
# =========================================================

def extract_text_from_pdf(uploaded_file) -> str:
    """Extract text from uploaded PDF file."""
    if PdfReader is None:
        raise ImportError("pypdf is not installed. Please add pypdf to requirements.txt.")

    reader = PdfReader(uploaded_file)
    pages = []

    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)

    return "\n".join(pages)


def extract_text_from_docx(uploaded_file) -> str:
    """Extract text from uploaded Word document."""
    if Document is None:
        raise ImportError("python-docx is not installed. Please add python-docx to requirements.txt.")

    document = Document(uploaded_file)
    paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


def extract_text_from_txt(uploaded_file) -> str:
    """Extract text from uploaded TXT file."""
    raw = uploaded_file.read()

    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def extract_text_from_uploaded_file(uploaded_file) -> str:
    """Detect file type and extract text."""
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)

    if file_name.endswith(".docx"):
        return extract_text_from_docx(uploaded_file)

    if file_name.endswith(".txt"):
        return extract_text_from_txt(uploaded_file)

    raise ValueError("Unsupported file type. Please upload PDF, DOCX, or TXT.")


# =========================================================
# Model Loading
# =========================================================

@st.cache_resource(show_spinner=False)
def load_esg_pipeline():
    """
    Load local fine-tuned ESG model if available.
    Fallback to FinBERT-ESG baseline if local model is unavailable.
    """
    device, device_name = get_device()

    local_model_loaded = False
    model_name = "yiyanghkust/finbert-esg"

    if LOCAL_ESG_MODEL_DIR.exists():
        try:
            esg_pipe = pipeline(
                task="text-classification",
                model=str(LOCAL_ESG_MODEL_DIR),
                tokenizer=str(LOCAL_ESG_MODEL_DIR),
                device=device,
            )
            local_model_loaded = True
            model_name = "Local fine-tuned DistilBERT ESG classifier"
            return esg_pipe, model_name, local_model_loaded, device_name
        except Exception:
            pass

    esg_pipe = pipeline(
        task="text-classification",
        model="yiyanghkust/finbert-esg",
        tokenizer="yiyanghkust/finbert-esg",
        device=device,
    )
    return esg_pipe, model_name, local_model_loaded, device_name


@st.cache_resource(show_spinner=False)
def load_sentiment_pipeline():
    """Load financial sentiment analysis pipeline."""
    device, device_name = get_device()

    sentiment_pipe = pipeline(
        task="text-classification",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert",
        device=device,
    )

    return sentiment_pipe, "ProsusAI/finbert", device_name


# =========================================================
# Analysis Logic
# =========================================================

def analyze_text(input_text: str) -> Dict[str, Any]:
    """Analyze short input text using ESG and sentiment pipelines."""
    cleaned_text = clean_input_text(input_text)

    if not cleaned_text:
        raise ValueError("Input text is empty. Please paste an ESG-related paragraph.")

    esg_pipe, esg_model_name, local_model_loaded, device_name = load_esg_pipeline()
    sentiment_pipe, sentiment_model_name, _ = load_sentiment_pipeline()

    esg_result = safe_pipeline_predict(esg_pipe, cleaned_text, max_length=512)
    sentiment_result = safe_pipeline_predict(sentiment_pipe, cleaned_text, max_length=512)

    esg_category = normalize_esg_label(esg_result.get("label", ""))
    esg_confidence = float(esg_result.get("score", np.nan))

    sentiment = normalize_sentiment_label(sentiment_result.get("label", ""))
    sentiment_confidence = float(sentiment_result.get("score", np.nan))

    risk_level = calculate_risk_level(esg_category, sentiment)
    risk_explanation = generate_risk_explanation(esg_category, sentiment, risk_level)
    recommended_action = get_recommended_action(risk_level)

    return {
        "input_text": cleaned_text,
        "esg_category": esg_category,
        "esg_confidence": esg_confidence,
        "sentiment": sentiment,
        "sentiment_confidence": sentiment_confidence,
        "risk_level": risk_level,
        "risk_explanation": risk_explanation,
        "recommended_action": recommended_action,
        "esg_model_name": esg_model_name,
        "sentiment_model_name": sentiment_model_name,
        "local_model_loaded": local_model_loaded,
        "device": device_name,
    }


def split_text_into_chunks(
    text: str,
    max_words: int = 180,
    overlap_words: int = 30,
    max_chunks: int = 80,
) -> List[str]:
    """
    Split long text into model-safe chunks.

    Uses a word-based sliding window. Each chunk is small enough to fit
    within transformer token limits, with token-level truncation as backup.
    """
    cleaned = clean_input_text(text)
    words = cleaned.split()

    if not words:
        return []

    if len(words) <= max_words:
        return [" ".join(words)]

    chunks = []
    step = max(max_words - overlap_words, 1)

    start = 0
    while start < len(words) and len(chunks) < max_chunks:
        end = min(start + max_words, len(words))
        chunk_words = words[start:end]
        chunk = " ".join(chunk_words)

        if len(chunk_words) >= 20:
            chunks.append(chunk)

        if end >= len(words):
            break

        start += step

    return chunks


def analyze_single_chunk(chunk_text: str, chunk_id: int) -> Dict[str, Any]:
    """Analyze one document chunk."""
    esg_pipe, _, _, _ = load_esg_pipeline()
    sentiment_pipe, _, _ = load_sentiment_pipeline()

    esg_result = safe_pipeline_predict(esg_pipe, chunk_text, max_length=512)
    sentiment_result = safe_pipeline_predict(sentiment_pipe, chunk_text, max_length=512)

    esg_category = normalize_esg_label(esg_result.get("label", ""))
    esg_confidence = float(esg_result.get("score", np.nan))

    sentiment = normalize_sentiment_label(sentiment_result.get("label", ""))
    sentiment_confidence = float(sentiment_result.get("score", np.nan))

    risk_level = calculate_risk_level(esg_category, sentiment)

    preview = chunk_text[:220] + "..." if len(chunk_text) > 220 else chunk_text

    return {
        "chunk_id": chunk_id,
        "chunk_text": chunk_text,
        "chunk_preview": preview,
        "word_count": len(chunk_text.split()),
        "esg_category": esg_category,
        "esg_confidence": esg_confidence,
        "sentiment": sentiment,
        "sentiment_confidence": sentiment_confidence,
        "risk_level": risk_level,
    }


def choose_dominant_sentiment(sentiments: pd.Series) -> str:
    """
    Choose dominant sentiment.
    In a tie, prioritize Negative, then Neutral, then Positive.
    """
    counts = sentiments.value_counts().to_dict()

    if not counts:
        return "Neutral"

    max_count = max(counts.values())
    tied = [label for label, count in counts.items() if count == max_count]

    for preferred in ["Negative", "Neutral", "Positive"]:
        if preferred in tied:
            return preferred

    return tied[0]


def build_document_summary(results_df: pd.DataFrame, total_words: int) -> Dict[str, Any]:
    """Build overall document-level ESG evaluation from chunk-level outputs."""
    if results_df.empty:
        return {
            "total_words": total_words,
            "chunks_analyzed": 0,
            "overall_risk_level": "Low",
            "top_esg_category": "Non-ESG",
            "dominant_sentiment": "Neutral",
            "average_esg_confidence": 0.0,
            "average_sentiment_confidence": 0.0,
            "high_risk_chunks": 0,
            "medium_risk_chunks": 0,
            "low_risk_chunks": 0,
            "overall_interpretation": (
                "No analyzable ESG content was detected in the submitted document."
            ),
            "recommended_action": "No immediate ESG risk escalation is required based on this document.",
        }

    risk_counts = results_df["risk_level"].value_counts().to_dict()

    high_risk_chunks = int(risk_counts.get("High", 0))
    medium_risk_chunks = int(risk_counts.get("Medium", 0))
    low_risk_chunks = int(risk_counts.get("Low", 0))

    if high_risk_chunks > 0:
        overall_risk_level = "High"
    elif medium_risk_chunks > 0:
        overall_risk_level = "Medium"
    else:
        overall_risk_level = "Low"

    esg_only = results_df[results_df["esg_category"] != "Non-ESG"]

    if len(esg_only) > 0:
        top_esg_category = esg_only["esg_category"].value_counts().idxmax()
    else:
        top_esg_category = "Non-ESG"

    dominant_sentiment = choose_dominant_sentiment(results_df["sentiment"])

    average_esg_confidence = float(results_df["esg_confidence"].mean())
    average_sentiment_confidence = float(results_df["sentiment_confidence"].mean())

    if overall_risk_level == "High":
        overall_interpretation = (
            f"This document is primarily related to {top_esg_category} ESG issues, "
            f"with the dominant financial sentiment being {dominant_sentiment}. "
            f"Based on the project’s rule-based screening logic, the overall ESG risk "
            f"level is High because the document contains negative ESG-related signals. "
            f"The result should be used as an early-stage screening signal and reviewed "
            f"together with supporting evidence and additional analyst judgment."
        )
    elif overall_risk_level == "Medium":
        overall_interpretation = (
            f"This document contains ESG-related content, primarily associated with "
            f"{top_esg_category}. The dominant sentiment is {dominant_sentiment}. "
            f"The overall ESG risk level is Medium, suggesting that analysts should "
            f"monitor the issue and review supporting company disclosures or external evidence."
        )
    else:
        overall_interpretation = (
            f"This document is primarily classified as {top_esg_category}, with "
            f"{dominant_sentiment} financial sentiment. Based on the analyzed content, "
            f"the document does not indicate immediate ESG risk escalation."
        )

    recommended_action = get_document_recommended_action(overall_risk_level)

    return {
        "total_words": total_words,
        "chunks_analyzed": int(len(results_df)),
        "overall_risk_level": overall_risk_level,
        "top_esg_category": top_esg_category,
        "dominant_sentiment": dominant_sentiment,
        "average_esg_confidence": average_esg_confidence,
        "average_sentiment_confidence": average_sentiment_confidence,
        "high_risk_chunks": high_risk_chunks,
        "medium_risk_chunks": medium_risk_chunks,
        "low_risk_chunks": low_risk_chunks,
        "overall_interpretation": overall_interpretation,
        "recommended_action": recommended_action,
    }


def analyze_long_document(
    text: str,
    max_words: int = 180,
    overlap_words: int = 30,
    max_chunks: int = 80,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Analyze long document internally by chunks.
    Only overall summary should be shown to the user.
    """
    cleaned_text = clean_input_text(text)
    total_words = len(cleaned_text.split())

    chunks = split_text_into_chunks(
        cleaned_text,
        max_words=max_words,
        overlap_words=overlap_words,
        max_chunks=max_chunks,
    )

    rows = []

    progress = st.progress(0)
    status = st.empty()

    for idx, chunk in enumerate(chunks, start=1):
        status.write(f"Analyzing internal segment {idx} of {len(chunks)}...")
        rows.append(analyze_single_chunk(chunk, chunk_id=idx))
        progress.progress(idx / len(chunks))

    status.empty()
    progress.empty()

    results_df = pd.DataFrame(rows)
    summary = build_document_summary(results_df, total_words=total_words)

    return results_df, summary


# =========================================================
# Rendering Functions
# =========================================================

def render_top_brand() -> None:
    """Render brand row."""
    st.markdown(
        """
        <div class="top-brand">
            <div class="brand-lockup">
                <div class="brand-mark">◒</div>
                <div>
                    <div class="brand-text-main">BlackRock</div>
                    <div class="brand-text-sub">ESG Risk Analyzer</div>
                </div>
            </div>
            <div class="prototype-pill">Educational Prototype</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    """Render hero section."""
    st.markdown('<div class="hero-shell">', unsafe_allow_html=True)

    left, right = st.columns([1.75, 1], gap="large")

    with left:
        st.markdown(
            """
            <div class="hero-content">
                <div class="eyebrow">ESG investment research assistant</div>
                <div class="hero-title">BlackRock ESG Risk Analyzer</div>
                <div class="hero-subtitle">
                    AI-powered ESG screening for investment research
                </div>
                <div class="hero-description">
                    Use deep learning pipelines to classify corporate text into ESG topics,
                    detect financial sentiment, and generate an explainable ESG risk level
                    for early-stage analyst review.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            """
            <div class="workflow-card">
                <div class="workflow-title">ESG Screening Workflow</div>
                <div class="workflow-step">
                    <div class="step-index">1</div>
                    <div>Input corporate text or upload document</div>
                </div>
                <div class="workflow-step">
                    <div class="step-index">2</div>
                    <div>Classify ESG topic</div>
                </div>
                <div class="workflow-step">
                    <div class="step-index">3</div>
                    <div>Detect financial sentiment</div>
                </div>
                <div class="workflow-step">
                    <div class="step-index">4</div>
                    <div>Generate overall risk level</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


def render_context_cards() -> None:
    """Render purpose, models, risk rule, and disclaimer cards."""
    _, device_name = get_device()

    c1, c2, c3, c4 = st.columns(4, gap="medium")

    with c1:
        st.markdown(
            """
            <div class="context-card">
                <div class="context-icon">◎</div>
                <div>
                    <div class="context-title">Purpose</div>
                    <div class="context-text">
                        ESG document screening for investment research.
                        Identify ESG topics, sentiment, and potential risk signals.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="context-card">
                <div class="context-icon">▱</div>
                <div>
                    <div class="context-title">Models</div>
                    <div class="context-text">
                        ESG Classifier: DistilBERT fine-tuned or FinBERT-ESG fallback<br>
                        Sentiment: ProsusAI/finbert<br>
                        Device: {device_name}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            """
            <div class="context-card">
                <div class="context-icon">◇</div>
                <div>
                    <div class="context-title">Risk Rule</div>
                    <div class="context-text">
                        Non-ESG → Low<br>
                        ESG + Negative → High<br>
                        ESG + Neutral → Medium<br>
                        ESG + Positive → Low
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        st.markdown(
            """
            <div class="context-card">
                <div class="context-icon">i</div>
                <div>
                    <div class="context-title">Disclaimer</div>
                    <div class="context-text">
                        Educational prototype only. This application supports ESG
                        risk screening and does not provide investment advice.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_advanced_settings() -> Tuple[int, int, int]:
    """Render advanced analysis settings and return values."""
    with st.expander("Advanced analysis settings", expanded=False):
        max_words = st.slider(
            "Words per internal segment",
            min_value=120,
            max_value=220,
            value=180,
            step=10,
            help="Smaller segments reduce model input errors. Larger segments preserve more context.",
        )

        overlap_words = st.slider(
            "Overlap words between internal segments",
            min_value=0,
            max_value=50,
            value=30,
            step=5,
            help="Overlap helps preserve context between adjacent segments.",
        )

        max_chunks = st.slider(
            "Maximum internal segments to analyze",
            min_value=10,
            max_value=120,
            value=80,
            step=10,
            help="Limits runtime for very long documents.",
        )

    return max_words, overlap_words, max_chunks


def render_single_text_results(result: Dict[str, Any]) -> None:
    """Render analysis result for short text."""
    st.markdown('<div class="result-shell">', unsafe_allow_html=True)
    st.markdown('<div class="result-title">Overall Analysis Results</div>', unsafe_allow_html=True)

    m1, m2, m3, m4, m5 = st.columns(5, gap="small")

    with m1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Overall ESG Risk</div>
                <div class="metric-value">{result["risk_level"]}</div>
                <div class="metric-caption">Overall assessment</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Top ESG Category</div>
                <div class="metric-value">{result["esg_category"]}</div>
                <div class="metric-caption">Dominant ESG topic</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Financial Sentiment</div>
                <div class="metric-value">{result["sentiment"]}</div>
                <div class="metric-caption">Detected sentiment</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">ESG Confidence</div>
                <div class="metric-value">{format_pct(result["esg_confidence"])}</div>
                <div class="metric-caption">Classifier confidence</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m5:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Sentiment Confidence</div>
                <div class="metric-value">{format_pct(result["sentiment_confidence"])}</div>
                <div class="metric-caption">Sentiment confidence</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    risk_class = risk_css_class(result["risk_level"])
    st.markdown(
        f"""
        <div class="risk-card {risk_class}">
            <div class="risk-label">Final ESG Risk Level</div>
            <div class="risk-value">{result["risk_level"]}</div>
            <div class="interpretation-text">
                Generated from ESG category and financial sentiment using the project’s
                rule-based screening logic.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="interpretation-card">
            <div class="interpretation-title">Business Interpretation</div>
            <div class="interpretation-text">{result["risk_explanation"]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="interpretation-card">
            <div class="interpretation-title">Recommended Action</div>
            <div class="interpretation-text">{result["recommended_action"]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    download_df = pd.DataFrame(
        [
            {
                "input_text": result["input_text"],
                "esg_category": result["esg_category"],
                "esg_confidence": result["esg_confidence"],
                "sentiment": result["sentiment"],
                "sentiment_confidence": result["sentiment_confidence"],
                "risk_level": result["risk_level"],
                "risk_explanation": result["risk_explanation"],
                "recommended_action": result["recommended_action"],
            }
        ]
    )

    csv_buffer = StringIO()
    download_df.to_csv(csv_buffer, index=False)

    st.download_button(
        label="Download Overall Summary as CSV",
        data=csv_buffer.getvalue(),
        file_name="overall_esg_summary.csv",
        mime="text/csv",
    )

    st.caption(
        f"ESG model: {result['esg_model_name']} · "
        f"Sentiment model: {result['sentiment_model_name']} · "
        f"Device: {result['device']}"
    )

    st.markdown("</div>", unsafe_allow_html=True)


def render_overall_document_results(summary: Dict[str, Any]) -> None:
    """Render only overall document-level ESG evaluation."""
    st.markdown('<div class="result-shell">', unsafe_allow_html=True)
    st.markdown('<div class="result-title">Overall Analysis Results</div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5, c6 = st.columns(6, gap="small")

    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Overall ESG Risk</div>
                <div class="metric-value">{summary.get("overall_risk_level", "N/A")}</div>
                <div class="metric-caption">Overall assessment</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Top ESG Category</div>
                <div class="metric-value">{summary.get("top_esg_category", "N/A")}</div>
                <div class="metric-caption">Dominant ESG topic</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Financial Sentiment</div>
                <div class="metric-value">{summary.get("dominant_sentiment", "N/A")}</div>
                <div class="metric-caption">Dominant sentiment</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">ESG Confidence</div>
                <div class="metric-value">{format_pct(summary.get("average_esg_confidence", 0))}</div>
                <div class="metric-caption">Average confidence</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c5:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Sentiment Confidence</div>
                <div class="metric-value">{format_pct(summary.get("average_sentiment_confidence", 0))}</div>
                <div class="metric-caption">Average confidence</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c6:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Estimated Words</div>
                <div class="metric-value">{summary.get("total_words", 0):,}</div>
                <div class="metric-caption">Words analyzed</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    risk_level = summary.get("overall_risk_level", "Low")
    risk_class = risk_css_class(risk_level)

    st.markdown(
        f"""
        <div class="risk-card {risk_class}">
            <div class="risk-label">Final ESG Risk Level</div>
            <div class="risk-value">{risk_level}</div>
            <div class="interpretation-text">
                The document is analyzed internally in smaller model-safe segments.
                Only the overall evaluation is shown to support a concise research workflow.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="interpretation-card">
            <div class="interpretation-title">Business Interpretation</div>
            <div class="interpretation-text">
                {summary.get("overall_interpretation", "")}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="interpretation-card">
            <div class="interpretation-title">Recommended Action</div>
            <div class="interpretation-text">
                {summary.get("recommended_action", "")}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    summary_df = pd.DataFrame([summary])

    st.download_button(
        label="Download Overall Summary as CSV",
        data=summary_df.to_csv(index=False),
        file_name="overall_esg_summary.csv",
        mime="text/csv",
    )

    st.caption(
        "Long documents are split and analyzed internally to support 10,000+ word inputs "
        "and avoid transformer token limits. Chunk-level details are not displayed."
    )

    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# Main App Layout
# =========================================================

render_top_brand()
render_hero()
render_context_cards()


# =========================================================
# Refined Input Studio
# =========================================================

st.markdown(
    """
    <div class="input-studio">
        <div class="input-studio-title">Start an ESG Screening</div>
        <div class="input-studio-subtitle">
            Choose a source type, then provide ESG-related text or upload a document.
            The app will generate an overall ESG risk evaluation for investment research screening.
        </div>
        <div class="mode-caption">Choose analysis source</div>
    """,
    unsafe_allow_html=True,
)

input_mode = st.radio(
    "Choose analysis source",
    ["Paste Text", "Upload PDF / Word"],
    horizontal=True,
    label_visibility="collapsed",
)

st.markdown('<div class="module-card">', unsafe_allow_html=True)

if input_mode == "Paste Text":
    st.markdown(
        """
        <div class="module-title-row">
            <div class="module-icon">✎</div>
            <div>
                <div class="module-title">Paste ESG-related text</div>
            </div>
        </div>
        <div class="module-subtitle">
            Paste one ESG-relevant paragraph, article excerpt, or full report text.
            Long content will be analyzed internally in smaller model-safe segments,
            while only the overall evaluation is shown.
        </div>
        """,
        unsafe_allow_html=True,
    )

    input_text = st.text_area(
        label="Corporate ESG or financial text",
        placeholder=(
            "Paste your ESG-related corporate disclosure, sustainability report, "
            "annual report, or financial news article here..."
        ),
        height=190,
        max_chars=50000,
    )

    st.markdown(
        """
        <div class="input-help-note">
            For best results, use text that contains company actions, ESG issues,
            regulatory events, disclosure language, or sustainability-related claims.
        </div>
        """,
        unsafe_allow_html=True,
    )

    max_words, overlap_words, max_chunks = render_advanced_settings()

    analyze_text_clicked = st.button("Analyze Text", type="primary")

    if analyze_text_clicked:
        cleaned_input = clean_input_text(input_text)

        if not cleaned_input:
            st.warning("Please paste ESG-related text before running the analysis.")
        else:
            word_count = len(cleaned_input.split())

            if word_count > 250:
                if word_count > 10000:
                    st.info(
                        f"Large text detected: approximately {word_count:,} words. "
                        f"The app will analyze up to {max_chunks} internal segments "
                        f"and show only the overall evaluation."
                    )

                with st.spinner("Analyzing long text and generating overall ESG evaluation..."):
                    _, summary = analyze_long_document(
                        cleaned_input,
                        max_words=max_words,
                        overlap_words=overlap_words,
                        max_chunks=max_chunks,
                    )

                render_overall_document_results(summary)

            else:
                with st.spinner("Analyzing ESG topic, financial sentiment, and risk level..."):
                    result = analyze_text(cleaned_input)

                render_single_text_results(result)


elif input_mode == "Upload PDF / Word":
    st.markdown(
        """
        <div class="module-title-row">
            <div class="module-icon">↥</div>
            <div>
                <div class="module-title">Upload an ESG document</div>
            </div>
        </div>
        <div class="module-subtitle">
            Upload a PDF, Word document, or TXT file. The app will extract the text,
            analyze the document internally in smaller model-safe segments, and return
            only the overall ESG evaluation.
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload a PDF, Word document, or TXT file",
        type=["pdf", "docx", "txt"],
        help="Supported formats: PDF, DOCX, TXT",
    )

    max_words, overlap_words, max_chunks = render_advanced_settings()

    analyze_doc_clicked = st.button("Analyze Document", type="primary")

    if analyze_doc_clicked:
        if uploaded_file is None:
            st.warning("Please upload a PDF, Word document, or TXT file before analysis.")
        else:
            try:
                with st.spinner("Extracting text from uploaded document..."):
                    extracted_text = extract_text_from_uploaded_file(uploaded_file)
                    extracted_text = clean_input_text(extracted_text)

                if not extracted_text:
                    st.error("No readable text could be extracted from the uploaded document.")
                else:
                    total_words = len(extracted_text.split())

                    if total_words > 10000:
                        st.info(
                            f"Large document detected: approximately {total_words:,} words. "
                            f"The app will analyze up to {max_chunks} internal segments "
                            f"and show only the overall evaluation."
                        )

                    with st.spinner("Analyzing document and generating overall ESG evaluation..."):
                        _, summary = analyze_long_document(
                            extracted_text,
                            max_words=max_words,
                            overlap_words=overlap_words,
                            max_chunks=max_chunks,
                        )

                    render_overall_document_results(summary)

            except Exception as exc:
                st.error("The uploaded document could not be analyzed.")
                st.write(str(exc))

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# Footer
# =========================================================

st.markdown(
    """
    <div class="footer-note">
        Educational prototype for ISOM5240. This application supports ESG risk screening only
        and does not provide investment advice.
    </div>
    """,
    unsafe_allow_html=True,
)
