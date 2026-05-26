# app.py
# Project: ESG Analyst Briefing System for BlackRock
# ISOM5240 Educational Prototype

from __future__ import annotations

import html
import math
import re
from collections import Counter
from io import BytesIO
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import torch
from transformers import pipeline

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    from docx import Document
except ImportError:
    Document = None

try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        PageBreak,
    )
except ImportError:
    colors = None
    SimpleDocTemplate = None


# =========================================================
# Configuration
# =========================================================

PROJECT_NAME = "ESG Analyst Briefing System for BlackRock"
APP_NAME = "ESG Analyst Briefing System"

HF_FINE_TUNED_ESG_MODEL = "johnsonzhangzzz/blackrock_esg_classifier"
HF_SENTIMENT_MODEL = "ProsusAI/finbert"

DISCLAIMER_TEXT = (
    "This report is for informational purposes only, the information and opinions "
    "contained herein do not constitute investment advice to any person."
)

DEFAULT_MAX_WORDS = 180
DEFAULT_OVERLAP_WORDS = 30
DEFAULT_MAX_CHUNKS = 80

st.set_page_config(
    page_title=APP_NAME,
    page_icon="ESG",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# CSS
# =========================================================

st.markdown(
    """
    <style>
    :root {
        --bg: #f6f4f0;
        --bg-soft: #fbfaf7;
        --ink: #232522;
        --ink-soft: #5f625b;
        --muted: #858a80;
        --line: #e5e1d8;
        --line-strong: #d7d2c6;
        --sage-soft: #e9eee4;
        --sage-dark: #536a5f;
        --green-soft: #e6eee5;
        --amber-soft: #f4eedc;
        --red-soft: #f2e5e3;
        --shadow: 0 18px 45px rgba(47, 48, 45, 0.08);
        --shadow-soft: 0 10px 25px rgba(47, 48, 45, 0.05);
    }

    .stApp {
        background: radial-gradient(circle at top left, #f0ede6 0, #f6f4f0 34%, #fbfaf7 100%);
        color: var(--ink);
    }

    html, body, [class*="css"] {
        font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1240px;
    }

    section[data-testid="stSidebar"] {
        display: none;
    }

    div.stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #536a5f 0%, #6f8178 100%) !important;
        color: #ffffff !important;
        border: 1px solid rgba(83, 106, 95, 0.95) !important;
        border-radius: 16px !important;
        padding: 0.95rem 1.2rem !important;
        font-weight: 780 !important;
        font-size: 1rem !important;
        box-shadow: 0 10px 24px rgba(47, 48, 45, 0.14) !important;
    }

    div.stButton > button p,
    div.stButton > button span,
    div.stButton > button div {
        color: #ffffff !important;
        font-weight: 780 !important;
    }

    div.stButton > button:hover {
        background: linear-gradient(135deg, #465a51 0%, #61736a 100%) !important;
        color: #ffffff !important;
        transform: translateY(-1px);
    }

    div.stDownloadButton > button {
        width: 100%;
        background: linear-gradient(135deg, #2f302d 0%, #536a5f 100%) !important;
        color: #ffffff !important;
        border: 1px solid rgba(47, 48, 45, 0.95) !important;
        border-radius: 16px !important;
        padding: 0.9rem 1.1rem !important;
        font-weight: 760 !important;
        box-shadow: 0 10px 24px rgba(47, 48, 45, 0.14) !important;
    }

    div.stDownloadButton > button p,
    div.stDownloadButton > button span,
    div.stDownloadButton > button div {
        color: #ffffff !important;
        font-weight: 760 !important;
    }

    textarea {
        min-height: 175px !important;
        border-radius: 22px !important;
        border: 1px solid var(--line) !important;
        background: #ffffff !important;
        color: var(--ink) !important;
        padding: 1rem !important;
    }

    [data-testid="stFileUploader"] section {
        border-radius: 22px;
        border: 1.5px dashed #cfc9bd;
        background: #ffffff;
        padding: 1.2rem;
    }

    .top-brand {
        display: flex;
        align-items: center;
        justify-content: flex-start;
        margin-bottom: 0.9rem;
    }

    .brand-lockup {
        display: flex;
        align-items: center;
        gap: 0.85rem;
    }

    .brand-badge {
        width: 3.1rem;
        height: 3.1rem;
        border-radius: 999px;
        background: linear-gradient(135deg, #e9eee4 0%, #f8f6f1 100%);
        border: 1px solid var(--line-strong);
        display: flex;
        align-items: center;
        justify-content: center;
        color: var(--sage-dark);
        font-size: 1.1rem;
        font-weight: 850;
    }

    .brand-text-main {
        color: var(--ink);
        font-size: 1.06rem;
        font-weight: 800;
        line-height: 1.05;
    }

    .brand-text-sub {
        color: var(--ink-soft);
        font-size: 0.92rem;
        line-height: 1.1;
    }

    .hero-shell {
        position: relative;
        background: linear-gradient(135deg, #efebe2 0%, #f8f6f1 52%, #e9eee4 100%);
        border: 1px solid rgba(215, 210, 198, 0.9);
        border-radius: 30px;
        padding: 2.25rem;
        margin-bottom: 1.1rem;
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

    .hero-content {
        position: relative;
        z-index: 1;
    }

    .eyebrow {
        display: inline-flex;
        align-items: center;
        padding: 0.38rem 0.78rem;
        border: 1px solid rgba(47,48,45,0.10);
        border-radius: 999px;
        background: rgba(255,255,255,0.54);
        color: var(--ink-soft);
        font-size: 0.8rem;
        font-weight: 740;
        letter-spacing: 0.035em;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }

    .hero-title {
        font-size: clamp(2.25rem, 4.8vw, 4.4rem);
        line-height: 0.96;
        font-weight: 850;
        letter-spacing: -0.07em;
        color: var(--ink);
        margin-bottom: 0.95rem;
    }

    .hero-subtitle {
        font-size: 1.14rem;
        line-height: 1.55;
        color: var(--sage-dark);
        max-width: 780px;
        font-weight: 620;
        margin-bottom: 0.9rem;
    }

    .hero-description {
        font-size: 0.98rem;
        line-height: 1.6;
        color: var(--ink-soft);
        max-width: 780px;
    }

    .workflow-card {
        position: relative;
        z-index: 1;
        background: rgba(255,255,255,0.80);
        border: 1px solid rgba(215,210,198,0.95);
        border-radius: 24px;
        padding: 1.35rem;
        box-shadow: var(--shadow-soft);
        backdrop-filter: blur(10px);
        min-height: 238px;
    }

    .workflow-title {
        color: var(--ink);
        font-size: 1.02rem;
        font-weight: 800;
        margin-bottom: 0.9rem;
    }

    .workflow-step {
        display: flex;
        gap: 0.72rem;
        align-items: flex-start;
        padding: 0.66rem 0;
        border-bottom: 1px solid rgba(215,210,198,0.55);
        color: var(--ink-soft);
        font-size: 0.92rem;
        line-height: 1.45;
    }

    .workflow-step:last-child {
        border-bottom: none;
    }

    .step-index {
        width: 1.6rem;
        height: 1.6rem;
        border-radius: 999px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: var(--sage-soft);
        color: var(--ink);
        font-size: 0.78rem;
        font-weight: 780;
        flex: 0 0 auto;
    }

    .input-panel {
        background: rgba(255, 255, 255, 0.86);
        border: 1px solid var(--line);
        border-radius: 30px;
        padding: 1.45rem;
        box-shadow: 0 18px 45px rgba(47, 48, 45, 0.07);
        margin-top: 1.1rem;
        margin-bottom: 1.2rem;
    }

    .input-panel-title {
        font-size: 1.45rem;
        font-weight: 830;
        letter-spacing: -0.04em;
        color: var(--ink);
        margin-bottom: 0.35rem;
    }

    .input-panel-subtitle {
        font-size: 0.96rem;
        color: var(--ink-soft);
        line-height: 1.55;
        max-width: 880px;
    }

    .mode-caption {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--muted);
        font-weight: 800;
        margin-top: 1.15rem;
        margin-bottom: 0.5rem;
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
        color: var(--ink-soft);
    }

    div[role="radiogroup"] label:hover {
        background: rgba(255,255,255,0.72);
        border-color: #ddd8cd;
    }

    .module-card {
        background: var(--bg-soft);
        border: 1px solid var(--line);
        border-radius: 26px;
        padding: 1.35rem;
    }

    .module-title-row {
        display: flex;
        align-items: center;
        gap: 0.72rem;
        margin-bottom: 0.35rem;
    }

    .module-badge {
        width: 2.3rem;
        height: 2.3rem;
        border-radius: 999px;
        background: var(--sage-soft);
        display: flex;
        align-items: center;
        justify-content: center;
        color: var(--sage-dark);
        font-size: 0.78rem;
        font-weight: 850;
        flex: 0 0 auto;
        border: 1px solid var(--line);
    }

    .module-title {
        font-size: 1.15rem;
        font-weight: 820;
        color: var(--ink);
    }

    .module-subtitle {
        color: var(--ink-soft);
        font-size: 0.93rem;
        line-height: 1.55;
        margin-bottom: 1rem;
    }

    .input-help-note {
        font-size: 0.85rem;
        color: #7a7d75;
        line-height: 1.45;
        margin-top: 0.75rem;
        margin-bottom: 1rem;
        background: #f1efeb;
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 0.85rem 0.95rem;
    }

    .result-shell {
        background: rgba(255,255,255,0.92);
        border: 1px solid var(--line);
        border-radius: 30px;
        padding: 1.45rem;
        box-shadow: var(--shadow);
        margin-top: 1.25rem;
    }

    .result-title {
        font-size: 1.65rem;
        font-weight: 850;
        letter-spacing: -0.04em;
        color: var(--ink);
        margin-bottom: 1rem;
    }

    .briefing-section {
        background: var(--bg-soft);
        border: 1px solid var(--line);
        border-radius: 22px;
        padding: 1.25rem;
        margin-top: 1rem;
    }

    .briefing-title {
        color: var(--ink);
        font-weight: 830;
        margin-bottom: 0.7rem;
        font-size: 1.03rem;
    }

    .briefing-text {
        color: var(--ink-soft);
        line-height: 1.62;
        font-size: 0.94rem;
    }

    .metric-card {
        background: var(--bg-soft);
        border: 1px solid var(--line);
        border-radius: 22px;
        padding: 1.25rem;
        height: 172px;
        min-height: 172px;
        display: grid;
        grid-template-rows: 42px 1fr 34px;
        align-items: center;
        overflow: hidden;
        margin-bottom: 1rem;
    }

    .metric-label {
        color: #7b8077;
        font-size: 0.78rem;
        font-weight: 850;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        line-height: 1.35;
        align-self: start;
    }

    .metric-value {
        color: #1e2420;
        font-size: clamp(1.28rem, 1.8vw, 1.6rem);
        font-weight: 900;
        letter-spacing: -0.045em;
        line-height: 1.08;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        align-self: center;
    }

    .metric-caption {
        color: #5b6058;
        font-size: 0.88rem;
        font-weight: 600;
        line-height: 1.35;
        align-self: end;
    }

    .risk-card {
        border-radius: 22px;
        padding: 1.2rem 1.25rem;
        margin-top: 0.2rem;
        border: 1px solid var(--line);
    }

    .risk-low { background: var(--green-soft); }
    .risk-medium { background: var(--amber-soft); }
    .risk-high { background: var(--red-soft); }

    .risk-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--muted);
        font-weight: 830;
        margin-bottom: 0.45rem;
    }

    .risk-value {
        color: var(--ink);
        font-size: 1.85rem;
        font-weight: 880;
        letter-spacing: -0.05em;
        margin-bottom: 0.3rem;
    }

    .breakdown-row {
        display: grid;
        grid-template-columns: 140px 1fr 76px;
        gap: 0.75rem;
        align-items: center;
        margin: 0.6rem 0;
    }

    .breakdown-label {
        color: var(--ink);
        font-size: 0.9rem;
        font-weight: 680;
    }

    .breakdown-track {
        height: 0.75rem;
        border-radius: 999px;
        background: #ede9df;
        overflow: hidden;
        border: 1px solid #e2ded5;
    }

    .breakdown-fill {
        height: 100%;
        background: linear-gradient(135deg, #8fa891 0%, #b7c3ad 100%);
        border-radius: 999px;
    }

    .breakdown-value {
        color: var(--ink-soft);
        font-size: 0.86rem;
        text-align: right;
    }

    .driver-pill {
        display: inline-flex;
        align-items: center;
        margin: 0.28rem 0.32rem 0.28rem 0;
        padding: 0.5rem 0.72rem;
        border-radius: 999px;
        background: var(--sage-soft);
        border: 1px solid rgba(174,184,162,0.45);
        color: var(--ink-soft);
        font-size: 0.84rem;
        font-weight: 680;
    }

    .evidence-card {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 1rem;
        margin: 0.7rem 0;
        color: var(--ink-soft);
        line-height: 1.58;
        font-size: 0.9rem;
    }

    .materiality-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
        gap: 0.85rem;
    }

    .materiality-card {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 1rem;
        min-height: 112px;
    }

    .materiality-label {
        color: var(--ink-soft);
        font-size: 0.82rem;
        margin-bottom: 0.45rem;
        line-height: 1.35;
    }

    .materiality-value {
        color: var(--ink);
        font-weight: 850;
        font-size: 1.05rem;
    }

    .checklist-item {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 0.85rem 0.95rem;
        margin: 0.5rem 0;
        color: var(--ink-soft);
        line-height: 1.5;
        font-size: 0.9rem;
    }

    .scope-note {
        background: #f1efeb;
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 1rem;
        margin-top: 1rem;
        color: var(--ink-soft);
        line-height: 1.55;
        font-size: 0.9rem;
    }

    .pdf-note {
        background: #e9eee4;
        border: 1px solid rgba(83, 106, 95, 0.18);
        border-radius: 18px;
        padding: 1rem;
        margin-top: 1rem;
        color: var(--sage-dark);
        line-height: 1.55;
        font-size: 0.9rem;
        font-weight: 600;
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

        .breakdown-row {
            grid-template-columns: 1fr;
        }

        .metric-card {
            height: auto;
            min-height: 150px;
        }

        .metric-value {
            white-space: normal;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# Utility Functions
# =========================================================

def get_device() -> Tuple[int, str]:
    if torch.cuda.is_available():
        return 0, "GPU"
    return -1, "CPU"


def clean_input_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def esc(value: Any) -> str:
    return html.escape(str(value))


def normalize_esg_label(label: str) -> str:
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
        "LABEL_0": "Environmental",
        "LABEL_1": "Social",
        "LABEL_2": "Governance",
        "LABEL_3": "Non-ESG",
    }
    return mapping.get(label, label)


def normalize_sentiment_label(label: str) -> str:
    label = str(label).strip().lower()
    mapping = {
        "positive": "Positive",
        "neutral": "Neutral",
        "negative": "Negative",
    }
    return mapping.get(label, label.capitalize())


def calculate_segment_risk_level(esg_label: str, sentiment_label: str) -> str:
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
    try:
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return "N/A"


def risk_css_class(risk_level: str) -> str:
    if risk_level == "High":
        return "risk-high"
    if risk_level == "Medium":
        return "risk-medium"
    return "risk-low"


def safe_pipeline_predict(pipe, text: str, max_length: int = 512) -> Dict[str, Any]:
    return pipe(text, truncation=True, max_length=max_length)[0]


def sanitize_filename(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"\.[a-zA-Z0-9]+$", "", value)
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value)
    value = value.strip("_")
    return value


def generate_report_filename(
    text: str,
    uploaded_file_name: str | None = None,
    summary: Dict[str, Any] | None = None,
) -> str:
    """
    Generate a stable and meaningful PDF filename from document content.
    This avoids repeated generic names such as esg_analyst_briefing_report (1).pdf.
    """

    stopwords = {
        "the", "and", "for", "with", "that", "this", "from", "into", "their", "there",
        "about", "were", "was", "are", "has", "have", "had", "will", "would", "could",
        "should", "can", "may", "not", "but", "its", "our", "they", "them", "his",
        "her", "you", "your", "company", "companies", "business", "report", "annual",
        "sustainability", "financial", "statement", "statements", "management", "group",
        "limited", "inc", "corp", "corporation", "plc", "ltd", "llc", "esg", "risk",
        "risks", "analysis", "analyst", "briefing", "document", "material", "materials",
        "information", "data", "year", "years", "including", "related", "also",
    }

    esg_priority_terms = [
        "climate", "emissions", "carbon", "scope", "renewable", "netzero", "net",
        "biodiversity", "pollution", "waste", "water", "deforestation",
        "labor", "worker", "safety", "diversity", "privacy", "human", "rights",
        "supply", "chain", "community", "product",
        "governance", "board", "audit", "compliance", "bribery", "corruption",
        "greenwashing", "shareholder", "oversight", "controls", "regulatory",
    ]

    cleaned_text = clean_input_text(text).lower()
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9]{2,}", cleaned_text)

    filtered_tokens = [
        token for token in tokens
        if token not in stopwords and len(token) >= 3 and not token.isdigit()
    ]

    counter = Counter(filtered_tokens)

    selected_keywords = []

    for term in esg_priority_terms:
        if term in counter and term not in selected_keywords:
            selected_keywords.append(term)

    for token, _ in counter.most_common(20):
        if token not in selected_keywords:
            selected_keywords.append(token)
        if len(selected_keywords) >= 5:
            break

    if summary:
        top_category = sanitize_filename(str(summary.get("top_esg_category", "")))
        sentiment = sanitize_filename(str(summary.get("dominant_sentiment", "")))
        overall_risk = sanitize_filename(str(summary.get("overall_risk_level", "")))

        for item in [top_category, sentiment, overall_risk]:
            if item and item not in selected_keywords and item not in {"n_a", "non_esg"}:
                selected_keywords.append(item)

    selected_keywords = selected_keywords[:6]

    base_parts = []

    if uploaded_file_name:
        uploaded_base = sanitize_filename(uploaded_file_name)
        uploaded_base_words = [word for word in uploaded_base.split("_") if word and word not in stopwords]
        uploaded_base_short = "_".join(uploaded_base_words[:4])

        if uploaded_base_short:
            base_parts.append(uploaded_base_short)

    if selected_keywords:
        base_parts.append("_".join(selected_keywords[:5]))

    if not base_parts:
        base_parts.append("esg_document")

    base_name = "_".join(base_parts)
    base_name = sanitize_filename(base_name)

    if len(base_name) > 85:
        base_name = base_name[:85].rstrip("_")

    return f"esg_report_{base_name}.pdf"


# =========================================================
# Document Extraction
# =========================================================

def extract_text_from_pdf(uploaded_file) -> str:
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
    if Document is None:
        raise ImportError("python-docx is not installed. Please add python-docx to requirements.txt.")

    document = Document(uploaded_file)
    paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


def extract_text_from_txt(uploaded_file) -> str:
    raw = uploaded_file.read()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def extract_text_from_uploaded_file(uploaded_file) -> str:
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
    device, _ = get_device()

    return pipeline(
        task="text-classification",
        model=HF_FINE_TUNED_ESG_MODEL,
        tokenizer=HF_FINE_TUNED_ESG_MODEL,
        device=device,
    )


@st.cache_resource(show_spinner=False)
def load_sentiment_pipeline():
    device, _ = get_device()

    return pipeline(
        task="text-classification",
        model=HF_SENTIMENT_MODEL,
        tokenizer=HF_SENTIMENT_MODEL,
        device=device,
    )


# =========================================================
# ESG Briefing Logic
# =========================================================

RISK_KEYWORDS = {
    "Environmental": [
        "carbon emissions", "greenhouse gas", "climate risk", "pollution",
        "waste", "water usage", "biodiversity", "deforestation",
        "fossil fuel", "scope 1", "scope 2", "scope 3",
        "renewable energy", "net zero", "climate disclosure",
        "emissions", "decarbonization", "climate transition",
    ],
    "Social": [
        "labor", "worker safety", "workplace safety", "diversity",
        "discrimination", "harassment", "privacy", "human rights",
        "supply chain", "employee injury", "community", "product safety",
        "employee safety", "customer privacy", "workforce",
    ],
    "Governance": [
        "bribery", "corruption", "board oversight", "internal controls",
        "audit", "compliance", "executive compensation", "shareholder",
        "greenwashing", "misleading disclosure", "regulatory scrutiny",
        "board independence", "governance", "fraud", "ethics",
    ],
}


def get_document_recommended_action(overall_risk_level: str) -> str:
    if overall_risk_level == "High":
        return (
            "Escalate this document for ESG analyst review. Prioritize source verification, "
            "management response, remediation evidence, and whether the issue is recurring. "
            "Use the briefing as a screening input rather than an investment recommendation."
        )

    if overall_risk_level == "Medium":
        return (
            "Place the issue on an ESG watchlist. Review supporting disclosures, compare with "
            "peer practices, and monitor whether the topic appears in future reports or external news."
        )

    return (
        "No immediate ESG escalation is required based on this document. Keep the output as a "
        "screening record and review again if new evidence or negative developments emerge."
    )


def extract_risk_drivers(text: str) -> List[Dict[str, str]]:
    text_lower = clean_input_text(text).lower()
    drivers = []

    for category, keywords in RISK_KEYWORDS.items():
        for keyword in keywords:
            if keyword.lower() in text_lower:
                drivers.append({"category": category, "keyword": keyword})

    seen = set()
    unique_drivers = []

    for item in drivers:
        key = (item["category"], item["keyword"])
        if key not in seen:
            seen.add(key)
            unique_drivers.append(item)

    return unique_drivers[:12]


def get_evidence_highlights(results_df: pd.DataFrame, max_items: int = 3) -> List[str]:
    if results_df.empty:
        return []

    priority_df = results_df.copy()
    priority_df["priority_score"] = 0
    priority_df.loc[priority_df["risk_level"] == "High", "priority_score"] += 3
    priority_df.loc[priority_df["sentiment"] == "Negative", "priority_score"] += 2
    priority_df.loc[priority_df["esg_category"] != "Non-ESG", "priority_score"] += 1

    priority_df = priority_df.sort_values(
        by=["priority_score", "esg_confidence", "sentiment_confidence"],
        ascending=[False, False, False],
    )

    highlights = []

    for _, row in priority_df.head(max_items).iterrows():
        snippet = str(row["chunk_text"])[:420]
        if len(str(row["chunk_text"])) > 420:
            snippet += "..."
        highlights.append(snippet)

    return highlights


def build_materiality_assessment(
    top_esg_category: str,
    dominant_sentiment: str,
    overall_risk: str,
) -> Dict[str, str]:
    assessment = {
        "Regulatory Exposure": "Low",
        "Reputation Risk": "Low",
        "Operational Risk": "Low",
        "Disclosure Quality Concern": "Low",
    }

    if overall_risk == "High":
        if top_esg_category == "Environmental":
            assessment["Regulatory Exposure"] = "High"
            assessment["Reputation Risk"] = "Medium"
            assessment["Operational Risk"] = "Medium"
            assessment["Disclosure Quality Concern"] = "High"
        elif top_esg_category == "Social":
            assessment["Regulatory Exposure"] = "Medium"
            assessment["Reputation Risk"] = "High"
            assessment["Operational Risk"] = "Medium"
            assessment["Disclosure Quality Concern"] = "Medium"
        elif top_esg_category == "Governance":
            assessment["Regulatory Exposure"] = "High"
            assessment["Reputation Risk"] = "High"
            assessment["Operational Risk"] = "Medium"
            assessment["Disclosure Quality Concern"] = "High"

    elif overall_risk == "Medium":
        assessment["Regulatory Exposure"] = "Medium" if dominant_sentiment == "Negative" else "Low"
        assessment["Reputation Risk"] = "Medium"
        assessment["Operational Risk"] = "Low"
        assessment["Disclosure Quality Concern"] = "Medium"

    return assessment


def build_analyst_checklist(overall_risk: str, top_esg_category: str) -> List[str]:
    checklist = [
        "Identify the exact document section or paragraph behind the ESG signal and verify it against the original source.",
        "Check whether the issue is recurring across prior sustainability reports, annual reports, controversies, or news coverage.",
        "Compare the company disclosure quality and ESG response with close peer companies.",
        "Verify whether regulators, auditors, NGOs, employees, suppliers, or media have raised similar concerns.",
        "Assess whether management provides a measurable remediation plan, timeline, accountability owner, and progress indicators.",
        "Determine whether the issue affects financial materiality, operational continuity, regulatory exposure, or stakeholder trust.",
    ]

    if top_esg_category == "Environmental":
        checklist.extend(
            [
                "Review Scope 1, Scope 2, and Scope 3 emissions disclosure completeness and consistency.",
                "Check whether climate targets are science-based, time-bound, and supported by capital allocation or transition plans.",
                "Assess exposure to carbon pricing, environmental regulation, pollution incidents, and climate transition risk.",
            ]
        )
    elif top_esg_category == "Social":
        checklist.extend(
            [
                "Review workforce safety metrics, employee turnover, discrimination allegations, and labor relations history.",
                "Assess supply-chain labor standards, human rights controls, customer privacy incidents, and product safety records.",
                "Check whether the company discloses measurable social performance targets and remediation outcomes.",
            ]
        )
    elif top_esg_category == "Governance":
        checklist.extend(
            [
                "Review board independence, audit oversight, internal control weaknesses, and compliance incidents.",
                "Check whether executive compensation is aligned with long-term performance and ESG accountability.",
                "Assess potential greenwashing, misleading disclosure, bribery, corruption, shareholder-rights, or ethics risks.",
            ]
        )

    if overall_risk == "High":
        checklist.insert(
            0,
            "Escalate the case for senior ESG analyst review and preserve the source document for audit trail purposes.",
        )

    return checklist


def build_confidence_interpretation(avg_esg_conf: float, avg_sent_conf: float) -> str:
    average_confidence = np.nanmean([avg_esg_conf, avg_sent_conf])

    if average_confidence >= 0.85:
        signal = "Strong model signal"
        action = (
            "The model output is directionally reliable for screening, but analysts should still verify "
            "the supporting evidence and original document context."
        )
    elif average_confidence >= 0.65:
        signal = "Moderate model signal"
        action = (
            "The output is useful for triage, but manual review is recommended before using it in a "
            "research memo or ESG monitoring workflow."
        )
    else:
        signal = "Low model signal"
        action = (
            "Treat this result as uncertain. Analysts should manually inspect the document and avoid "
            "drawing conclusions from the model output alone."
        )

    return f"{signal}. {action} {DISCLAIMER_TEXT}"


def build_decision_support_note(overall_risk: str) -> str:
    if overall_risk == "High":
        return (
            "Decision support note: prioritize this document for deeper ESG due diligence. "
            "Recommended follow-up includes evidence verification, peer comparison, regulatory scan, "
            "management-response review, and escalation to the ESG research team if the issue is material."
        )

    if overall_risk == "Medium":
        return (
            "Decision support note: use this output as a watchlist signal. Track whether the issue becomes "
            "more frequent, more severe, or externally validated by regulators, auditors, media, or stakeholders."
        )

    return (
        "Decision support note: the document does not indicate immediate ESG risk escalation. "
        "Keep the output as a screening record and reassess if new negative evidence appears."
    )


def distribution_dict(series: pd.Series) -> Dict[str, Dict[str, float]]:
    counts = series.value_counts().to_dict()
    total = max(int(series.count()), 1)

    output = {}

    for label, count in counts.items():
        output[label] = {
            "count": int(count),
            "share": float(count / total),
        }

    return output


def choose_dominant_sentiment(sentiments: pd.Series) -> str:
    counts = sentiments.value_counts().to_dict()

    if not counts:
        return "Neutral"

    max_count = max(counts.values())
    tied = [label for label, count in counts.items() if count == max_count]

    for preferred in ["Negative", "Neutral", "Positive"]:
        if preferred in tied:
            return preferred

    return tied[0]


def choose_overall_document_risk(
    high_risk_chunks: int,
    medium_risk_chunks: int,
    chunks_analyzed: int,
    esg_related_chunks: int,
) -> str:
    if chunks_analyzed <= 0:
        return "Low"

    high_share = high_risk_chunks / chunks_analyzed
    medium_share = medium_risk_chunks / chunks_analyzed
    esg_share = esg_related_chunks / chunks_analyzed

    if high_risk_chunks >= 2 or high_share >= 0.15:
        return "High"
    if high_risk_chunks == 1:
        return "Medium"
    if medium_risk_chunks >= 2 or medium_share >= 0.25:
        return "Medium"
    if esg_share >= 0.35 and medium_risk_chunks >= 1:
        return "Medium"

    return "Low"


def build_executive_summary(summary: Dict[str, Any]) -> str:
    risk = summary.get("overall_risk_level", "N/A")
    category = summary.get("top_esg_category", "N/A")
    sentiment = summary.get("dominant_sentiment", "N/A")
    high_count = summary.get("high_risk_chunks", 0)
    chunks = summary.get("chunks_analyzed", 0)

    if risk == "High":
        return (
            f"The document shows a High ESG risk profile, primarily driven by {category} issues "
            f"and {sentiment} sentiment. The system detected {high_count} high-risk internal "
            f"segment(s) across {chunks} analyzed segment(s), suggesting potentially material "
            f"ESG risk signals. Further analyst review is recommended before using this information "
            f"in ESG investment research."
        )

    if risk == "Medium":
        return (
            f"The document shows a Medium ESG risk profile. The dominant ESG topic is {category}, "
            f"and the dominant sentiment is {sentiment}. The content may warrant monitoring and "
            f"follow-up review, particularly if similar issues appear in external sources, peer disclosures, "
            f"or prior company reports."
        )

    return (
        f"The document shows a Low ESG risk profile based on the analyzed text. The dominant ESG "
        f"topic is {category}, and the dominant sentiment is {sentiment}. No immediate ESG risk "
        f"escalation is indicated by this screening output."
    )


def build_scope_note(total_words: int, chunks_analyzed: int) -> str:
    if total_words <= 0:
        return "No readable text was analyzed."

    estimated_possible_chunks = max(
        1,
        math.ceil(total_words / max(DEFAULT_MAX_WORDS - DEFAULT_OVERLAP_WORDS, 1)),
    )

    if chunks_analyzed >= estimated_possible_chunks:
        return (
            f"The system analyzed the submitted document using {chunks_analyzed} internal segment(s). "
            f"The document was split internally to respect transformer token limits."
        )

    return (
        f"The document contains approximately {total_words:,} words. To keep runtime manageable, "
        f"the system analyzed the first {chunks_analyzed} internal segment(s) out of an estimated "
        f"{estimated_possible_chunks} possible segment(s). For final research, analysts should review "
        f"the original source document in full."
    )


# =========================================================
# Analysis
# =========================================================

def split_text_into_chunks(
    text: str,
    max_words: int = DEFAULT_MAX_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
    max_chunks: int = DEFAULT_MAX_CHUNKS,
) -> List[str]:
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

        if len(chunk_words) >= 20:
            chunks.append(" ".join(chunk_words))

        if end >= len(words):
            break

        start += step

    return chunks


def analyze_single_chunk(chunk_text: str, chunk_id: int) -> Dict[str, Any]:
    esg_pipe = load_esg_pipeline()
    sentiment_pipe = load_sentiment_pipeline()

    esg_result = safe_pipeline_predict(esg_pipe, chunk_text, max_length=512)
    sentiment_result = safe_pipeline_predict(sentiment_pipe, chunk_text, max_length=512)

    esg_category = normalize_esg_label(esg_result.get("label", ""))
    esg_confidence = float(esg_result.get("score", np.nan))

    sentiment = normalize_sentiment_label(sentiment_result.get("label", ""))
    sentiment_confidence = float(sentiment_result.get("score", np.nan))

    risk_level = calculate_segment_risk_level(esg_category, sentiment)

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


def build_document_summary(results_df: pd.DataFrame, total_words: int) -> Dict[str, Any]:
    if results_df.empty:
        summary = {
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
            "recommended_action": get_document_recommended_action("Low"),
            "esg_category_distribution": {},
            "sentiment_distribution": {},
            "risk_driver_keywords": [],
            "evidence_highlights": [],
            "materiality_assessment": build_materiality_assessment("Non-ESG", "Neutral", "Low"),
            "analyst_checklist": build_analyst_checklist("Low", "Non-ESG"),
            "confidence_interpretation": build_confidence_interpretation(0.0, 0.0),
            "decision_support_note": build_decision_support_note("Low"),
            "scope_note": build_scope_note(total_words, 0),
        }
        summary["executive_summary"] = build_executive_summary(summary)
        return summary

    chunks_analyzed = int(len(results_df))
    risk_counts = results_df["risk_level"].value_counts().to_dict()
    high_risk_chunks = int(risk_counts.get("High", 0))
    medium_risk_chunks = int(risk_counts.get("Medium", 0))
    low_risk_chunks = int(risk_counts.get("Low", 0))

    esg_only = results_df[results_df["esg_category"] != "Non-ESG"]
    esg_related_chunks = int(len(esg_only))

    if esg_related_chunks > 0:
        top_esg_category = esg_only["esg_category"].value_counts().idxmax()
    else:
        top_esg_category = "Non-ESG"

    dominant_sentiment = choose_dominant_sentiment(results_df["sentiment"])
    average_esg_confidence = float(results_df["esg_confidence"].mean())
    average_sentiment_confidence = float(results_df["sentiment_confidence"].mean())

    overall_risk_level = choose_overall_document_risk(
        high_risk_chunks=high_risk_chunks,
        medium_risk_chunks=medium_risk_chunks,
        chunks_analyzed=chunks_analyzed,
        esg_related_chunks=esg_related_chunks,
    )

    all_text = " ".join(results_df["chunk_text"].astype(str).tolist())

    summary = {
        "total_words": total_words,
        "chunks_analyzed": chunks_analyzed,
        "overall_risk_level": overall_risk_level,
        "top_esg_category": top_esg_category,
        "dominant_sentiment": dominant_sentiment,
        "average_esg_confidence": average_esg_confidence,
        "average_sentiment_confidence": average_sentiment_confidence,
        "high_risk_chunks": high_risk_chunks,
        "medium_risk_chunks": medium_risk_chunks,
        "low_risk_chunks": low_risk_chunks,
        "recommended_action": get_document_recommended_action(overall_risk_level),
        "esg_category_distribution": distribution_dict(results_df["esg_category"]),
        "sentiment_distribution": distribution_dict(results_df["sentiment"]),
        "risk_driver_keywords": extract_risk_drivers(all_text),
        "evidence_highlights": get_evidence_highlights(results_df, max_items=3),
        "materiality_assessment": build_materiality_assessment(
            top_esg_category,
            dominant_sentiment,
            overall_risk_level,
        ),
        "analyst_checklist": build_analyst_checklist(overall_risk_level, top_esg_category),
        "confidence_interpretation": build_confidence_interpretation(
            average_esg_confidence,
            average_sentiment_confidence,
        ),
        "decision_support_note": build_decision_support_note(overall_risk_level),
        "scope_note": build_scope_note(total_words, chunks_analyzed),
    }

    summary["executive_summary"] = build_executive_summary(summary)

    return summary


def analyze_long_document(
    text: str,
    max_words: int = DEFAULT_MAX_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
    max_chunks: int = DEFAULT_MAX_CHUNKS,
    uploaded_file_name: str | None = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
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
        progress.progress(idx / max(len(chunks), 1))

    status.empty()
    progress.empty()

    results_df = pd.DataFrame(rows)
    summary = build_document_summary(results_df, total_words=total_words)
    summary["report_filename"] = generate_report_filename(
        text=cleaned_text,
        uploaded_file_name=uploaded_file_name,
        summary=summary,
    )

    return results_df, summary


# =========================================================
# PDF Report Generation
# =========================================================

def create_standard_table_style() -> TableStyle:
    return TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9eee4")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#232522")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d7d2c6")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fbfaf7")]),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]
    )


def create_pdf_report(summary: Dict[str, Any]) -> bytes:
    if SimpleDocTemplate is None:
        raise ImportError("reportlab is not installed. Please add reportlab to requirements.txt.")

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
        title="ESG Analyst Briefing Report",
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#232522"),
        spaceAfter=12,
        alignment=TA_LEFT,
    )

    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#5f625b"),
        spaceAfter=14,
    )

    h2_style = ParagraphStyle(
        "SectionHeader",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#536a5f"),
        spaceBefore=12,
        spaceAfter=8,
    )

    body_style = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#2f302d"),
        spaceAfter=7,
    )

    small_style = ParagraphStyle(
        "Small",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#5f625b"),
        spaceAfter=5,
    )

    story = []

    story.append(Paragraph("ESG Analyst Briefing Report", title_style))
    story.append(Paragraph("ESG screening output for research triage.", subtitle_style))

    story.append(Paragraph("1. Executive ESG Summary", h2_style))
    story.append(Paragraph(esc(summary.get("executive_summary", "")), body_style))

    story.append(Paragraph("2. Key Metrics", h2_style))

    metric_data = [
        ["Metric", "Value"],
        ["Overall ESG Risk", str(summary.get("overall_risk_level", "N/A"))],
        ["Top ESG Category", str(summary.get("top_esg_category", "N/A"))],
        ["Dominant Sentiment", str(summary.get("dominant_sentiment", "N/A"))],
        ["Estimated Words", f"{summary.get('total_words', 0):,}"],
        ["Internal Segments Analyzed", str(summary.get("chunks_analyzed", "N/A"))],
        ["ESG Confidence", format_pct(summary.get("average_esg_confidence", 0))],
        ["Sentiment Confidence", format_pct(summary.get("average_sentiment_confidence", 0))],
    ]

    metric_table = Table(metric_data, colWidths=[2.3 * inch, 4.4 * inch])
    metric_table.setStyle(create_standard_table_style())
    story.append(metric_table)

    story.append(Paragraph("3. Risk Interpretation", h2_style))
    story.append(Paragraph(esc(summary.get("recommended_action", "")), body_style))

    story.append(Paragraph("4. ESG Category Breakdown", h2_style))

    esg_distribution = summary.get("esg_category_distribution", {})
    esg_rows = [["Category", "Share", "Count"]]

    for label in ["Environmental", "Social", "Governance", "Non-ESG"]:
        item = esg_distribution.get(label, {"share": 0, "count": 0})
        esg_rows.append([label, f"{float(item.get('share', 0)) * 100:.1f}%", str(item.get("count", 0))])

    esg_table = Table(esg_rows, colWidths=[2.6 * inch, 2 * inch, 2 * inch])
    esg_table.setStyle(create_standard_table_style())
    story.append(esg_table)

    story.append(Paragraph("5. Sentiment Breakdown", h2_style))

    sent_distribution = summary.get("sentiment_distribution", {})
    sent_rows = [["Sentiment", "Share", "Count"]]

    for label in ["Negative", "Neutral", "Positive"]:
        item = sent_distribution.get(label, {"share": 0, "count": 0})
        sent_rows.append([label, f"{float(item.get('share', 0)) * 100:.1f}%", str(item.get("count", 0))])

    sent_table = Table(sent_rows, colWidths=[2.6 * inch, 2 * inch, 2 * inch])
    sent_table.setStyle(create_standard_table_style())
    story.append(sent_table)

    story.append(Paragraph("6. Key Risk Drivers", h2_style))

    drivers = summary.get("risk_driver_keywords", [])

    if drivers:
        for driver in drivers:
            story.append(
                Paragraph(
                    f"- {esc(driver.get('category', 'N/A'))}: {esc(driver.get('keyword', 'N/A'))}",
                    body_style,
                )
            )
    else:
        story.append(Paragraph("No major rule-based ESG risk keywords were detected.", body_style))

    story.append(Paragraph("7. Evidence Highlights", h2_style))

    evidence = summary.get("evidence_highlights", [])

    if evidence:
        for idx, snippet in enumerate(evidence, start=1):
            story.append(Paragraph(f"{idx}. {esc(snippet)}", small_style))
    else:
        story.append(Paragraph("No strong evidence snippets were selected.", body_style))

    story.append(PageBreak())

    story.append(Paragraph("8. Materiality Assessment", h2_style))

    materiality = summary.get("materiality_assessment", {})
    materiality_rows = [["Dimension", "Assessment"]]

    for key in [
        "Regulatory Exposure",
        "Reputation Risk",
        "Operational Risk",
        "Disclosure Quality Concern",
    ]:
        materiality_rows.append([key, str(materiality.get(key, "Low"))])

    materiality_table = Table(materiality_rows, colWidths=[3.2 * inch, 3.4 * inch])
    materiality_table.setStyle(create_standard_table_style())
    story.append(materiality_table)

    story.append(Paragraph("9. Analyst Review Checklist", h2_style))

    checklist = summary.get("analyst_checklist", [])

    if checklist:
        for item in checklist:
            story.append(Paragraph(f"[ ] {esc(item)}", small_style))
    else:
        story.append(Paragraph("No checklist items were generated.", body_style))

    story.append(Paragraph("10. Confidence and Limitations", h2_style))
    story.append(Paragraph(esc(summary.get("confidence_interpretation", "")), body_style))

    story.append(Paragraph("11. Analysis Scope", h2_style))
    story.append(Paragraph(esc(summary.get("scope_note", "")), body_style))

    story.append(Paragraph("12. Decision Support Note", h2_style))
    story.append(Paragraph(esc(summary.get("decision_support_note", "")), body_style))

    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Disclaimer: {esc(DISCLAIMER_TEXT)}", small_style))

    doc.build(story)

    pdf_bytes = buffer.getvalue()
    buffer.close()

    return pdf_bytes


# =========================================================
# Rendering
# =========================================================

def render_top_brand() -> None:
    st.markdown(
        """
        <div class="top-brand">
            <div class="brand-lockup">
                <div class="brand-badge">ESG</div>
                <div>
                    <div class="brand-text-main">BlackRock</div>
                    <div class="brand-text-sub">ESG Analyst Briefing System</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown('<div class="hero-shell">', unsafe_allow_html=True)

    left, right = st.columns([1.75, 1], gap="large")

    with left:
        st.markdown(
            f"""
            <div class="hero-content">
                <div class="eyebrow">ESG investment research assistant</div>
                <div class="hero-title">{esc(APP_NAME)}</div>
                <div class="hero-subtitle">
                    ESG analyst briefing for investment research
                </div>
                <div class="hero-description">
                    Classify corporate text into ESG topics, detect financial sentiment,
                    identify risk drivers, and produce a practical analyst briefing for
                    early-stage review.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            """
            <div class="workflow-card">
                <div class="workflow-title">ESG Briefing Workflow</div>
                <div class="workflow-step"><div class="step-index">1</div><div>Input ESG text or upload a document</div></div>
                <div class="workflow-step"><div class="step-index">2</div><div>Classify ESG category and sentiment</div></div>
                <div class="workflow-step"><div class="step-index">3</div><div>Identify risk drivers and evidence</div></div>
                <div class="workflow-step"><div class="step-index">4</div><div>Generate analyst briefing</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


def render_metric_cards(summary: Dict[str, Any]) -> None:
    metrics = [
        ("Overall ESG Risk", summary.get("overall_risk_level", "N/A"), "Risk profile"),
        ("Top ESG Category", summary.get("top_esg_category", "N/A"), "Dominant topic"),
        ("Dominant Sentiment", summary.get("dominant_sentiment", "N/A"), "Tone of document"),
        ("Estimated Words", f"{summary.get('total_words', 0):,}", "Words submitted"),
        ("ESG Confidence", format_pct(summary.get("average_esg_confidence", 0)), "Avg. ESG signal"),
        ("Sentiment Confidence", format_pct(summary.get("average_sentiment_confidence", 0)), "Avg. sentiment signal"),
    ]

    row1 = st.columns(3, gap="medium")
    row2 = st.columns(3, gap="medium")
    columns = list(row1) + list(row2)

    for col, (label, value, caption) in zip(columns, metrics):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{esc(label)}</div>
                    <div class="metric-value" title="{esc(value)}">{esc(value)}</div>
                    <div class="metric-caption">{esc(caption)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_distribution(title: str, distribution: Dict[str, Dict[str, float]], labels: List[str]) -> None:
    st.markdown(
        f"""
        <div class="briefing-section">
            <div class="briefing-title">{esc(title)}</div>
        """,
        unsafe_allow_html=True,
    )

    for label in labels:
        item = distribution.get(label, {"count": 0, "share": 0.0})
        share = float(item.get("share", 0.0))
        count = int(item.get("count", 0))
        width = max(share * 100, 1 if count > 0 else 0)

        st.markdown(
            f"""
            <div class="breakdown-row">
                <div class="breakdown-label">{esc(label)}</div>
                <div class="breakdown-track">
                    <div class="breakdown-fill" style="width: {width:.2f}%;"></div>
                </div>
                <div class="breakdown-value">{share * 100:.1f}% ({count})</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


def render_esg_analyst_briefing(summary: Dict[str, Any]) -> None:
    st.markdown('<div class="result-shell">', unsafe_allow_html=True)
    st.markdown('<div class="result-title">ESG Analyst Briefing</div>', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="briefing-section">
            <div class="briefing-title">Executive ESG Summary</div>
            <div class="briefing-text">{esc(summary.get("executive_summary", ""))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_metric_cards(summary)

    risk_level = summary.get("overall_risk_level", "Low")
    risk_class = risk_css_class(risk_level)

    st.markdown(
        f"""
        <div class="risk-card {risk_class}">
            <div class="risk-label">Overall ESG Risk Level</div>
            <div class="risk-value">{esc(risk_level)}</div>
            <div class="briefing-text">
                This score is generated from ESG classification, financial sentiment, and document-level screening logic.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    b1, b2 = st.columns(2, gap="medium")

    with b1:
        render_distribution(
            "ESG Risk Breakdown",
            summary.get("esg_category_distribution", {}),
            ["Environmental", "Social", "Governance", "Non-ESG"],
        )

    with b2:
        render_distribution(
            "Sentiment Breakdown",
            summary.get("sentiment_distribution", {}),
            ["Negative", "Neutral", "Positive"],
        )

    drivers = summary.get("risk_driver_keywords", [])
    driver_html = ""

    if drivers:
        for item in drivers:
            driver_html += (
                f'<span class="driver-pill">{esc(item.get("category", ""))}: '
                f'{esc(item.get("keyword", ""))}</span>'
            )
    else:
        driver_html = '<div class="briefing-text">No major rule-based ESG risk keywords were detected.</div>'

    st.markdown(
        f"""
        <div class="briefing-section">
            <div class="briefing-title">Key Risk Drivers</div>
            {driver_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    evidence = summary.get("evidence_highlights", [])
    evidence_html = ""

    if evidence:
        for snippet in evidence:
            evidence_html += f'<div class="evidence-card">"{esc(snippet)}"</div>'
    else:
        evidence_html = '<div class="briefing-text">No strong evidence snippets were selected.</div>'

    st.markdown(
        f"""
        <div class="briefing-section">
            <div class="briefing-title">Evidence Highlights</div>
            {evidence_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    materiality = summary.get("materiality_assessment", {})
    materiality_html = ""

    for key in [
        "Regulatory Exposure",
        "Reputation Risk",
        "Operational Risk",
        "Disclosure Quality Concern",
    ]:
        materiality_html += (
            f'<div class="materiality-card">'
            f'<div class="materiality-label">{esc(key)}</div>'
            f'<div class="materiality-value">{esc(materiality.get(key, "Low"))}</div>'
            f'</div>'
        )

    st.markdown(
        f"""
        <div class="briefing-section">
            <div class="briefing-title">Materiality Assessment</div>
            <div class="materiality-grid">{materiality_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    checklist = summary.get("analyst_checklist", [])
    checklist_html = ""

    for item in checklist:
        checklist_html += f'<div class="checklist-item">[ ] {esc(item)}</div>'

    st.markdown(
        f"""
        <div class="briefing-section">
            <div class="briefing-title">Analyst Review Checklist</div>
            {checklist_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="briefing-section">
            <div class="briefing-title">Confidence and Limitations</div>
            <div class="briefing-text">{esc(summary.get("confidence_interpretation", ""))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="scope-note">
            <strong>Analysis Scope:</strong> {esc(summary.get("scope_note", ""))}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="briefing-section">
            <div class="briefing-title">Decision Support Note</div>
            <div class="briefing-text">{esc(summary.get("decision_support_note", ""))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        pdf_bytes = create_pdf_report(summary)
        report_filename = summary.get("report_filename", "esg_report_document.pdf")

        st.markdown(
            """
            <div class="pdf-note">
                Download a formatted PDF report for documentation, presentation, or analyst review.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.download_button(
            label="Download ESG Analyst Briefing PDF",
            data=pdf_bytes,
            file_name=report_filename,
            mime="application/pdf",
        )

    except Exception as exc:
        st.error("PDF report could not be generated.")
        st.write(str(exc))
        st.info("Please make sure reportlab is included in requirements.txt.")

    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# Main App
# =========================================================

render_top_brand()
render_hero()

st.markdown(
    """
    <div class="input-panel">
        <div class="input-panel-title">Start an ESG Screening</div>
        <div class="input-panel-subtitle">
            Choose a source type, then provide ESG-related text or upload a document.
            The app will generate an ESG analyst briefing for investment research screening.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="mode-caption">Choose analysis source</div>', unsafe_allow_html=True)

input_mode = st.radio(
    "Choose analysis source",
    ["Paste Text", "Upload PDF / Word"],
    horizontal=True,
    label_visibility="collapsed",
)

if input_mode == "Paste Text":
    st.markdown(
        """
        <div class="module-card">
            <div class="module-title-row">
                <div class="module-badge">TXT</div>
                <div><div class="module-title">Paste ESG-related text</div></div>
            </div>
            <div class="module-subtitle">
                Paste one ESG-relevant paragraph, article excerpt, or full report text.
                Long content will be analyzed internally in smaller model-safe segments,
                while only the ESG analyst briefing is shown.
            </div>
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

    analyze_text_clicked = st.button("Generate ESG Analyst Briefing", type="primary")

    if analyze_text_clicked:
        cleaned_input = clean_input_text(input_text)

        if not cleaned_input:
            st.warning("Please paste ESG-related text before running the analysis.")
        else:
            word_count = len(cleaned_input.split())

            if word_count > 10000:
                st.info(
                    f"Large text detected: approximately {word_count:,} words. "
                    f"The app will analyze up to {DEFAULT_MAX_CHUNKS} internal segments "
                    f"and show only the ESG analyst briefing."
                )

            with st.spinner("Analyzing text and generating ESG analyst briefing..."):
                _, summary = analyze_long_document(
                    cleaned_input,
                    max_words=DEFAULT_MAX_WORDS,
                    overlap_words=DEFAULT_OVERLAP_WORDS,
                    max_chunks=DEFAULT_MAX_CHUNKS,
                    uploaded_file_name=None,
                )

            render_esg_analyst_briefing(summary)


elif input_mode == "Upload PDF / Word":
    st.markdown(
        """
        <div class="module-card">
            <div class="module-title-row">
                <div class="module-badge">DOC</div>
                <div><div class="module-title">Upload an ESG document</div></div>
            </div>
            <div class="module-subtitle">
                Upload a PDF, Word document, or TXT file. The app will extract the text,
                analyze the document internally in smaller model-safe segments, and return
                a concise ESG analyst briefing.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload a PDF, Word document, or TXT file",
        type=["pdf", "docx", "txt"],
        help="Supported formats: PDF, DOCX, TXT",
    )

    analyze_doc_clicked = st.button("Generate ESG Analyst Briefing", type="primary")

    if analyze_doc_clicked:
        if uploaded_file is None:
            st.warning("Please upload a PDF, Word document, or TXT file before analysis.")
        else:
            try:
                with st.spinner("Extracting text from uploaded document..."):
                    extracted_text = extract_text_from_uploaded_file(uploaded_file)
                    extracted_text = clean_input_text(extracted_text)

                if not extracted_text:
                    st.error(
                        "No readable text could be extracted from the uploaded document. "
                        "If the PDF is scanned, OCR is required before analysis."
                    )
                else:
                    total_words = len(extracted_text.split())

                    if total_words > 10000:
                        st.info(
                            f"Large document detected: approximately {total_words:,} words. "
                            f"The app will analyze up to {DEFAULT_MAX_CHUNKS} internal segments "
                            f"and show only the ESG analyst briefing."
                        )

                    with st.spinner("Analyzing document and generating ESG analyst briefing..."):
                        _, summary = analyze_long_document(
                            extracted_text,
                            max_words=DEFAULT_MAX_WORDS,
                            overlap_words=DEFAULT_OVERLAP_WORDS,
                            max_chunks=DEFAULT_MAX_CHUNKS,
                            uploaded_file_name=uploaded_file.name,
                        )

                    render_esg_analyst_briefing(summary)

            except Exception as exc:
                st.error("The uploaded document could not be analyzed.")
                st.write(str(exc))

st.markdown(
    f"""
    <div class="footer-note">
        {esc(DISCLAIMER_TEXT)}
    </div>
    """,
    unsafe_allow_html=True,
)
