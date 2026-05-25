# BlackRock ESG Risk Analyzer

## Description

BlackRock ESG Risk Analyzer is an ISOM5240 educational prototype for ESG investment research. The application screens corporate text with deep learning pipelines to identify ESG topics, detect financial sentiment, and generate an explainable rule-based ESG risk level for early-stage analyst review.

## Features

- Professional ESG SaaS-style Streamlit interface
- One main text input area for corporate ESG or financial text
- ESG category classification: Environmental, Social, Governance, or Non-ESG
- Financial sentiment detection: Positive, Neutral, or Negative
- Rule-based ESG risk level: Low, Medium, or High
- Business interpretation and recommended action
- Token-level truncation protection for long inputs
- Paragraph-style segmentation for long articles so the app can screen long text more safely
- CSV download for one-row analysis output

## Models Used

- ESG Classification: local fine-tuned model at `04_models/blackrock_esg_classifier` when available
- ESG Fallback: `yiyanghkust/finbert-esg`
- Financial Sentiment: `ProsusAI/finbert`

## How to Run Locally

From the project root folder:

```bash
pip install -r requirements.txt
streamlit run 03_app/app.py
```

## Streamlit Deployment Settings

- Main file path: `03_app/app.py`

## Expected Folder Structure

```text
BlackRock_ESG_Risk_Analyzer/
|-- 01_data/
|-- 02_notebooks/
|-- 03_app/
|   |-- app.py
|   |-- requirements.txt
|   `-- README.md
|-- 04_models/
|   `-- blackrock_esg_classifier/
|-- 05_results/
|-- 06_report/
`-- 07_presentation/
```

## Disclaimer

Educational prototype only. Not investment advice.
