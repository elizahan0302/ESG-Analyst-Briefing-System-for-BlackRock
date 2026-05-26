# ESG Analyst Briefing System for BlackRock

## 1. Project Overview

**ESG Analyst Briefing System for BlackRock** is an ESG document screening application designed to help analysts convert corporate ESG-related text and documents into structured analyst briefings.

The system supports pasted text and uploaded documents, classifies ESG topics, detects financial sentiment, identifies ESG risk drivers, and generates a formatted ESG analyst briefing report in PDF format.

This project is an educational prototype for ISOM5240. It supports ESG screening only and does not provide investment advice.

---

## 2. Business Problem

Investment analysts often need to review long ESG-related documents, including sustainability reports, annual report sections, regulatory disclosures, and ESG news articles. Manual review can be time-consuming, inconsistent, and difficult to scale.

This project addresses the following problems:

- ESG documents are often long and unstructured.
- Analysts need a faster way to screen ESG risk signals.
- ESG classification and sentiment interpretation can be inconsistent across users.
- Raw model outputs are not directly useful for business research.
- Analysts need structured briefing outputs that include risk level, evidence, materiality, and review recommendations.

---

## 3. Project Objective

The objective of this project is to build an ESG analyst briefing system for BlackRock that classifies ESG topics, detects financial sentiment, and generates structured ESG risk briefings to support faster and more consistent ESG document screening for investment research.

---

## 4. Target Users

The intended users include:

- ESG investment analysts
- Sustainable investing researchers
- Portfolio research teams
- Risk management teams
- Students and evaluators reviewing ESG text analytics prototypes

---

## 5. Key Features

### 5.1 Text Input

Users can paste ESG-related text directly into the application.

Supported examples include:

- ESG news articles
- Corporate sustainability disclosures
- Annual report excerpts
- ESG controversy descriptions
- Company announcements

### 5.2 Document Upload

Users can upload ESG-related documents for analysis.

Supported file formats:

- PDF
- DOCX
- TXT

### 5.3 Long Document Processing

The system supports long documents by splitting the text internally into smaller model-safe segments.

Default processing settings:

| Setting | Value |
|---|---:|
| Words per segment | 180 |
| Overlap words | 30 |
| Maximum segments | 80 |
| Transformer max length | 512 tokens |

The app does not show chunk-level tables to users. It only shows the overall ESG analyst briefing.

### 5.4 ESG Classification

The system classifies text into ESG categories:

- Environmental
- Social
- Governance
- Non-ESG

The ESG classifier is the project fine-tuned DistilBERT model uploaded to Hugging Face:

- `johnsonzhangzzz/blackrock_esg_classifier`

Model URL:

- `https://huggingface.co/johnsonzhangzzz/blackrock_esg_classifier`

### 5.5 Financial Sentiment Analysis

The system detects financial sentiment using:

- `ProsusAI/finbert`

Model URL:

- `https://huggingface.co/ProsusAI/finbert`

Sentiment labels:

- Positive
- Neutral
- Negative

### 5.6 ESG Risk Scoring

The system combines ESG category and sentiment to generate ESG risk level.

Risk logic:

| ESG Category | Sentiment | Risk Level |
|---|---|---|
| Non-ESG | Any | Low |
| Environmental / Social / Governance | Negative | High |
| Environmental / Social / Governance | Neutral | Medium |
| Environmental / Social / Governance | Positive | Low |

For long documents, segment-level outputs are aggregated into a document-level ESG risk profile.

### 5.7 ESG Analyst Briefing

The final output includes:

- Executive ESG Summary
- Overall ESG Risk Level
- Top ESG Category
- Dominant Sentiment
- ESG Confidence
- Sentiment Confidence
- ESG Risk Breakdown
- Sentiment Breakdown
- Key Risk Drivers
- Evidence Highlights
- Materiality Assessment
- Analyst Review Checklist
- Confidence and Limitations
- Analysis Scope
- Decision Support Note

### 5.8 PDF Report Export

The app generates a formatted PDF report instead of a CSV export.

The PDF report includes:

- Executive summary
- Key metrics table
- ESG category breakdown
- Sentiment breakdown
- Risk drivers
- Evidence highlights
- Materiality assessment
- Analyst review checklist
- Confidence and limitations
- Scope note
- Decision support note
- Disclaimer

---

## 6. Models Used

### 6.1 ESG Classification Model

Fine-tuned ESG classifier:

- `johnsonzhangzzz/blackrock_esg_classifier`

Hugging Face model URL:

- `https://huggingface.co/johnsonzhangzzz/blackrock_esg_classifier`

Base model:

- `distilbert-base-uncased`

Output labels:

- Environmental
- Social
- Governance
- Non-ESG

### 6.2 Financial Sentiment Model

Model:

- `ProsusAI/finbert`

Hugging Face model URL:

- `https://huggingface.co/ProsusAI/finbert`

Output labels:

- Positive
- Neutral
- Negative

---

## 7. Application Workflow

1. User inputs text or uploads a document.
2. The system extracts and cleans text.
3. Long documents are split into internal segments.
4. ESG classification is performed using the fine-tuned ESG classifier.
5. Financial sentiment analysis is performed using FinBERT.
6. Rule-based ESG risk scoring is applied.
7. Risk drivers are extracted.
8. Evidence highlights are selected.
9. A structured ESG analyst briefing is generated.
10. A PDF report can be downloaded.

---

## 8. Project Structure

```text
BlackRock_ESG_Risk_Analyzer/
│
├── 01_data/
│   ├── raw/
│   ├── processed/
│   └── manual_test/
│
├── 02_notebooks/
│   ├── 01_baseline_pipeline_testing.ipynb
│   ├── 02_esg_model_finetuning.ipynb
│   └── 03_model_comparison.ipynb
│
├── 03_app/
│   ├── app.py
│   └── requirements.txt
│
├── 04_models/
│   └── blackrock_esg_classifier/
│
├── 05_results/
│   ├── baseline_experimental_results.xlsx
│   ├── finetuned_esg_experimental_results.xlsx
│   └── model_comparison_results.xlsx
│
├── README.md
└── requirements.txt
