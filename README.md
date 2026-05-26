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

Model logic:

- Uses a local fine-tuned DistilBERT ESG classifier if available.
- Falls back to `yiyanghkust/finbert-esg` if the local model is unavailable.

### 5.5 Financial Sentiment Analysis

The system detects financial sentiment using:

- `ProsusAI/finbert`

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

The app generates a formatted PDF report instead of a simple CSV export.

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

Primary model:

- Local fine-tuned DistilBERT ESG classifier

Fallback model:

- `yiyanghkust/finbert-esg`

Hugging Face model URL:

- `https://huggingface.co/yiyanghkust/finbert-esg`

### 6.2 Financial Sentiment Model

Model:

- `ProsusAI/finbert`

Hugging Face model URL:

- `https://huggingface.co/ProsusAI/finbert`

---

## 7. Risk Logic

The app uses transparent rule-based ESG risk scoring.

### 7.1 Segment-Level Risk Logic

| Condition | Risk Level |
|---|---|
| ESG + Negative | High |
| ESG + Neutral | Medium |
| ESG + Positive | Low |
| Non-ESG | Low |

### 7.2 Document-Level Risk Logic

For long documents, the app aggregates segment-level results.

General rules:

- Multiple high-risk segments indicate High overall ESG risk.
- One high-risk segment may indicate Medium risk.
- Multiple medium-risk segments indicate Medium risk.
- Mostly low-risk or Non-ESG content indicates Low risk.

This design prevents one weak signal from automatically dominating the entire document.

---

## 8. Application Workflow

1. User inputs text or uploads a document.
2. The system extracts and cleans text.
3. Long documents are split into internal segments.
4. ESG classification is performed.
5. Financial sentiment analysis is performed.
6. Rule-based ESG risk scoring is applied.
7. Risk drivers are extracted.
8. Evidence highlights are selected.
9. A structured ESG analyst briefing is generated.
10. A PDF report can be downloaded.

---

## 9. Project Structure

```text
BlackRock_ESG_Risk_Analyzer/
│
├── 01_data/
│   ├── raw/
│   ├── processed/
│   └── manual_test/
│
├── 02_notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_baseline_model_testing.ipynb
│   ├── 03_finetune_distilbert.ipynb
│   ├── 04_model_comparison.ipynb
│   └── 05_streamlit_app_testing.ipynb
│
├── 03_app/
│   ├── app.py
│   └── requirements.txt
│
├── 04_models/
│   └── blackrock_esg_classifier/
│
├── 05_reports/
│   ├── project_report.pdf
│   ├── presentation.pptx
│   └── experimental_results.xlsx
│
├── README.md
└── requirements.txt

## 10. Installation

### 10.1 Clone the Repository

    git clone <your-github-repository-url>
    cd BlackRock_ESG_Risk_Analyzer

### 10.2 Create a Virtual Environment

    python -m venv venv

Activate the environment.

For macOS / Linux:

    source venv/bin/activate

For Windows:

    venv\Scripts\activate

### 10.3 Install Dependencies

    pip install -r requirements.txt

---

## 11. Requirements

The required Python packages are:

    streamlit
    transformers
    torch
    pandas
    numpy
    accelerate
    safetensors
    pypdf
    python-docx
    reportlab

---

## 12. Running the App Locally

Run the Streamlit app with:

    streamlit run 03_app/app.py

Then open the local URL shown in the terminal.

Usually:

    http://localhost:8501

---

## 13. Streamlit Deployment

The app can be deployed on Streamlit Cloud.

### 13.1 Deployment Steps

1. Push the project to GitHub.
2. Go to Streamlit Cloud.
3. Create a new app.
4. Select the GitHub repository.
5. Set the main app file path:

       03_app/app.py

6. Make sure `requirements.txt` includes all required packages.
7. Deploy the app.

### 13.2 Current App URL

    https://esg-analyst-briefing-system-for-blackrock.streamlit.app/

---

## 14. How to Use the App

### Option 1: Paste Text

1. Open the app.
2. Select `Paste Text`.
3. Paste ESG-related text into the text area.
4. Click `Generate ESG Analyst Briefing`.
5. Review the ESG analyst briefing.
6. Download the PDF report.

### Option 2: Upload PDF / Word

1. Open the app.
2. Select `Upload PDF / Word`.
3. Upload a PDF, DOCX, or TXT document.
4. Click `Generate ESG Analyst Briefing`.
5. Review the ESG analyst briefing.
6. Download the PDF report.

---

## 15. Example Test Inputs

### Example 1

    The company disclosed that it is under regulatory investigation for misleading climate-related claims and insufficient emissions reporting. Investors raised concerns about weak board oversight and potential greenwashing risk.

Expected signals:

- Top ESG Category: Governance or Environmental
- Dominant Sentiment: Negative
- Overall ESG Risk: High or Medium

### Example 2

    The company announced a new renewable energy procurement plan and committed to reducing Scope 1 and Scope 2 emissions by 2030. Management stated that the transition plan will be reviewed annually.

Expected signals:

- Top ESG Category: Environmental
- Dominant Sentiment: Positive or Neutral
- Overall ESG Risk: Low or Medium

---

## 16. Output Explanation

### Overall ESG Risk

The main ESG risk level assigned to the document:

- Low
- Medium
- High

### Top ESG Category

The dominant ESG topic detected in the document:

- Environmental
- Social
- Governance
- Non-ESG

### Dominant Sentiment

The dominant financial sentiment detected in the document:

- Positive
- Neutral
- Negative

### ESG Confidence

Average confidence score from the ESG classification model.

### Sentiment Confidence

Average confidence score from the financial sentiment model.

### Key Risk Drivers

Rule-based ESG keywords detected in the analyzed text.

### Evidence Highlights

Selected text snippets that support the ESG risk assessment.

### Materiality Assessment

A practical view of potential ESG materiality across:

- Regulatory Exposure
- Reputation Risk
- Operational Risk
- Disclosure Quality Concern

### Analyst Review Checklist

A list of practical review steps for analysts.

---

## 17. PDF Report

The downloaded PDF report is designed for documentation, presentation, and analyst review.

The PDF includes:

1. Executive ESG Summary
2. Key Metrics
3. Risk Interpretation
4. ESG Category Breakdown
5. Sentiment Breakdown
6. Key Risk Drivers
7. Evidence Highlights
8. Materiality Assessment
9. Analyst Review Checklist
10. Confidence and Limitations
11. Analysis Scope
12. Decision Support Note
13. Disclaimer

---

## 18. Limitations

This prototype has several limitations:

- It does not provide investment advice.
- It is not a final ESG rating system.
- It depends on transformer model outputs, which may be imperfect.
- Long documents are analyzed through internal segmentation.
- Scanned PDFs may not work unless OCR is performed before upload.
- The app does not currently scrape live web data.
- Risk scoring is rule-based and simplified for educational use.
- Analysts should verify results against original source documents.

---

## 19. Ethical and Compliance Notes

This app should be used only as a screening and research support tool.

Users should not treat the output as:

- Investment advice
- A final ESG score
- A credit rating
- A buy / sell / hold recommendation
- A replacement for professional analyst judgment

All outputs should be reviewed with original documents, company disclosures, external evidence, and human judgment.

---

## 20. Future Improvements

Potential future improvements include:

- Add OCR support for scanned PDFs.
- Add company name extraction.
- Add industry-specific ESG materiality logic.
- Add peer comparison module.
- Add controversy timeline detection.
- Add source citation extraction from uploaded reports.
- Add ESG topic heatmap.
- Add dashboard-level portfolio screening.
- Improve PDF visual design with charts.
- Fine-tune ESG classifier on a larger labeled dataset.

---

## 21. Disclaimer

This application is an educational prototype for ESG screening. It does not provide investment advice, does not replace professional judgment, and should not be used as a final ESG rating system.
