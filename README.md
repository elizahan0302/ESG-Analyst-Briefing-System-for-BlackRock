# ESG Analyst Briefing System for BlackRock

## Overview

The ESG Analyst Briefing System for BlackRock is a Streamlit application for ESG document screening and analyst support. It helps users review corporate disclosures, sustainability reports, annual report excerpts, and financial news by:

- classifying ESG topics,
- detecting financial sentiment,
- generating ESG risk signals, and
- producing structured analyst briefing outputs.

The final project uses the v3 fine-tuned ESG classifier as the main project-specific model and keeps a baseline ESG classifier as a safeguard for low-confidence cases.

This report is for informational purposes only, the information and opinions contained herein do not constitute investment advice to any person.

## Final Models

### ESG Classification

Primary model:

- `04_models/blackrock_esg_classifier_v3`

Baseline safeguard model:

- `yiyanghkust/finbert-esg`

### Financial Sentiment Analysis

- `ProsusAI/finbert`

## Final Model Logic

The final application uses the v3 fine-tuned ESG classifier first.

If confidence-based fallback is enabled, the logic is:

1. Run the local v3 ESG classifier.
2. If the v3 prediction confidence is high enough, use the v3 result.
3. If the v3 confidence is low, use `yiyanghkust/finbert-esg` as a safeguard.

This keeps the fine-tuned v3 model as the primary project-specific classifier while reducing low-confidence prediction risk.

## ESG Labels

The final ESG classification labels are:

- Environmental
- Social
- Governance
- Non-ESG

## Main Application File

- `03_app/app.py`

## Final Data Files

Processed training data:

- `01_data/processed/esg_combined_v3.csv`
- `01_data/processed/esg_train_v3.csv`
- `01_data/processed/esg_validation_v3.csv`
- `01_data/processed/esg_test_v3.csv`

Manual and evaluation data:

- `01_data/manual_test/esg_real_world_50_samples.csv`
- `01_data/manual_test/esg_hard_examples_v3.csv`

## Final Model Files

- `04_models/blackrock_esg_classifier_v3/`

This folder contains the final fine-tuned DistilBERT ESG classifier used by the project.

## Final Result Files

- `05_results/v3_finetuned_esg_metrics.xlsx`
- `05_results/v3_finetuned_vs_baseline_comparison.xlsx`
- `05_results/v3_finetuned_vs_baseline_comparison.csv`
- `05_results/v3_finetuned_error_analysis.csv`
- `05_results/v3_model_recommendation.txt`

These files provide the final evidence for model evaluation, comparison, error analysis, and model selection.

## Project Structure

```text
BlackRock_ESG_Risk_Analyzer/
├── README.md
├── requirements.txt
├── 01_data/
│   ├── processed/
│   │   ├── esg_combined_v3.csv
│   │   ├── esg_train_v3.csv
│   │   ├── esg_validation_v3.csv
│   │   └── esg_test_v3.csv
│   └── manual_test/
│       ├── esg_real_world_50_samples.csv
│       └── esg_hard_examples_v3.csv
├── 02_notebooks/
│   └── 05_finetune_esg_distilbert_v3.py
├── 03_app/
│   └── app.py
├── 04_models/
│   └── blackrock_esg_classifier_v3/
├── 05_results/
│   ├── v3_finetuned_esg_metrics.xlsx
│   ├── v3_finetuned_vs_baseline_comparison.xlsx
│   ├── v3_finetuned_vs_baseline_comparison.csv
│   ├── v3_finetuned_error_analysis.csv
│   └── v3_model_recommendation.txt
├── 06_report/
└── 07_presentation/
```

## Installation

From the project root:

```bash
pip install -r requirements.txt
```

## Requirements

The project requires:

- streamlit
- transformers
- torch
- pandas
- numpy
- accelerate
- safetensors
- pypdf
- python-docx
- reportlab
- huggingface_hub

## Run Locally

From the project root:

```bash
streamlit run 03_app/app.py
```

## Streamlit Deployment

For Streamlit Cloud deployment:

- Main file path: `03_app/app.py`
- Python dependencies: `requirements.txt`

Make sure the deployment environment includes the v3 model folder or has access to the required model artifacts.

## Core Functions

The application supports:

- pasted ESG text analysis,
- uploaded document analysis,
- ESG topic classification,
- financial sentiment detection,
- ESG risk screening, and
- analyst briefing output generation.

## Final Recommendation

The v3 fine-tuned ESG classifier is the best current project-specific model in this repository.

It outperformed the older fine-tuned versions and exceeded the baseline ESG classifier on the final v3 test set. The baseline model remains useful as a safeguard for low-confidence cases, but the v3 model is the main ESG classifier for the final project.

## Notes

- The repository may contain archived development artifacts in `archive_old_files/`, but they are not part of the final submission workflow.
- The files listed above represent the final project state for reproduction, review, and deployment.
