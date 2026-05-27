# ESG Analyst Briefing System for BlackRock

## Overview

The ESG Analyst Briefing System for BlackRock is a Streamlit application for ESG document screening and analyst support. It helps users review corporate disclosures, sustainability reports, annual report excerpts, and financial news by:

- classifying ESG topics,
- detecting financial sentiment,
- generating ESG risk signals, and
- producing structured analyst briefing outputs.

The final project keeps the deployed app model repo, the matching local training artifact, and the baseline reference model clearly separated.

This report is for informational purposes only, the information and opinions contained herein do not constitute investment advice to any person.

## Final Models

### ESG Classification

Primary app model:

- `johnsonzhangzzz/blackrock_esg_classifier`

Local training artifact:

- `04_models/local_final_esg_model`

Baseline reference model:

- `yiyanghkust/finbert-esg`

### Financial Sentiment Analysis

- `ProsusAI/finbert`

## Final Model Logic

The deployed app model is the Hugging Face repo `johnsonzhangzzz/blackrock_esg_classifier`.

The local folder `04_models/local_final_esg_model` is the matching training artifact kept in this repository for reproducibility.

If confidence-based fallback is enabled in a future update, the intended logic is:

1. Run the deployed app model first.
2. If the deployed model confidence is high enough, use that result.
3. If the deployed model confidence is low, use `yiyanghkust/finbert-esg` as a safeguard.

This naming keeps the app-facing model, the local training artifact, and the baseline reference model clearly separated.

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

- `01_data/processed/esg_combined.csv`
- `01_data/processed/esg_train.csv`
- `01_data/processed/esg_validation.csv`
- `01_data/processed/esg_test.csv`

Manual and evaluation data:

- `01_data/manual_test/esg_real_world_50_samples.csv`
- `01_data/manual_test/esg_hard_examples.csv`

## Final Model Files

- `04_models/local_final_esg_model/`

This folder contains the final fine-tuned DistilBERT ESG classifier used by the project.

## Final Result Files

- `05_results/local_final_model_evaluation_workbook.xlsx`
- `05_results/local_final_model_vs_reference_models_comparison.xlsx`
- `05_results/local_final_model_vs_reference_models_comparison.csv`
- `05_results/local_final_model_error_analysis.csv`
- `05_results/final_model_deployment_recommendation.txt`

These files provide the final evidence for model evaluation, comparison, error analysis, and model selection.

## Project Structure

```text
BlackRock_ESG_Risk_Analyzer/
├── README.md
├── requirements.txt
├── 01_data/
│   ├── processed/
│   │   ├── esg_combined.csv
│   │   ├── esg_train.csv
│   │   ├── esg_validation.csv
│   │   └── esg_test.csv
│   └── manual_test/
│       ├── esg_real_world_50_samples.csv
│       └── esg_hard_examples.csv
├── 02_notebooks/
│   └── 05_train_local_final_esg_model.ipynb
├── 03_app/
│   └── app.py
├── 04_models/
│   └── local_final_esg_model/
├── 05_results/
│   ├── local_final_model_evaluation_workbook.xlsx
│   ├── local_final_model_vs_reference_models_comparison.xlsx
│   ├── local_final_model_vs_reference_models_comparison.csv
│   ├── local_final_model_error_analysis.csv
│   └── final_model_deployment_recommendation.txt
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

Make sure the deployment environment includes the local final model folder or has access to the required model artifacts.

## Core Functions

The application supports:

- pasted ESG text analysis,
- uploaded document analysis,
- ESG topic classification,
- financial sentiment detection,
- ESG risk screening, and
- analyst briefing output generation.

## Final Recommendation

The local final ESG model is the best reproducible training artifact in this repository.

It outperformed the baseline reference model on the final internal test set. The deployed app model repo should be kept aligned with this local final training artifact.

## Notes

- The repository may contain archived development artifacts in `archive_old_files/`, but they are not part of the final submission workflow.
- The files listed above represent the final project state for reproduction, review, and deployment.
