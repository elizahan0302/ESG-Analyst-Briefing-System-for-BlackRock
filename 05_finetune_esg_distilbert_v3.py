from __future__ import annotations

import json
import os
import re
import shutil
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / '05_results'
PROCESSED_DIR = PROJECT_ROOT / '01_data' / 'processed'
MANUAL_DIR = PROJECT_ROOT / '01_data' / 'manual_test'
MODELS_DIR = PROJECT_ROOT / '04_models'
APP_PATH = PROJECT_ROOT / '03_app' / 'app.py'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MANUAL_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = PROJECT_ROOT / '.hf_cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ['HF_HOME'] = str(CACHE_DIR)
os.environ['HUGGINGFACE_HUB_CACHE'] = str(CACHE_DIR / 'hub')
os.environ['TRANSFORMERS_CACHE'] = str(CACHE_DIR / 'transformers')
os.environ['HF_HUB_CACHE'] = str(CACHE_DIR / 'hub')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    pipeline,
)

FINE_TUNED_MODEL_NAME = 'johnsonzhangzzz/blackrock_esg_classifier'
FINE_TUNED_MODEL_URL = 'https://huggingface.co/johnsonzhangzzz/blackrock_esg_classifier'
BASELINE_MODEL_NAME = 'yiyanghkust/finbert-esg'
BASELINE_MODEL_URL = 'https://huggingface.co/yiyanghkust/finbert-esg'
BASE_MODEL_NAME = 'distilbert-base-uncased'
V3_MODEL_DIR = MODELS_DIR / 'blackrock_esg_classifier_v3'
LABEL_ORDER = ['Environmental', 'Social', 'Governance', 'Non-ESG']
LABEL2ID = {'Environmental': 0, 'Social': 1, 'Governance': 2, 'Non-ESG': 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

OUTPUTS = {
    'combined': PROCESSED_DIR / 'esg_combined_v3.csv',
    'hard_examples': MANUAL_DIR / 'esg_hard_examples_v3.csv',
    'train': PROCESSED_DIR / 'esg_train_v3.csv',
    'validation': PROCESSED_DIR / 'esg_validation_v3.csv',
    'test': PROCESSED_DIR / 'esg_test_v3.csv',
    'test_predictions': RESULTS_DIR / 'v3_finetuned_esg_test_predictions.csv',
    'real_predictions': RESULTS_DIR / 'v3_finetuned_real_world_50_predictions.csv',
    'manual_predictions': RESULTS_DIR / 'v3_finetuned_manual_test_predictions.csv',
    'metrics': RESULTS_DIR / 'v3_finetuned_esg_metrics.xlsx',
    'comparison_csv': RESULTS_DIR / 'v3_finetuned_vs_baseline_comparison.csv',
    'comparison_xlsx': RESULTS_DIR / 'v3_finetuned_vs_baseline_comparison.xlsx',
    'error_analysis': RESULTS_DIR / 'v3_finetuned_error_analysis.csv',
    'recommendation': RESULTS_DIR / 'v3_model_recommendation.txt',
    'readme_note': RESULTS_DIR / 'v3_readme_update_note.md',
}


def ts() -> str:
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def backup_if_exists(path: Path) -> Optional[Path]:
    if not path.exists():
        return None
    if path.is_dir():
        backup = path.parent / f'{path.name}.bak_{ts()}'
        shutil.copytree(path, backup)
    else:
        backup = path.parent / f'{path.stem}.bak_{ts()}{path.suffix}'
        shutil.copy2(path, backup)
    return backup


def normalize_esg_label(label: Any) -> str:
    mapping = {
        'Environmental': 'Environmental',
        'environmental': 'Environmental',
        'Social': 'Social',
        'social': 'Social',
        'Governance': 'Governance',
        'governance': 'Governance',
        'None': 'Non-ESG',
        'none': 'Non-ESG',
        'Non-ESG': 'Non-ESG',
        'non-esg': 'Non-ESG',
        'Non ESG': 'Non-ESG',
        'non esg': 'Non-ESG',
        'LABEL_0': 'Environmental',
        'LABEL_1': 'Social',
        'LABEL_2': 'Governance',
        'LABEL_3': 'Non-ESG',
    }
    return mapping.get(str(label).strip(), str(label).strip())


def normalize_sentiment_label(label: Any) -> str:
    mapping = {
        'positive': 'Positive',
        'Positive': 'Positive',
        'neutral': 'Neutral',
        'Neutral': 'Neutral',
        'negative': 'Negative',
        'Negative': 'Negative',
        '': '',
        'nan': '',
    }
    return mapping.get(str(label).strip(), str(label).strip())


def normalize_risk_label(label: Any, esg_label: str, sentiment_label: str) -> str:
    value = str(label).strip()
    if value in {'Low', 'Medium', 'High'}:
        return value
    if esg_label == 'Non-ESG':
        return 'Low'
    if sentiment_label == 'Negative':
        return 'High'
    if sentiment_label == 'Neutral':
        return 'Medium'
    if sentiment_label == 'Positive':
        return 'Low'
    return ''


def clean_text(text: Any) -> str:
    return re.sub(r'\s+', ' ', str(text).strip())


def inspect_project_data() -> List[Path]:
    candidates = [
        MANUAL_DIR / 'manual_esg_test.csv',
        MANUAL_DIR / 'esg_real_world_50_samples.csv',
        MANUAL_DIR / 'esg_hard_examples_v3.csv',
        PROCESSED_DIR / 'esg_train.csv',
        PROCESSED_DIR / 'esg_validation.csv',
        PROCESSED_DIR / 'esg_test.csv',
    ]
    found = []
    print('Inspecting ESG datasets...')
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            print(f'- {path.relative_to(PROJECT_ROOT)} :: shape={df.shape} columns={list(df.columns)}')
            found.append(path)
    return found


def canonicalize_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    filename = path.name
    if 'text' not in df.columns or 'esg_label' not in df.columns:
        return pd.DataFrame(columns=['sample_id', 'text', 'esg_label', 'sentiment_label', 'risk_label', 'source_name', 'source_url', 'company', 'split_source'])

    working = pd.DataFrame()
    working['sample_id'] = df['sample_id'] if 'sample_id' in df.columns else [f"{path.stem}_{i+1:03d}" for i in range(len(df))]
    working['text'] = df['text'].map(clean_text)
    working['esg_label'] = df['esg_label'].map(normalize_esg_label)
    working['sentiment_label'] = df['sentiment_label'].map(normalize_sentiment_label) if 'sentiment_label' in df.columns else ''
    working['source_name'] = ''
    if 'source_name' in df.columns:
        working['source_name'] = df['source_name'].fillna('').astype(str)
    elif 'source' in df.columns:
        working['source_name'] = df['source'].fillna('').astype(str)
    else:
        working['source_name'] = path.stem
    working['source_url'] = df['source_url'].fillna('').astype(str) if 'source_url' in df.columns else ''
    working['company'] = df['company'].fillna('Unknown').astype(str) if 'company' in df.columns else 'Unknown'
    risk_col = df['risk_label'] if 'risk_label' in df.columns else [''] * len(df)
    working['risk_label'] = [
        normalize_risk_label(risk_col.iloc[i] if hasattr(risk_col, 'iloc') else risk_col[i], working.iloc[i]['esg_label'], working.iloc[i]['sentiment_label'])
        for i in range(len(working))
    ]
    working['split_source'] = filename
    return working


def build_combined_dataset(dataset_paths: List[Path]) -> pd.DataFrame:
    frames = [canonicalize_dataset(path) for path in dataset_paths]
    combined = pd.concat(frames, ignore_index=True)
    combined['text'] = combined['text'].map(clean_text)
    combined['esg_label'] = combined['esg_label'].map(normalize_esg_label)
    combined['sentiment_label'] = combined['sentiment_label'].map(normalize_sentiment_label)
    combined = combined.dropna(subset=['text', 'esg_label'])
    combined = combined[combined['text'].str.split().str.len() >= 8].copy()
    combined = combined[combined['esg_label'].isin(LABEL_ORDER)].copy()
    combined = combined.drop_duplicates(subset=['text']).reset_index(drop=True)
    return combined


def create_splits(combined: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, temp_df = train_test_split(
        combined,
        test_size=0.30,
        random_state=42,
        stratify=combined['esg_label'],
    )
    validation_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df['esg_label'],
    )
    train_df = train_df.copy(); train_df['split'] = 'train'
    validation_df = validation_df.copy(); validation_df['split'] = 'validation'
    test_df = test_df.copy(); test_df['split'] = 'test'
    return train_df.reset_index(drop=True), validation_df.reset_index(drop=True), test_df.reset_index(drop=True)


def print_split_distribution(name: str, df: pd.DataFrame) -> None:
    print(f'{name} split shape: {df.shape}')
    print(df['esg_label'].value_counts().reindex(LABEL_ORDER, fill_value=0).to_dict())


def class_weight_tensor(train_df: pd.DataFrame) -> torch.Tensor:
    counts = train_df['esg_label'].value_counts().reindex(LABEL_ORDER, fill_value=1)
    total = counts.sum()
    weights = total / (len(LABEL_ORDER) * counts)
    return torch.tensor(weights.values, dtype=torch.float)


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights: Optional[torch.Tensor] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        logits = outputs.get('logits')
        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
            loss_fct = torch.nn.CrossEntropyLoss(weight=weight)
        else:
            loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def load_hf_pipeline(model_name_or_path: str):
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, cache_dir=str(CACHE_DIR))
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=str(CACHE_DIR))
    pipe = pipeline('text-classification', model=model, tokenizer=tokenizer)
    return pipe, model, tokenizer


def safe_pipeline_predict(pipe, text: str) -> Dict[str, Any]:
    result = pipe(text, truncation=True, max_length=512)[0]
    return {
        'raw_label': result['label'],
        'predicted_label': normalize_esg_label(result['label']),
        'confidence': float(result['score']),
    }


def compute_metrics_from_labels(y_true: Iterable[str], y_pred: Iterable[str]) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    y_true = list(y_true)
    y_pred = list(y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=LABEL_ORDER, average='macro', zero_division=0)
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=LABEL_ORDER, average='weighted', zero_division=0)
    per_class_precision, per_class_recall, per_class_f1, support = precision_recall_fscore_support(y_true, y_pred, labels=LABEL_ORDER, average=None, zero_division=0)
    metrics = {
        'accuracy': accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
    }
    per_class_df = pd.DataFrame({'label': LABEL_ORDER, 'precision': per_class_precision, 'recall': per_class_recall, 'f1': per_class_f1, 'support': support})
    cm_df = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=LABEL_ORDER), index=LABEL_ORDER, columns=LABEL_ORDER)
    return metrics, per_class_df, cm_df


def trainer_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    y_true = [ID2LABEL[i] for i in labels]
    y_pred = [ID2LABEL[i] for i in predictions]
    metrics, _, _ = compute_metrics_from_labels(y_true, y_pred)
    return metrics


def tokenize_dataset(dataset_dict: DatasetDict, tokenizer) -> DatasetDict:
    def tokenize_batch(batch):
        return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=256)
    tokenized = dataset_dict.map(tokenize_batch, batched=True)
    removable = [c for c in tokenized['train'].column_names if c not in {'labels', 'input_ids', 'attention_mask'}]
    tokenized = tokenized.remove_columns(removable)
    tokenized.set_format('torch')
    return tokenized


def build_training_args(output_dir: Path, logging_dir: Path) -> TrainingArguments:
    common = dict(
        output_dir=str(output_dir),
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01,
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='macro_f1',
        greater_is_better=True,
        logging_dir=str(logging_dir),
        logging_steps=5,
        save_total_limit=2,
        report_to='none',
        seed=42,
    )
    try:
        return TrainingArguments(eval_strategy='epoch', **common)
    except TypeError:
        return TrainingArguments(evaluation_strategy='epoch', **common)


def train_v3_model(train_df: pd.DataFrame, validation_df: pd.DataFrame) -> Dict[str, Any]:
    train_work = train_df[['text', 'esg_label']].copy()
    val_work = validation_df[['text', 'esg_label']].copy()
    train_work['labels'] = train_work['esg_label'].map(LABEL2ID)
    val_work['labels'] = val_work['esg_label'].map(LABEL2ID)
    dataset_dict = DatasetDict({
        'train': Dataset.from_pandas(train_work[['text', 'labels']], preserve_index=False),
        'validation': Dataset.from_pandas(val_work[['text', 'labels']], preserve_index=False),
    })
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, cache_dir=str(CACHE_DIR))
    tokenized = tokenize_dataset(dataset_dict, tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_NAME,
        num_labels=4,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        cache_dir=str(CACHE_DIR),
    )
    class_weights = class_weight_tensor(train_df)
    backup_if_exists(V3_MODEL_DIR)
    training_args = build_training_args(MODELS_DIR / 'training_outputs' / 'v3_distilbert', RESULTS_DIR / 'logs' / 'v3_distilbert')
    callbacks = []
    try:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=2))
    except Exception:
        callbacks = []
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized['train'],
        eval_dataset=tokenized['validation'],
        tokenizer=tokenizer,
        compute_metrics=trainer_metrics,
        callbacks=callbacks,
        class_weights=class_weights,
    )
    start = time.time()
    trainer.train()
    training_runtime = time.time() - start
    eval_metrics = trainer.evaluate(tokenized['validation'])
    if V3_MODEL_DIR.exists():
        shutil.rmtree(V3_MODEL_DIR)
    V3_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(V3_MODEL_DIR))
    tokenizer.save_pretrained(str(V3_MODEL_DIR))
    (V3_MODEL_DIR / 'label_mapping.json').write_text(json.dumps({'label2id': LABEL2ID, 'id2label': ID2LABEL}, indent=2), encoding='utf-8')
    return {
        'trainer': trainer,
        'tokenizer': tokenizer,
        'training_runtime': training_runtime,
        'validation_metrics': eval_metrics,
        'class_weights': class_weights.tolist(),
    }


def evaluate_dataset(df: pd.DataFrame, pipe, dataset_name: str, model_type: str, model_name: str) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    work = df.copy()
    if 'sample_id' not in work.columns:
        work['sample_id'] = [f'{dataset_name}_{i+1:03d}' for i in range(len(work))]
    work['text'] = work['text'].map(clean_text)
    work['expected_label'] = work['esg_label'].map(normalize_esg_label)
    preds = []
    start = time.time()
    for text in work['text']:
        preds.append(safe_pipeline_predict(pipe, text))
    runtime = time.time() - start
    work['raw_label'] = [p['raw_label'] for p in preds]
    work['predicted_label'] = [p['predicted_label'] for p in preds]
    work['confidence'] = [p['confidence'] for p in preds]
    work['dataset'] = dataset_name
    work['model_type'] = model_type
    work['model_name'] = model_name
    metrics, per_class_df, cm_df = compute_metrics_from_labels(work['expected_label'], work['predicted_label'])
    metrics.update({
        'dataset': dataset_name,
        'model_type': model_type,
        'model_name': model_name,
        'runtime_seconds': runtime,
        'average_inference_time_per_sample': runtime / len(work) if len(work) else 0.0,
        'test_samples': len(work),
    })
    return work, metrics, per_class_df, cm_df


def error_type(expected_label: str, predicted_label: str) -> str:
    if expected_label == 'Environmental' and predicted_label == 'Governance':
        return 'Environmental_vs_Governance'
    if expected_label == 'Governance' and predicted_label == 'Environmental':
        return 'Governance_vs_Environmental'
    if expected_label == 'Social' and predicted_label == 'Governance':
        return 'Social_vs_Governance'
    if expected_label == 'Governance' and predicted_label == 'Social':
        return 'Governance_vs_Social'
    if expected_label in {'Environmental', 'Social', 'Governance'} and predicted_label == 'Non-ESG':
        return 'ESG_missed'
    if expected_label == 'Non-ESG' and predicted_label in {'Environmental', 'Social', 'Governance'}:
        return 'False_ESG_signal'
    return 'Other'


def build_error_analysis(prediction_frames: List[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for frame in prediction_frames:
        wrong = frame[frame['expected_label'] != frame['predicted_label']]
        for _, row in wrong.iterrows():
            rows.append({
                'dataset': row['dataset'],
                'sample_id': row['sample_id'],
                'text': row['text'],
                'expected_label': row['expected_label'],
                'predicted_label': row['predicted_label'],
                'confidence': row['confidence'],
                'error_type': error_type(row['expected_label'], row['predicted_label']),
            })
    return pd.DataFrame(rows)


def write_metrics_workbook(v3_tables: Dict[str, pd.DataFrame], summary_tables: Dict[str, pd.DataFrame], cm_tables: Dict[str, pd.DataFrame], error_df: pd.DataFrame) -> None:
    backup_if_exists(OUTPUTS['metrics'])
    with pd.ExcelWriter(OUTPUTS['metrics'], engine='openpyxl') as writer:
        v3_tables['v3_test_predictions'].to_excel(writer, sheet_name='v3_test_predictions', index=False)
        v3_tables['real_world_50_predictions'].to_excel(writer, sheet_name='real_world_50_predictions', index=False)
        v3_tables['manual_test_predictions'].to_excel(writer, sheet_name='manual_test_predictions', index=False)
        summary_tables['v3_test_metrics'].to_excel(writer, sheet_name='v3_test_metrics', index=False)
        summary_tables['real_world_50_metrics'].to_excel(writer, sheet_name='real_world_50_metrics', index=False)
        summary_tables['manual_test_metrics'].to_excel(writer, sheet_name='manual_test_metrics', index=False)
        cm_tables['confusion_matrix_v3_test'].to_excel(writer, sheet_name='confusion_matrix_v3_test')
        cm_tables['confusion_matrix_real_world_50'].to_excel(writer, sheet_name='confusion_matrix_real_world_50')
        cm_tables['confusion_matrix_manual_test'].to_excel(writer, sheet_name='confusion_matrix_manual_test')
        error_df.to_excel(writer, sheet_name='error_analysis', index=False)


def write_comparison_files(comparison_df: pd.DataFrame) -> None:
    backup_if_exists(OUTPUTS['comparison_csv'])
    backup_if_exists(OUTPUTS['comparison_xlsx'])
    comparison_df.to_csv(OUTPUTS['comparison_csv'], index=False)
    with pd.ExcelWriter(OUTPUTS['comparison_xlsx'], engine='openpyxl') as writer:
        comparison_df.to_excel(writer, sheet_name='comparison', index=False)


def create_recommendation(v3_primary: Dict[str, Any], baseline_primary: Dict[str, Any], old_primary: Optional[Dict[str, Any]], error_df: pd.DataFrame) -> str:
    delta = baseline_primary['accuracy'] - v3_primary['accuracy']
    if v3_primary['accuracy'] >= baseline_primary['accuracy']:
        selection = 'Recommend v3 only.'
        fallback_note = 'Confidence fallback is optional and can remain disabled if you want a pure v3 deployment.'
    elif delta <= 0.10:
        selection = 'Recommend v3 as primary with baseline fallback when confidence < 0.55.'
        fallback_note = 'Confidence fallback is recommended because v3 is close to baseline but not stronger.'
    else:
        selection = 'Recommend hybrid mode: use v3 only when confidence >= 0.65, otherwise fall back to baseline.'
        fallback_note = 'Confidence fallback is required because v3 trails the baseline by more than 0.10 accuracy.'
    if v3_primary['accuracy'] < 0.50:
        selection += ' More data is recommended before production.'
    pattern_counts = error_df['error_type'].value_counts().to_dict() if not error_df.empty else {}
    lines = [
        f'v3 model path: {V3_MODEL_DIR}',
        f'old fine-tuned model URL: {FINE_TUNED_MODEL_URL}',
        f'baseline model URL: {BASELINE_MODEL_URL}',
        'datasets used: esg_test_v3, esg_real_world_50_samples, manual_esg_test',
        f"v3 accuracy on esg_test_v3: {v3_primary['accuracy']:.4f}",
        f"v3 macro F1 on esg_test_v3: {v3_primary['macro_f1']:.4f}",
        f"baseline accuracy on esg_test_v3: {baseline_primary['accuracy']:.4f}",
        f"baseline macro F1 on esg_test_v3: {baseline_primary['macro_f1']:.4f}",
    ]
    if old_primary is not None:
        lines.extend([
            f"old fine-tuned accuracy on esg_test_v3: {old_primary['accuracy']:.4f}",
            f"old fine-tuned macro F1 on esg_test_v3: {old_primary['macro_f1']:.4f}",
        ])
    lines.extend([
        f"runtime comparison on esg_test_v3: v3={v3_primary['runtime_seconds']:.4f}s baseline={baseline_primary['runtime_seconds']:.4f}s",
        f'main error patterns: {pattern_counts}',
        f"v3 reliable enough for app: {'yes' if v3_primary['accuracy'] >= 0.50 else 'not yet'}",
        fallback_note,
        f'final recommendation: {selection}',
    ])
    return '\n'.join(lines)


def update_app_minimally() -> bool:
    if not APP_PATH.exists():
        return False
    backup_if_exists(APP_PATH)
    text = APP_PATH.read_text(encoding='utf-8')
    text = text.replace(
        'LOCAL_ESG_MODEL_DIR = PROJECT_ROOT / "04_models" / "blackrock_esg_classifier"',
        'LOCAL_ESG_MODEL_DIR = PROJECT_ROOT / "04_models" / "blackrock_esg_classifier_v3"\nBASELINE_ESG_MODEL_NAME = "yiyanghkust/finbert-esg"\nV3_CONFIDENCE_THRESHOLD = 0.65'
    )
    old_load = '''@st.cache_resource(show_spinner=False)\ndef load_esg_pipeline():\n    """\n    Load local fine-tuned ESG model if available.\n    Fallback to FinBERT-ESG baseline if local model is unavailable.\n    """\n    device, device_name = get_device()\n\n    local_model_loaded = False\n    model_name = "yiyanghkust/finbert-esg"\n\n    if LOCAL_ESG_MODEL_DIR.exists():\n        try:\n            esg_pipe = pipeline(\n                task="text-classification",\n                model=str(LOCAL_ESG_MODEL_DIR),\n                tokenizer=str(LOCAL_ESG_MODEL_DIR),\n                device=device,\n            )\n            local_model_loaded = True\n            model_name = "Local fine-tuned DistilBERT ESG classifier"\n            return esg_pipe, model_name, local_model_loaded, device_name\n        except Exception:\n            pass\n\n    esg_pipe = pipeline(\n        task="text-classification",\n        model="yiyanghkust/finbert-esg",\n        tokenizer="yiyanghkust/finbert-esg",\n        device=device,\n    )\n    return esg_pipe, model_name, local_model_loaded, device_name\n'''
    new_load = '''@st.cache_resource(show_spinner=False)\ndef load_esg_pipeline():\n    """Load the local v3 ESG model when available."""\n    device, device_name = get_device()\n    if LOCAL_ESG_MODEL_DIR.exists():\n        try:\n            esg_pipe = pipeline(\n                task="text-classification",\n                model=str(LOCAL_ESG_MODEL_DIR),\n                tokenizer=str(LOCAL_ESG_MODEL_DIR),\n                device=device,\n            )\n            return esg_pipe, True, device_name\n        except Exception:\n            pass\n    return None, False, device_name\n\n\n@st.cache_resource(show_spinner=False)\ndef load_baseline_esg_pipeline():\n    """Load the baseline ESG classifier used as a low-confidence safeguard."""\n    device, _ = get_device()\n    return pipeline(\n        task="text-classification",\n        model=BASELINE_ESG_MODEL_NAME,\n        tokenizer=BASELINE_ESG_MODEL_NAME,\n        device=device,\n    )\n\n\ndef choose_esg_prediction(text: str) -> Dict[str, Any]:\n    """Use v3 first and fall back to baseline when v3 confidence is low."""\n    esg_pipe, local_model_loaded, device_name = load_esg_pipeline()\n    baseline_pipe = load_baseline_esg_pipeline()\n\n    if esg_pipe is not None:\n        v3_result = safe_pipeline_predict(esg_pipe, text, max_length=512)\n        chosen = {\n            "esg_category": normalize_esg_label(v3_result.get("label", "")),\n            "esg_confidence": float(v3_result.get("score", np.nan)),\n            "device": device_name,\n        }\n        if chosen["esg_confidence"] >= V3_CONFIDENCE_THRESHOLD:\n            return chosen\n\n    baseline_result = safe_pipeline_predict(baseline_pipe, text, max_length=512)\n    return {\n        "esg_category": normalize_esg_label(baseline_result.get("label", "")),\n        "esg_confidence": float(baseline_result.get("score", np.nan)),\n        "device": device_name,\n    }\n'''
    if old_load in text:
        text = text.replace(old_load, new_load)
    text = text.replace(
        '    esg_pipe, esg_model_name, local_model_loaded, device_name = load_esg_pipeline()\n    sentiment_pipe, sentiment_model_name, _ = load_sentiment_pipeline()\n\n    esg_result = safe_pipeline_predict(esg_pipe, cleaned_text, max_length=512)\n    sentiment_result = safe_pipeline_predict(sentiment_pipe, cleaned_text, max_length=512)\n\n    esg_category = normalize_esg_label(esg_result.get("label", ""))\n    esg_confidence = float(esg_result.get("score", np.nan))\n',
        '    esg_choice = choose_esg_prediction(cleaned_text)\n    sentiment_pipe, _, _ = load_sentiment_pipeline()\n\n    sentiment_result = safe_pipeline_predict(sentiment_pipe, cleaned_text, max_length=512)\n\n    esg_category = esg_choice["esg_category"]\n    esg_confidence = float(esg_choice["esg_confidence"])\n    device_name = esg_choice["device"]\n'
    )
    text = text.replace(
        '        "esg_model_name": esg_model_name,\n        "sentiment_model_name": sentiment_model_name,\n        "local_model_loaded": local_model_loaded,\n        "device": device_name,\n',
        '        "device": device_name,\n'
    )
    text = text.replace(
        '    esg_pipe, _, _, _ = load_esg_pipeline()\n    sentiment_pipe, _, _ = load_sentiment_pipeline()\n\n    esg_result = safe_pipeline_predict(esg_pipe, chunk_text, max_length=512)\n    sentiment_result = safe_pipeline_predict(sentiment_pipe, chunk_text, max_length=512)\n\n    esg_category = normalize_esg_label(esg_result.get("label", ""))\n    esg_confidence = float(esg_result.get("score", np.nan))\n',
        '    esg_choice = choose_esg_prediction(chunk_text)\n    sentiment_pipe, _, _ = load_sentiment_pipeline()\n\n    sentiment_result = safe_pipeline_predict(sentiment_pipe, chunk_text, max_length=512)\n\n    esg_category = esg_choice["esg_category"]\n    esg_confidence = float(esg_choice["esg_confidence"])\n'
    )
    text = text.replace('ESG Classifier: DistilBERT fine-tuned or FinBERT-ESG fallback<br>\n                        Sentiment: ProsusAI/finbert<br>', 'Primary ESG classifier with confidence safeguard<br>\n                        Financial sentiment classifier<br>')
    text = text.replace('Educational prototype only. This application supports ESG\n                        risk screening and does not provide investment advice.', 'This report is for informational purposes only, the information and opinions contained herein do not constitute investment advice to any person.')
    text = text.replace('        f"ESG model: {result[\'esg_model_name\']} · "\n        f"Sentiment model: {result[\'sentiment_model_name\']} · "\n        f"Device: {result[\'device\']}"\n', '        f"Processed on {result[\'device\']}"\n')
    text = text.replace('        Educational prototype for ISOM5240. This application supports ESG risk screening only\n        and does not provide investment advice.', '        This report is for informational purposes only, the information and opinions contained herein do not constitute investment advice to any person.')
    APP_PATH.write_text(text, encoding='utf-8')
    return True


def main() -> None:
    created_files = []
    found_datasets = inspect_project_data()
    if OUTPUTS['hard_examples'] not in found_datasets and OUTPUTS['hard_examples'].exists():
        found_datasets.append(OUTPUTS['hard_examples'])

    for target in [OUTPUTS['combined'], OUTPUTS['train'], OUTPUTS['validation'], OUTPUTS['test']]:
        backup_if_exists(target)

    combined = build_combined_dataset(found_datasets)
    combined.to_csv(OUTPUTS['combined'], index=False)
    created_files.append(OUTPUTS['combined'])

    train_df, validation_df, test_df = create_splits(combined)
    train_df.to_csv(OUTPUTS['train'], index=False)
    validation_df.to_csv(OUTPUTS['validation'], index=False)
    test_df.to_csv(OUTPUTS['test'], index=False)
    created_files.extend([OUTPUTS['train'], OUTPUTS['validation'], OUTPUTS['test']])
    print_split_distribution('train', train_df)
    print_split_distribution('validation', validation_df)
    print_split_distribution('test', test_df)

    train_info = train_v3_model(train_df, validation_df)
    created_files.append(V3_MODEL_DIR)

    v3_pipe, _, _ = load_hf_pipeline(str(V3_MODEL_DIR))
    baseline_pipe, _, _ = load_hf_pipeline(BASELINE_MODEL_NAME)
    try:
        old_pipe, _, _ = load_hf_pipeline(FINE_TUNED_MODEL_NAME)
        old_available = True
    except Exception:
        old_pipe = None
        old_available = False

    eval_sets = {
        'esg_test_v3': test_df,
        'real_world_50': canonicalize_dataset(MANUAL_DIR / 'esg_real_world_50_samples.csv') if (MANUAL_DIR / 'esg_real_world_50_samples.csv').exists() else None,
        'manual_test': canonicalize_dataset(MANUAL_DIR / 'manual_esg_test.csv') if (MANUAL_DIR / 'manual_esg_test.csv').exists() else None,
    }

    v3_prediction_tables = {}
    metric_rows = []
    comparison_rows = []
    per_class_tables = {}
    cm_tables = {}
    baseline_metric_lookup = {}
    old_metric_lookup = {}
    error_frames = []

    prediction_output_map = {
        'esg_test_v3': OUTPUTS['test_predictions'],
        'real_world_50': OUTPUTS['real_predictions'],
        'manual_test': OUTPUTS['manual_predictions'],
    }

    for dataset_name, frame in eval_sets.items():
        if frame is None or frame.empty:
            continue
        v3_preds, v3_metrics, v3_per_class, v3_cm = evaluate_dataset(frame, v3_pipe, dataset_name, 'v3_fine_tuned', str(V3_MODEL_DIR))
        baseline_preds, baseline_metrics, baseline_per_class, baseline_cm = evaluate_dataset(frame, baseline_pipe, dataset_name, 'baseline', BASELINE_MODEL_NAME)
        v3_preds.to_csv(prediction_output_map[dataset_name], index=False)
        created_files.append(prediction_output_map[dataset_name])
        v3_prediction_tables[f'{dataset_name}_predictions' if dataset_name != 'real_world_50' else 'real_world_50_predictions'] = v3_preds
        key_name = dataset_name if dataset_name != 'real_world_50' else 'real_world_50'
        metric_rows.append({'dataset_name': dataset_name, **v3_metrics})
        comparison_rows.append(v3_metrics)
        comparison_rows.append(baseline_metrics)
        baseline_metric_lookup[dataset_name] = baseline_metrics
        per_class_tables[f'{dataset_name}_v3_per_class'] = v3_per_class
        cm_tables[f'confusion_matrix_{dataset_name if dataset_name != "real_world_50" else "real_world_50"}'] = v3_cm
        error_frames.append(v3_preds)

        if dataset_name == 'esg_test_v3':
            per_class_tables['baseline_esg_test_v3_per_class'] = baseline_per_class
        if dataset_name == 'real_world_50':
            per_class_tables['baseline_real_world_50_per_class'] = baseline_per_class
        if dataset_name == 'manual_test':
            per_class_tables['baseline_manual_test_per_class'] = baseline_per_class

        if old_available and old_pipe is not None:
            _, old_metrics, _, _ = evaluate_dataset(frame, old_pipe, dataset_name, 'old_fine_tuned', FINE_TUNED_MODEL_NAME)
            comparison_rows.append(old_metrics)
            old_metric_lookup[dataset_name] = old_metrics

    error_df = build_error_analysis(error_frames)
    backup_if_exists(OUTPUTS['error_analysis'])
    error_df.to_csv(OUTPUTS['error_analysis'], index=False)
    created_files.append(OUTPUTS['error_analysis'])

    v3_test_metrics_df = pd.DataFrame([next(row for row in metric_rows if row['dataset_name'] == 'esg_test_v3')])
    real_world_metrics_df = pd.DataFrame([next(row for row in metric_rows if row['dataset_name'] == 'real_world_50')]) if any(row['dataset_name'] == 'real_world_50' for row in metric_rows) else pd.DataFrame()
    manual_metrics_df = pd.DataFrame([next(row for row in metric_rows if row['dataset_name'] == 'manual_test')]) if any(row['dataset_name'] == 'manual_test' for row in metric_rows) else pd.DataFrame()

    # Normalize workbook keys exactly as requested.
    workbook_predictions = {
        'v3_test_predictions': v3_prediction_tables['esg_test_v3_predictions'],
        'real_world_50_predictions': v3_prediction_tables.get('real_world_50_predictions', pd.DataFrame()),
        'manual_test_predictions': v3_prediction_tables.get('manual_test_predictions', pd.DataFrame()),
    }
    workbook_metrics = {
        'v3_test_metrics': v3_test_metrics_df,
        'real_world_50_metrics': real_world_metrics_df,
        'manual_test_metrics': manual_metrics_df,
    }
    workbook_cms = {
        'confusion_matrix_v3_test': cm_tables['confusion_matrix_esg_test_v3'],
        'confusion_matrix_real_world_50': cm_tables.get('confusion_matrix_real_world_50', pd.DataFrame()),
        'confusion_matrix_manual_test': cm_tables.get('confusion_matrix_manual_test', pd.DataFrame()),
    }
    write_metrics_workbook(workbook_predictions, workbook_metrics, workbook_cms, error_df)
    created_files.append(OUTPUTS['metrics'])

    comparison_df = pd.DataFrame(comparison_rows)[[
        'dataset', 'model_type', 'model_name', 'accuracy', 'macro_precision', 'macro_recall', 'macro_f1',
        'weighted_precision', 'weighted_recall', 'weighted_f1', 'runtime_seconds',
        'average_inference_time_per_sample', 'test_samples'
    ]]
    write_comparison_files(comparison_df)
    created_files.extend([OUTPUTS['comparison_csv'], OUTPUTS['comparison_xlsx']])

    v3_primary = next(row for row in comparison_rows if row['dataset'] == 'esg_test_v3' and row['model_type'] == 'v3_fine_tuned')
    baseline_primary = next(row for row in comparison_rows if row['dataset'] == 'esg_test_v3' and row['model_type'] == 'baseline')
    old_primary = next((row for row in comparison_rows if row['dataset'] == 'esg_test_v3' and row['model_type'] == 'old_fine_tuned'), None)
    recommendation_text = create_recommendation(v3_primary, baseline_primary, old_primary, error_df)
    backup_if_exists(OUTPUTS['recommendation'])
    OUTPUTS['recommendation'].write_text(recommendation_text, encoding='utf-8')
    created_files.append(OUTPUTS['recommendation'])

    readme_note = '\n'.join([
        '# v3 README update note',
        '- v3 fine-tuned model was trained on an expanded ESG dataset that combines existing labeled data, real-world samples, and hard examples.',
        '- The app uses confidence-based fallback logic.',
        '- The fine-tuned v3 model remains the primary project-specific classifier.',
        '- The baseline model is only used as a safeguard for low-confidence cases.',
        '- Output remains a screening signal, not investment advice.',
    ])
    backup_if_exists(OUTPUTS['readme_note'])
    OUTPUTS['readme_note'].write_text(readme_note, encoding='utf-8')
    created_files.append(OUTPUTS['readme_note'])

    app_updated = update_app_minimally()
    if app_updated:
        created_files.append(APP_PATH)

    print('DONE')
    for path in created_files:
        print(path)


if __name__ == '__main__':
    main()
