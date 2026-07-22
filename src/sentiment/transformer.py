"""Reusable Hugging Face training utilities for sentiment classification.

Supports:
- standard full fine-tuning (recommended for DistilBERT)
- LoRA parameter-efficient fine-tuning
- optional 4-bit QLoRA when bitsandbytes + compatible hardware are available
- automatic fallback to ordinary LoRA on Apple Silicon/CPU
"""
from __future__ import annotations

import inspect
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import Dataset

from src.config import ID_TO_LABEL, LABEL_TO_ID, RANDOM_STATE, SENTIMENT_LABELS
from src.data.evaluation import (
    calculate_classification_metrics,
    create_classification_report,
    create_confusion_matrix_dataframe,
    plot_confusion_matrix,
)

@dataclass
class TransformerExperimentConfig:
    model_id: str
    model_name: str
    pretrained_model: str
    output_dir: str | Path
    max_length: int = 256
    epochs: int = 3
    train_batch_size: int = 16
    eval_batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 1
    use_lora: bool = False
    use_4bit: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: tuple[str, ...] = ()
    random_state: int = RANDOM_STATE
    early_stopping_patience: int = 2
    use_class_weights: bool = True
    use_fp16: bool = False
    use_bf16: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["output_dir"] = str(data["output_dir"])
        return data

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length: int):
        self.texts = list(texts)
        self.labels = [LABEL_TO_ID[str(label)] for label in labels]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        encoded = self.tokenizer(
            self.texts[index], truncation=True, max_length=self.max_length
        )
        encoded["labels"] = self.labels[index]
        return encoded

def prepare_transformer_sentiment_data(
    df: pd.DataFrame,
    *, text_column: str = "transformer_text",
    target_column: str = "sentiment_label",
) -> pd.DataFrame:
    required = {text_column, target_column}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")
    model_df = df[[text_column, target_column]].dropna().copy()
    model_df[text_column] = model_df[text_column].astype(str).str.strip()
    model_df = model_df[model_df[text_column].str.len() > 0]
    model_df = model_df[model_df[target_column].isin(SENTIMENT_LABELS)]
    return model_df.reset_index(drop=True)

def _supports_4bit() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import bitsandbytes  # noqa: F401
        return True
    except ImportError:
        return False

def runtime_device() -> str:
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

def build_tokenizer_and_model(config: TransformerExperimentConfig):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model, use_fast=True)
    model_kwargs: dict[str, Any] = {
        "num_labels": len(SENTIMENT_LABELS),
        "label2id": LABEL_TO_ID,
        "id2label": ID_TO_LABEL,
    }
    using_4bit = bool(config.use_4bit and _supports_4bit())
    if config.use_4bit and not using_4bit:
        print("4-bit requested but unavailable on this machine; falling back to regular LoRA/full precision.")
    if using_4bit:
        from transformers import BitsAndBytesConfig
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        model_kwargs["device_map"] = "auto"

    model = AutoModelForSequenceClassification.from_pretrained(
        config.pretrained_model, **model_kwargs
    )

    if config.use_lora:
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        if using_4bit:
            model = prepare_model_for_kbit_training(model)
        targets = list(config.lora_target_modules) or ["query", "value"]
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=targets,
            bias="none",
            modules_to_save=["classifier", "score"],
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    return tokenizer, model, using_4bit

def _compute_metrics(eval_prediction):
    logits, labels = eval_prediction
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_precision": precision_score(labels, predictions, average="macro", zero_division=0),
        "macro_recall": recall_score(labels, predictions, average="macro", zero_division=0),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "weighted_f1": f1_score(labels, predictions, average="weighted", zero_division=0),
    }

class WeightedTrainerMixin:
    class_weights: torch.Tensor | None = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        weights = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss = nn.CrossEntropyLoss(weight=weights)(logits, labels)
        return (loss, outputs) if return_outputs else loss

def train_transformer(
    config: TransformerExperimentConfig,
    X_train, y_train, X_val, y_val,
):
    from transformers import (
        DataCollatorWithPadding, EarlyStoppingCallback, Trainer, TrainingArguments, set_seed,
    )
    set_seed(config.random_state)
    tokenizer, model, using_4bit = build_tokenizer_and_model(config)
    train_dataset = SentimentDataset(X_train, y_train, tokenizer, config.max_length)
    val_dataset = SentimentDataset(X_val, y_val, tokenizer, config.max_length)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(SENTIMENT_LABELS)),
        y=np.array([LABEL_TO_ID[str(label)] for label in y_train]),
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    args_kwargs = dict(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        seed=config.random_state,
        data_seed=config.random_state,
        
        fp16=config.use_fp16,
        bf16=config.use_bf16,
        dataloader_pin_memory=torch.cuda.is_available(),
    )
    signature = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in signature:
        args_kwargs["eval_strategy"] = "epoch"
    else:
        args_kwargs["evaluation_strategy"] = "epoch"
    training_args = TrainingArguments(**args_kwargs)

    WeightedTrainer = type("WeightedTrainer", (WeightedTrainerMixin, Trainer), {})
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=_compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)],
    )
    trainer_signature = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_signature:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = WeightedTrainer(**trainer_kwargs)
    trainer.class_weights = (
        torch.tensor(class_weights, dtype=torch.float32)
        if config.use_class_weights else None
    )

    start = time.perf_counter()
    train_result = trainer.train()
    training_time = time.perf_counter() - start
    return {
        "trainer": trainer, "model": model, "tokenizer": tokenizer,
        "training_time_seconds": training_time,
        "train_result": train_result, "using_4bit": using_4bit,
        "config": config,
    }

def evaluate_trained_transformer(run, X, y_true, *, confusion_matrix_path=None):
    trainer, tokenizer = run["trainer"], run["tokenizer"]
    dataset = SentimentDataset(X, y_true, tokenizer, run["config"].max_length)
    start = time.perf_counter()
    output = trainer.predict(dataset)
    elapsed = time.perf_counter() - start
    pred_ids = np.argmax(output.predictions, axis=-1)
    predictions = np.array([ID_TO_LABEL[int(i)] for i in pred_ids])
    probabilities = torch.softmax(torch.tensor(output.predictions), dim=-1).numpy()
    metrics = calculate_classification_metrics(y_true, predictions)
    report = create_classification_report(y_true, predictions, labels=SENTIMENT_LABELS)
    matrix = create_confusion_matrix_dataframe(y_true, predictions, labels=SENTIMENT_LABELS)
    plot_confusion_matrix(
        y_true, predictions, labels=SENTIMENT_LABELS,
        title=f"{run['config'].model_name} Confusion Matrix",
        save_path=confusion_matrix_path,
    )
    return {
        "predictions": predictions, "probabilities": probabilities,
        "metrics": metrics, "classification_report": report,
        "confusion_matrix": matrix,
        "inference": {
            "total_inference_seconds": elapsed,
            "average_inference_ms": elapsed / len(X) * 1000,
        },
    }

def save_transformer_artifact(run, path: str | Path) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    run["trainer"].save_model(str(output))
    run["tokenizer"].save_pretrained(str(output))
    return output

def save_transformer_predictions(X, y_true, evaluation, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({
        "text": pd.Series(X).reset_index(drop=True),
        "true_label": pd.Series(y_true).reset_index(drop=True),
        "predicted_label": evaluation["predictions"],
    })
    for index, label in ID_TO_LABEL.items():
        frame[f"probability_{label}"] = evaluation["probabilities"][:, index]
    frame.to_csv(output, index=False)
    return output

def plot_training_history(run, path: str | Path) -> Path:
    history = pd.DataFrame(run["trainer"].state.log_history)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    if "loss" in history:
        train_rows = history.dropna(subset=["loss"])
        ax.plot(train_rows["epoch"], train_rows["loss"], label="train loss")
    if "eval_loss" in history:
        eval_rows = history.dropna(subset=["eval_loss"])
        ax.plot(eval_rows["epoch"], eval_rows["eval_loss"], label="validation loss")
    ax.set_title(f"{run['config'].model_name} Training History")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.show()
    return output
