"""
StayBuddy — DistilBERT Retraining Script
Author: Samiya Saleem (22I-1065)

Usage:
    py -3.13 retrain.py

What this does:
    1. Loads training_data.json
    2. Encodes labels with LabelEncoder
    3. Fine-tunes DistilBERT for sequence classification
    4. Saves the model to intent_model/
    5. Saves label_encoder.pkl
    6. Prints per-intent F1 scores

Requirements (install if missing):
    py -3.13 -m pip install transformers torch scikit-learn
"""

import os
import json
import random
import joblib
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW

# ── Config ────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent.resolve()
TRAINING_FILE   = BASE_DIR / "training_data.json"
OUTPUT_DIR      = BASE_DIR / "intent_model"
LABEL_ENC_PATH  = BASE_DIR / "label_encoder.pkl"

MODEL_NAME      = "distilbert-base-uncased"
MAX_LEN         = 64
BATCH_SIZE      = 16
EPOCHS          = 10
LEARNING_RATE   = 2e-5
WARMUP_RATIO    = 0.1
TEST_SIZE       = 0.15
SEED            = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n{'='*55}")
print(f"  StayBuddy — DistilBERT Retraining")
print(f"  Device : {DEVICE}")
print(f"{'='*55}\n")


# ══════════════════════════════════════════════════════════════════
# 1. LOAD & AUGMENT DATA
# ══════════════════════════════════════════════════════════════════

def augment(text: str) -> list:
    """
    Simple augmentation: lowercase, add common variants.
    Returns a list of augmented strings (including original).
    """
    variants = [text]
    t = text.lower()
    variants.append(t)

    # Add question mark variant
    if not t.endswith("?"):
        variants.append(t + "?")

    # Add "please" prefix
    variants.append("please " + t)

    # Common abbreviation swaps
    replacements = [
        ("fast nuces", "fast"), ("fast nuces", "university"),
        ("per month", "/mo"), ("rupees", "rs"),
        ("islamabad", "isb"), ("hostel", "pg"),
    ]
    for old, new in replacements:
        if old in t:
            variants.append(t.replace(old, new))

    return list(set(variants))   # deduplicate


def load_data():
    with open(TRAINING_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts, labels = [], []
    for intent_block in data["intents"]:
        intent = intent_block["intent"]
        for example in intent_block["examples"]:
            # Original
            texts.append(example)
            labels.append(intent)
            # Augmented
            for aug in augment(example):
                if aug != example:
                    texts.append(aug)
                    labels.append(intent)

    # Shuffle
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts, labels = zip(*combined)

    print(f"Total examples after augmentation: {len(texts)}")
    from collections import Counter
    counts = Counter(labels)
    print("\nExamples per intent:")
    for intent, count in sorted(counts.items()):
        print(f"  {intent:<30} {count}")

    return list(texts), list(labels)


# ══════════════════════════════════════════════════════════════════
# 2. DATASET CLASS
# ══════════════════════════════════════════════════════════════════

class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ══════════════════════════════════════════════════════════════════
# 3. TRAINING LOOP
# ══════════════════════════════════════════════════════════════════

def train():
    # ── Load data ─────────────────────────────────────────────────
    texts, raw_labels = load_data()

    le = LabelEncoder()
    le.fit(raw_labels)
    encoded_labels = le.transform(raw_labels)
    num_classes    = len(le.classes_)
    print(f"\nClasses ({num_classes}): {list(le.classes_)}")

    # ── Train / test split ────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        texts, encoded_labels,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=encoded_labels,
    )
    print(f"\nTrain: {len(X_train)}  |  Test: {len(X_test)}")

    # ── Tokenizer & model ─────────────────────────────────────────
    print(f"\nLoading DistilBERT tokenizer and model...")
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    model     = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_classes
    )
    model.to(DEVICE)

    # ── DataLoaders ───────────────────────────────────────────────
    train_ds = IntentDataset(X_train, y_train, tokenizer, MAX_LEN)
    test_ds  = IntentDataset(X_test,  y_test,  tokenizer, MAX_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

    # ── Optimizer & scheduler ────────────────────────────────────
    optimizer    = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps  = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler    = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ── Training ─────────────────────────────────────────────────
    print(f"\nStarting training for {EPOCHS} epochs...\n")
    best_acc = 0.0

    for epoch in range(EPOCHS):
        # Train
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels_b       = batch["label"].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels_b,
            )
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_loss = total_loss / len(train_loader)

        # Evaluate
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in test_loader:
                input_ids      = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels_b       = batch["label"].to(DEVICE)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                batch_preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                preds.extend(batch_preds)
                trues.extend(labels_b.cpu().numpy())

        acc = np.mean(np.array(preds) == np.array(trues))
        print(f"Epoch {epoch+1:2d}/{EPOCHS}  |  Loss: {avg_loss:.4f}  |  Accuracy: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            # Save best checkpoint
            OUTPUT_DIR.mkdir(exist_ok=True)
            model.save_pretrained(str(OUTPUT_DIR))
            tokenizer.save_pretrained(str(OUTPUT_DIR))
            joblib.dump(le, str(LABEL_ENC_PATH))
            print(f"             ✅  New best model saved (acc={acc:.4f})")

    # ── Final evaluation ──────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Training complete.  Best accuracy: {best_acc:.4f}")
    print(f"{'='*55}\n")

    # Reload best model for final report
    best_model = DistilBertForSequenceClassification.from_pretrained(str(OUTPUT_DIR))
    best_model.to(DEVICE)
    best_model.eval()

    preds, trues = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels_b       = batch["label"].to(DEVICE)
            outputs        = best_model(input_ids=input_ids, attention_mask=attention_mask)
            batch_preds    = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            preds.extend(batch_preds)
            trues.extend(labels_b.cpu().numpy())

    print("Per-Intent F1 Report:")
    print(classification_report(
        trues, preds,
        target_names=le.classes_,
        digits=4,
    ))

    print(f"\nSaved to:")
    print(f"  Model  →  {OUTPUT_DIR}")
    print(f"  Encoder→  {LABEL_ENC_PATH}\n")


if __name__ == "__main__":
    train()
