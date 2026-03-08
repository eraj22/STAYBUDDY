import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup



import joblib
import os

# ── 1. Load training data ──────────────────────────────
with open("training_data.json") as f:
    data = json.load(f)

texts, labels = [], []
for intent in data["intents"]:
    for ex in intent["examples"]:
        texts.append(ex)
        labels.append(intent["intent"])

df = pd.DataFrame({"text": texts, "intent": labels})
print(f"Total examples: {len(df)}")
print("\nExamples per intent:")
print(df["intent"].value_counts())

# ── 2. Encode labels ───────────────────────────────────
le = LabelEncoder()
df["label"] = le.fit_transform(df["intent"])
num_labels = len(le.classes_)
print(f"\nNumber of intents: {num_labels}")
print(f"Intents: {list(le.classes_)}")

# ── 3. Train/test split ────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    df["text"].tolist(),
    df["label"].tolist(),
    test_size=0.15,
    random_state=42,
    stratify=df["label"].tolist()
)
print(f"\nTraining samples: {len(X_train)}")
print(f"Test samples    : {len(X_test)}")

# ── 4. Tokenizer ───────────────────────────────────────
print("\nLoading DistilBERT tokenizer...")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# ── 5. Dataset class ───────────────────────────────────
class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long)
        }

train_dataset = IntentDataset(X_train, y_train, tokenizer)
test_dataset  = IntentDataset(X_test,  y_test,  tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=16)

# ── 6. Load DistilBERT model ───────────────────────────
print("Loading DistilBERT model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=num_labels
)
model.to(device)

# ── 7. Training setup ──────────────────────────────────
EPOCHS = 20
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=total_steps // 10,
    num_training_steps=total_steps
)

# ── 8. Training loop ───────────────────────────────────
print("\nTraining DistilBERT...")
print("-" * 40)

best_accuracy = 0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch in train_loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels_batch   = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels_batch
        )
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    avg_loss = total_loss / len(train_loader)

    # Evaluate each epoch
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_batch   = batch["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Accuracy: {acc:.2%}")

    # Save best model
    if acc > best_accuracy:
        best_accuracy = acc
        model.save_pretrained("intent_model")
        tokenizer.save_pretrained("intent_model")
        print(f"  ✅ Best model saved (accuracy: {acc:.2%})")

# ── 9. Final evaluation ────────────────────────────────
print("\n" + "=" * 40)
print("FINAL EVALUATION")
print("=" * 40)
print(f"\nBest Accuracy: {best_accuracy:.2%}")
print("\nDetailed Report:")
print(classification_report(
    all_labels, all_preds,
    target_names=le.classes_))

# ── 10. Confusion matrix ───────────────────────────────
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=le.classes_,
            yticklabels=le.classes_,
            cmap="Blues")
plt.title("DistilBERT Intent Classification — Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("\nConfusion matrix saved")

# ── 11. Save label encoder ─────────────────────────────
joblib.dump(le, "label_encoder.pkl")
print("Label encoder saved")
print("\nAll done. Model saved in intent_model/ folder")