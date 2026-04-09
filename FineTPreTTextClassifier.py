import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    DataCollatorWithPadding
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# =========================
# 1. SETUP
# =========================

SEED = 99
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# =========================
# 2. LOAD DATA
# =========================

df = pd.read_csv("recipes_fruit_veg.csv")

df['label'] = 1
df.loc[df['category'] == 'fruit', 'label'] = 0

df = df.dropna(subset=['name'])

texts = df['name'].tolist()
labels = df['label'].tolist()

print(f"Total samples: {len(texts)}")

# =========================
# 3. SPLIT DATA
# =========================

train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=SEED
)

# =========================
# 4. TOKENIZER + MODEL
# =========================

model_name = "distilbert-base-uncased"

tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

model = DistilBertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)

model.to(device)

# =========================
# 5. DATASET CLASS
# =========================

class RecipeDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding=False,
            return_tensors="pt"
        )

        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item

# =========================
# 6. DATALOADERS
# =========================

train_dataset = RecipeDataset(train_texts, train_labels, tokenizer)
val_dataset = RecipeDataset(val_texts, val_labels, tokenizer)

collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=collator
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    collate_fn=collator
)

# =========================
# 7. CLASS WEIGHTS
# =========================

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_labels),
    y=train_labels
)

class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

# =========================
# 8. FREEZE STRATEGY (TOGGLE)
# =========================

FREEZE_BASE = True  # 🔥 change to False for full fine-tune

if FREEZE_BASE:
    print("Freezing base DistilBERT...")
    for param in model.distilbert.parameters():
        param.requires_grad = False

# =========================
# 9. OPTIMIZER
# =========================

trainable_params = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.AdamW(trainable_params, lr=5e-5)

print(f"Trainable parameters: {len(trainable_params)}")

# =========================
# 10. TRAIN FUNCTION
# =========================

def train_epoch(model, loader):
    model.train()
    total_loss = 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss  # HF handles CE internally

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

# =========================
# 11. EVALUATION FUNCTION
# =========================

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=1)

            correct += (preds == batch["labels"]).sum().item()
            total += batch["labels"].size(0)

    return correct / total

# =========================
# 12. TRAIN LOOP
# =========================

epochs = 3

for epoch in range(epochs):
    train_loss = train_epoch(model, train_loader)
    val_acc = evaluate(model, val_loader)

    print(f"\nEpoch {epoch+1}")
    print(f"Loss: {train_loss:.4f}")
    print(f"Val Accuracy: {val_acc:.4f}")

# =========================
# 13. INFERENCE FUNCTION
# =========================

def predict(text):
    model.eval()

    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)

    return probs.cpu().numpy()

# =========================
# 14. TEST
# =========================

print("\n--- Predictions ---")
print("Mango smoothie:", predict("Fresh mango smoothie"))
print("Grilled broccoli:", predict("Grilled broccoli with garlic"))
