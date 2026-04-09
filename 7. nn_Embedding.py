import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

import pandas as pd
import numpy as np
from collections import Counter
import re

# ---------------- DEVICE ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- LOAD DATA ----------------
df = pd.read_csv("recipes_fruit_veg.csv")
df = df.dropna(subset=["name"])

df["label"] = 1
df.loc[df["category"] == "fruit", "label"] = 0

texts = df["name"].tolist()
labels = df["label"].tolist()

# ---------------- SPLIT ----------------
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, stratify=labels
)

# ---------------- PREPROCESS ----------------
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.split()

train_tokens = [preprocess(t) for t in train_texts]
val_tokens = [preprocess(t) for t in val_texts]

# ---------------- VOCAB ----------------
min_freq = 2
word_counts = Counter(word for sent in train_tokens for word in sent)

word2idx = {"<pad>": 0, "<unk>": 1}

for word, count in word_counts.items():
    if count >= min_freq:
        word2idx[word] = len(word2idx)

# ---------------- ENCODE ----------------
def encode(tokens):
    return [word2idx.get(w, word2idx["<unk>"]) for w in tokens]

train_encoded = [encode(t) for t in train_tokens]
val_encoded = [encode(t) for t in val_tokens]

# ---------------- DATASET ----------------
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            "text": torch.tensor(self.texts[idx], dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }

train_dataset = TextDataset(train_encoded, train_labels)
val_dataset = TextDataset(val_encoded, val_labels)

# ---------------- COLLATE ----------------
def collate_fn(batch):
    texts = [item["text"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch])

    max_len = max(len(t) for t in texts)

    padded = torch.zeros(len(texts), max_len, dtype=torch.long)

    for i, t in enumerate(texts):
        padded[i, :len(t)] = t

    return padded.to(device), labels.to(device)

# ---------------- DATALOADER ----------------
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# ---------------- MODEL ----------------
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        emb = self.embedding(x)

        mask = (x != 0).unsqueeze(-1)
        emb = emb * mask

        pooled = emb.sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        return self.fc(pooled)

model = TextClassifier(len(word2idx), 64, 2).to(device)

# ---------------- LOSS ----------------
weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_labels),
    y=train_labels
)

weights = torch.tensor(weights, dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)

# ---------------- OPTIMIZER ----------------
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ---------------- TRAIN ----------------
epochs = 5

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for x, y in train_loader:
        optimizer.zero_grad()

        output = model(x)
        loss = criterion(output, y)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}")

# ---------------- EVAL ----------------
model.eval()

preds, actuals = [], []

with torch.no_grad():
    for x, y in val_loader:
        output = model(x)
        pred = torch.argmax(output, dim=1)

        preds.extend(pred.cpu().numpy())
        actuals.extend(y.cpu().numpy())

print("Accuracy:", accuracy_score(actuals, preds))
print("Precision:", precision_score(actuals, preds))
print("Recall:", recall_score(actuals, preds))
print("F1:", f1_score(actuals, preds))

# ---------------- PREDICT ----------------
def predict(text):
    tokens = preprocess(text)
    encoded = encode(tokens)

    tensor = torch.tensor(encoded).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        pred = torch.argmax(output, dim=1).item()

    return "vegetable" if pred == 1 else "fruit"

# ---------------- TEST ----------------
print(predict("Blueberry muffins"))
print(predict("Spinach curry"))
print(predict("Avocado toast"))
