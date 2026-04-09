import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from collections import Counter
import pandas as pd
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- DATA ----------------

data = pd.read_csv("recipe.txt")
df = data.dropna(subset=["name"]).copy()

df["label"] = 1
df.loc[df["category"] == "fruit", "label"] = 0

texts = df["name"].tolist()
labels = df["label"].tolist()

train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, stratify=labels, random_state=42
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

for w, c in word_counts.items():
    if c >= min_freq:
        word2idx[w] = len(word2idx)

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
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }

train_set = TextDataset(train_encoded, train_labels)
val_set = TextDataset(val_encoded, val_labels)

# ---------------- COLLATE (EmbeddingBag) ----------------

def collate_fn(batch):
    texts = []
    labels = []
    offsets = [0]

    for item in batch:
        tokens = item["text"]
        texts.append(tokens)
        labels.append(item["label"])
        offsets.append(tokens.size(0))

    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text = torch.cat(texts)
    labels = torch.tensor(labels, dtype=torch.long)

    return text.to(device), offsets.to(device), labels.to(device)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False, collate_fn=collate_fn)

# ---------------- MODEL ----------------

class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes=2):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, mode="mean")
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, text, offsets):
        x = self.embedding(text, offsets)
        return self.fc(x)

model = TextClassifier(len(word2idx), 64, 2).to(device)

# ---------------- TRAIN ----------------

optimizer = torch.optim.Adam(model.parameters(), lr=4e-4)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    model.train()
    running_loss = 0.0

    for text, offsets, label in train_loader:
        optimizer.zero_grad()

        output = model(text, offsets)
        loss = criterion(output, label)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader):.4f}")

# ---------------- EVAL ----------------

model.eval()
total_loss, total_acc = 0.0, 0.0

with torch.no_grad():
    for text, offsets, label in val_loader:
        output = model(text, offsets)
        loss = criterion(output, label)

        preds = torch.argmax(output, dim=1)
        acc = (preds == label).sum().item() / label.size(0)

        total_loss += loss.item()
        total_acc += acc

print(f"Val Loss: {total_loss / len(val_loader):.4f}, Val Acc: {total_acc / len(val_loader):.4f}")
