# =========================================================
# 1. IMPORTS
# =========================================================
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
from torchvision import transforms

# =========================================================
# 2. DEVICE
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# =========================================================
# 3. CONFIG
# =========================================================
NUM_CLASSES = 9
BATCH_SIZE = 64
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
NUM_WORKERS = 2

CLASSIFICATION_LOSS_WEIGHT = 1.0
REGRESSION_LOSS_WEIGHT = 1.0

subset_target_classes = [
    "orchid", "poppy", "sunflower",
    "fox", "raccoon", "skunk",
    "butterfly", "caterpillar", "cockroach"
]


# =========================================================
# 4. TRANSFORMS
# =========================================================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# =========================================================
# 5. LOAD DATA
# =========================================================
train_dataset_full = torchvision.datasets.CIFAR100(
    root="./data", train=True, download=True, transform=train_transform
)

val_dataset_full = torchvision.datasets.CIFAR100(
    root="./data", train=False, download=True, transform=val_transform
)


# =========================================================
# 6. FILTER CLASSES
# =========================================================
class_to_id = {cls: i for i, cls in enumerate(train_dataset_full.classes)}
target_indices = sorted([class_to_id[c] for c in subset_target_classes])

new_label_map = {old: new for new, old in enumerate(target_indices)}


def filter_dataset(dataset, allowed_labels):
    indices = []
    for i, (_, label) in enumerate(dataset):
        if label in allowed_labels:
            indices.append(i)
    return Subset(dataset, indices)


train_subset = filter_dataset(train_dataset_full, set(target_indices))
val_subset = filter_dataset(val_dataset_full, set(target_indices))


# =========================================================
# 7. REMAPPED MULTI-TASK DATASET
# =========================================================
class MultiTaskDataset(Dataset):
    def __init__(self, subset, label_map):
        self.subset = subset
        self.label_map = label_map

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, original_label = self.subset[idx]

        class_label = self.label_map[original_label]

        regression_target = torch.tensor(
            [original_label / 99.0], dtype=torch.float32
        )

        return x, class_label, regression_target


train_dataset = MultiTaskDataset(train_subset, new_label_map)
val_dataset = MultiTaskDataset(val_subset, new_label_map)


train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE,
    shuffle=True, num_workers=NUM_WORKERS
)

val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=NUM_WORKERS
)


# =========================================================
# 8. CUSTOM CNN MODEL
# =========================================================
class MultiTaskCNN(nn.Module):
    def __init__(self, num_classes=9, dropout_p=0.3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.dropout = nn.Dropout(dropout_p)

        self.classification_head = nn.Linear(256, num_classes)
        self.regression_head = nn.Linear(256, 1)

    def forward(self, x, return_features=False, use_dropout_head=True):
        features = self.features(x)
        features = torch.flatten(features, 1)

        if use_dropout_head:
            features = self.dropout(features)

        class_logits = self.classification_head(features)
        regression_output = self.regression_head(features)

        if return_features:
            return {
                "features": features,
                "class_logits": class_logits,
                "regression_output": regression_output
            }

        return class_logits, regression_output


# =========================================================
# 9. LOSS FUNCTION
# =========================================================
def compute_loss(cls_logits, reg_out, cls_target, reg_target):
    cls_loss = nn.CrossEntropyLoss()(cls_logits, cls_target)
    reg_loss = nn.MSELoss()(reg_out, reg_target)

    total = (
        CLASSIFICATION_LOSS_WEIGHT * cls_loss +
        REGRESSION_LOSS_WEIGHT * reg_loss
    )

    return total, cls_loss, reg_loss


# =========================================================
# 10. TRAIN
# =========================================================
def train_one_epoch(model, loader, optimizer):
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for x, y_cls, y_reg in loader:
        x = x.to(device)
        y_cls = y_cls.to(device)
        y_reg = y_reg.to(device)

        optimizer.zero_grad()

        cls_logits, reg_out = model(x)

        loss, _, _ = compute_loss(cls_logits, reg_out, y_cls, y_reg)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = cls_logits.argmax(1)
        correct += (preds == y_cls).sum().item()
        total += y_cls.size(0)

    return total_loss / len(loader), correct / total


# =========================================================
# 11. VALIDATE
# =========================================================
def evaluate(model, loader):
    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y_cls, y_reg in loader:
            x = x.to(device)
            y_cls = y_cls.to(device)
            y_reg = y_reg.to(device)

            cls_logits, reg_out = model(x)

            loss, _, _ = compute_loss(cls_logits, reg_out, y_cls, y_reg)

            total_loss += loss.item()

            preds = cls_logits.argmax(1)
            correct += (preds == y_cls).sum().item()
            total += y_cls.size(0)

    return total_loss / len(loader), correct / total


# =========================================================
# 12. TRAIN LOOP
# =========================================================
def fit(model):
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_wts = copy.deepcopy(model.state_dict())
    best_acc = 0

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer)
        val_loss, val_acc = evaluate(model, val_loader)

        if val_acc > best_acc:
            best_acc = val_acc
            best_wts = copy.deepcopy(model.state_dict())

        print(f"Epoch {epoch+1}")
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        print("-" * 40)

    model.load_state_dict(best_wts)
    return model


# =========================================================
# 13. DYNAMIC GRAPH DEMO
# =========================================================
def demo_dynamic(model):
    model.eval()

    x, _, _ = next(iter(val_loader))
    x = x[:4].to(device)

    with torch.no_grad():
        out = model(x)
        print("\nNormal Output Shapes:", out[0].shape, out[1].shape)

        out2 = model(x, return_features=True, use_dropout_head=False)
        print("Feature Shape:", out2["features"].shape)


# =========================================================
# 14. MAIN
# =========================================================
if __name__ == "__main__":
    print("\n=== Custom CNN Multi-task Training ===")

    model = MultiTaskCNN(NUM_CLASSES).to(device)

    model = fit(model)

    demo_dynamic(model)
