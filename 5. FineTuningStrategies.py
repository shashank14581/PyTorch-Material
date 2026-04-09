import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
from torchvision import transforms, models


# =========================================================
# 1. DEVICE
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# =========================================================
# 2. CONFIG
# =========================================================
NUM_CLASSES = 9
BATCH_SIZE = 64
NUM_EPOCHS = 5
NUM_WORKERS = 2
LEARNING_RATE_HEAD = 1e-3
LEARNING_RATE_FINE = 1e-5
LEARNING_RATE_FULL = 1e-4

subset_target_classes = [
    # Flowers
    "orchid", "poppy", "sunflower",
    # Mammals
    "fox", "raccoon", "skunk",
    # Insects
    "butterfly", "caterpillar", "cockroach"
]


# =========================================================
# 3. TRANSFORMS
# =========================================================
# Since we are using a pretrained ImageNet model, we resize to 224x224
# and normalize with ImageNet stats.

imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std = (0.229, 0.224, 0.225)

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])


# =========================================================
# 4. LOAD CIFAR100
# =========================================================
train_dataset_full = torchvision.datasets.CIFAR100(
    root="./cifar100",
    train=True,
    download=True,
    transform=train_transform
)

val_dataset_full = torchvision.datasets.CIFAR100(
    root="./cifar100",
    train=False,
    download=True,
    transform=val_transform
)


# =========================================================
# 5. FILTER TO 9 TARGET CLASSES
# =========================================================
class_to_id = {cls_name: idx for idx, cls_name in enumerate(train_dataset_full.classes)}
target_indices = sorted([class_to_id[cls_name] for cls_name in subset_target_classes])

print("Selected classes:")
for cls_name in subset_target_classes:
    print(f"  {cls_name} -> original label {class_to_id[cls_name]}")

# Remap original CIFAR100 labels to 0..8
new_label_map = {old_label: new_label for new_label, old_label in enumerate(target_indices)}

print("\nNew label map:")
for old, new in new_label_map.items():
    print(f"  original {old} -> new {new}")


def filter_dataset(dataset, allowed_labels):
    indices = []
    for i, (_, label) in enumerate(dataset):
        if label in allowed_labels:
            indices.append(i)
    return Subset(dataset, indices)


class RemappedDataset(Dataset):
    def __init__(self, subset, label_map):
        self.subset = subset
        self.label_map = label_map

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, old_label = self.subset[idx]
        new_label = self.label_map[old_label]
        return image, new_label


train_subset = filter_dataset(train_dataset_full, set(target_indices))
val_subset = filter_dataset(val_dataset_full, set(target_indices))

train_dataset = RemappedDataset(train_subset, new_label_map)
val_dataset = RemappedDataset(val_subset, new_label_map)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

print("\nDataset sizes:")
print("Train:", len(train_dataset))
print("Val:", len(val_dataset))


# =========================================================
# 6. MODEL SETUP
# =========================================================
def build_resnet18(num_classes=9):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


# =========================================================
# 7. FREEZE / UNFREEZE STRATEGIES
# =========================================================
def freeze_all(model):
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_head_only(model):
    # fc is already newly attached
    for param in model.fc.parameters():
        param.requires_grad = True


def unfreeze_layer4_and_head(model):
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True


def unfreeze_layer3_layer4_and_head(model):
    for param in model.layer3.parameters():
        param.requires_grad = True
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True


def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True


def count_trainable_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# =========================================================
# 8. TRAIN / EVAL FUNCTIONS
# =========================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / len(loader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / len(loader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def fit_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    num_epochs,
    strategy_name="strategy"
):
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    start_time = time.time()

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        print(
            f"[{strategy_name}] "
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

    elapsed = time.time() - start_time
    print(f"[{strategy_name}] Best Val Acc: {best_val_acc:.4f}")
    print(f"[{strategy_name}] Time: {elapsed:.2f} sec")

    model.load_state_dict(best_model_wts)
    return model, history


# =========================================================
# 9. OPTIMIZER FACTORY
# =========================================================
def make_optimizer(model, lr):
    return optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )


# =========================================================
# 10. STRATEGY RUNNERS
# =========================================================
def run_feature_extraction():
    """
    Strategy 1:
    Freeze entire backbone, train only classifier head.
    """
    model = build_resnet18(NUM_CLASSES)
    freeze_all(model)
    unfreeze_head_only(model)
    model = model.to(device)

    total, trainable = count_trainable_parameters(model)
    print(f"\n[Feature Extraction] total params = {total:,}, trainable = {trainable:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(model, LEARNING_RATE_HEAD)

    model, history = fit_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=NUM_EPOCHS,
        strategy_name="Feature Extraction"
    )
    return model, history


def run_partial_fine_tuning():
    """
    Strategy 2:
    Freeze backbone, unfreeze layer4 + head.
    """
    model = build_resnet18(NUM_CLASSES)
    freeze_all(model)
    unfreeze_layer4_and_head(model)
    model = model.to(device)

    total, trainable = count_trainable_parameters(model)
    print(f"\n[Partial Fine-Tuning] total params = {total:,}, trainable = {trainable:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(model, LEARNING_RATE_FINE)

    model, history = fit_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=NUM_EPOCHS,
        strategy_name="Partial Fine-Tuning"
    )
    return model, history


def run_deeper_fine_tuning():
    """
    Strategy 3:
    Freeze lower backbone, unfreeze layer3 + layer4 + head.
    """
    model = build_resnet18(NUM_CLASSES)
    freeze_all(model)
    unfreeze_layer3_layer4_and_head(model)
    model = model.to(device)

    total, trainable = count_trainable_parameters(model)
    print(f"\n[Deeper Fine-Tuning] total params = {total:,}, trainable = {trainable:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(model, LEARNING_RATE_FINE)

    model, history = fit_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=NUM_EPOCHS,
        strategy_name="Deeper Fine-Tuning"
    )
    return model, history


def run_full_fine_tuning():
    """
    Strategy 4:
    Train entire model.
    """
    model = build_resnet18(NUM_CLASSES)
    unfreeze_all(model)
    model = model.to(device)

    total, trainable = count_trainable_parameters(model)
    print(f"\n[Full Fine-Tuning] total params = {total:,}, trainable = {trainable:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(model, LEARNING_RATE_FULL)

    model, history = fit_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=NUM_EPOCHS,
        strategy_name="Full Fine-Tuning"
    )
    return model, history


def run_two_stage_fine_tuning():
    """
    Optional educational strategy:
    Stage 1: train head only
    Stage 2: unfreeze layer4 and continue training
    """
    model = build_resnet18(NUM_CLASSES)

    # ---------- Stage 1 ----------
    freeze_all(model)
    unfreeze_head_only(model)
    model = model.to(device)

    total, trainable = count_trainable_parameters(model)
    print(f"\n[Two-Stage | Stage 1] total params = {total:,}, trainable = {trainable:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(model, LEARNING_RATE_HEAD)

    model, history_stage1 = fit_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=3,
        strategy_name="Two-Stage Stage-1 Head Only"
    )

    # ---------- Stage 2 ----------
    unfreeze_layer4_and_head(model)

    total, trainable = count_trainable_parameters(model)
    print(f"\n[Two-Stage | Stage 2] total params = {total:,}, trainable = {trainable:,}")

    optimizer = make_optimizer(model, LEARNING_RATE_FINE)

    model, history_stage2 = fit_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=3,
        strategy_name="Two-Stage Stage-2 Fine-Tune layer4"
    )

    return model, {
        "stage1": history_stage1,
        "stage2": history_stage2
    }


# =========================================================
# 11. MAIN EXECUTION
# =========================================================
if __name__ == "__main__":
    print("\n==============================")
    print("TRANSFER LEARNING STRATEGIES")
    print("==============================")

    # 1. Feature Extraction
    feature_model, feature_history = run_feature_extraction()

    # 2. Partial Fine-Tuning
    partial_model, partial_history = run_partial_fine_tuning()

    # 3. Deeper Fine-Tuning
    deeper_model, deeper_history = run_deeper_fine_tuning()

    # 4. Full Fine-Tuning
    full_model, full_history = run_full_fine_tuning()

    # 5. Two-Stage Fine-Tuning
    two_stage_model, two_stage_history = run_two_stage_fine_tuning()
