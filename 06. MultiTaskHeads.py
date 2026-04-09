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

CLASSIFICATION_LOSS_WEIGHT = 1.0
REGRESSION_LOSS_WEIGHT = 1.0

LEARNING_RATE = 1e-4

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

print("\nSelected classes:")
for cls_name in subset_target_classes:
    print(f"{cls_name:15s} -> original label {class_to_id[cls_name]}")

# remap original CIFAR labels to 0..8 for classification task
new_label_map = {old_label: new_label for new_label, old_label in enumerate(target_indices)}

print("\nNew classification label map:")
for old_label, new_label in new_label_map.items():
    print(f"original {old_label} -> class head label {new_label}")


def filter_dataset(dataset, allowed_labels):
    indices = []
    for i, (_, label) in enumerate(dataset):
        if label in allowed_labels:
            indices.append(i)
    return Subset(dataset, indices)


class MultiTaskRemappedDataset(Dataset):
    """
    Returns:
        image
        class_label: remapped 0..8
        regression_target: normalized original CIFAR100 label in [0,1]
    """
    def __init__(self, subset, label_map):
        self.subset = subset
        self.label_map = label_map

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, original_label = self.subset[idx]

        class_label = self.label_map[original_label]

        # synthetic regression target for learning pipeline demonstration
        regression_target = torch.tensor(
            [original_label / 99.0], dtype=torch.float32
        )

        return image, class_label, regression_target


train_subset = filter_dataset(train_dataset_full, set(target_indices))
val_subset = filter_dataset(val_dataset_full, set(target_indices))

train_dataset = MultiTaskRemappedDataset(train_subset, new_label_map)
val_dataset = MultiTaskRemappedDataset(val_subset, new_label_map)

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
# 6. MULTI-TASK MODEL WITH DYNAMIC GRAPH
# =========================================================
class MultiTaskResNet18(nn.Module):
    """
    Shared backbone + two heads:
    - classification head
    - regression head

    Dynamic graph aspect:
    - Python control flow inside forward
    - optional return_heads
    - optional use_dropout_head
    """
    def __init__(self, num_classes=9, use_pretrained=True, dropout_p=0.3):
        super().__init__()

        weights = models.ResNet18_Weights.IMAGENET1K_V1 if use_pretrained else None
        backbone = models.resnet18(weights=weights)

        # remove final fc, keep everything before it
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        self.feature_dim = backbone.fc.in_features

        self.dropout = nn.Dropout(dropout_p)

        # classification head
        self.classification_head = nn.Linear(self.feature_dim, num_classes)

        # regression head
        self.regression_head = nn.Linear(self.feature_dim, 1)

    def forward(self, x, return_features=False, use_dropout_head=True):
        """
        Dynamic graph behavior:
        - return_features can change output structure
        - use_dropout_head can change path through graph
        """

        features = self.feature_extractor(x)
        features = torch.flatten(features, 1)

        if use_dropout_head:
            head_input = self.dropout(features)
        else:
            head_input = features

        class_logits = self.classification_head(head_input)
        regression_output = self.regression_head(head_input)

        if return_features:
            return {
                "features": features,
                "class_logits": class_logits,
                "regression_output": regression_output
            }

        return class_logits, regression_output


# =========================================================
# 7. FREEZE / UNFREEZE STRATEGIES
# =========================================================
def freeze_all_feature_extractor(model):
    for param in model.feature_extractor.parameters():
        param.requires_grad = False


def unfreeze_last_resnet_block(model):
    """
    feature_extractor corresponds to:
    conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool

    In ResNet18:
    index mapping inside feature_extractor:
      0 conv1
      1 bn1
      2 relu
      3 maxpool
      4 layer1
      5 layer2
      6 layer3
      7 layer4
      8 avgpool
    """
    for param in model.feature_extractor[7].parameters():
        param.requires_grad = True


def unfreeze_last_two_resnet_blocks(model):
    for param in model.feature_extractor[6].parameters():
        param.requires_grad = True
    for param in model.feature_extractor[7].parameters():
        param.requires_grad = True


def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# =========================================================
# 8. TRAIN / EVAL UTILITIES
# =========================================================
def compute_multitask_loss(
    class_logits,
    regression_output,
    class_targets,
    regression_targets,
    classification_criterion,
    regression_criterion,
    classification_weight=1.0,
    regression_weight=1.0
):
    classification_loss = classification_criterion(class_logits, class_targets)
    regression_loss = regression_criterion(regression_output, regression_targets)

    total_loss = (
        classification_weight * classification_loss
        + regression_weight * regression_loss
    )

    return total_loss, classification_loss, regression_loss


def train_one_epoch(
    model,
    loader,
    optimizer,
    classification_criterion,
    regression_criterion,
    device,
    classification_weight=1.0,
    regression_weight=1.0
):
    model.train()

    running_total_loss = 0.0
    running_classification_loss = 0.0
    running_regression_loss = 0.0

    correct = 0
    total = 0

    regression_mae_sum = 0.0
    regression_count = 0

    for images, class_labels, regression_targets in loader:
        images = images.to(device)
        class_labels = class_labels.to(device)
        regression_targets = regression_targets.to(device)

        optimizer.zero_grad()

        class_logits, regression_output = model(images)

        total_loss, classification_loss, regression_loss = compute_multitask_loss(
            class_logits=class_logits,
            regression_output=regression_output,
            class_targets=class_labels,
            regression_targets=regression_targets,
            classification_criterion=classification_criterion,
            regression_criterion=regression_criterion,
            classification_weight=classification_weight,
            regression_weight=regression_weight
        )

        total_loss.backward()
        optimizer.step()

        running_total_loss += total_loss.item()
        running_classification_loss += classification_loss.item()
        running_regression_loss += regression_loss.item()

        preds = class_logits.argmax(dim=1)
        correct += (preds == class_labels).sum().item()
        total += class_labels.size(0)

        regression_mae_sum += torch.abs(regression_output - regression_targets).sum().item()
        regression_count += regression_targets.numel()

    metrics = {
        "total_loss": running_total_loss / len(loader),
        "classification_loss": running_classification_loss / len(loader),
        "regression_loss": running_regression_loss / len(loader),
        "classification_acc": correct / total,
        "regression_mae": regression_mae_sum / regression_count
    }

    return metrics


def evaluate(
    model,
    loader,
    classification_criterion,
    regression_criterion,
    device,
    classification_weight=1.0,
    regression_weight=1.0
):
    model.eval()

    running_total_loss = 0.0
    running_classification_loss = 0.0
    running_regression_loss = 0.0

    correct = 0
    total = 0

    regression_mae_sum = 0.0
    regression_count = 0

    with torch.no_grad():
        for images, class_labels, regression_targets in loader:
            images = images.to(device)
            class_labels = class_labels.to(device)
            regression_targets = regression_targets.to(device)

            class_logits, regression_output = model(images)

            total_loss, classification_loss, regression_loss = compute_multitask_loss(
                class_logits=class_logits,
                regression_output=regression_output,
                class_targets=class_labels,
                regression_targets=regression_targets,
                classification_criterion=classification_criterion,
                regression_criterion=regression_criterion,
                classification_weight=classification_weight,
                regression_weight=regression_weight
            )

            running_total_loss += total_loss.item()
            running_classification_loss += classification_loss.item()
            running_regression_loss += regression_loss.item()

            preds = class_logits.argmax(dim=1)
            correct += (preds == class_labels).sum().item()
            total += class_labels.size(0)

            regression_mae_sum += torch.abs(regression_output - regression_targets).sum().item()
            regression_count += regression_targets.numel()

    metrics = {
        "total_loss": running_total_loss / len(loader),
        "classification_loss": running_classification_loss / len(loader),
        "regression_loss": running_regression_loss / len(loader),
        "classification_acc": correct / total,
        "regression_mae": regression_mae_sum / regression_count
    }

    return metrics


def fit_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    classification_criterion,
    regression_criterion,
    device,
    num_epochs,
    strategy_name,
    classification_weight=1.0,
    regression_weight=1.0
):
    history = {
        "train_total_loss": [],
        "train_classification_loss": [],
        "train_regression_loss": [],
        "train_classification_acc": [],
        "train_regression_mae": [],
        "val_total_loss": [],
        "val_classification_loss": [],
        "val_regression_loss": [],
        "val_classification_acc": [],
        "val_regression_mae": []
    }

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_score = -1.0  # based on classification acc minus small regression penalty

    start_time = time.time()

    for epoch in range(num_epochs):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            classification_criterion=classification_criterion,
            regression_criterion=regression_criterion,
            device=device,
            classification_weight=classification_weight,
            regression_weight=regression_weight
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            classification_criterion=classification_criterion,
            regression_criterion=regression_criterion,
            device=device,
            classification_weight=classification_weight,
            regression_weight=regression_weight
        )

        history["train_total_loss"].append(train_metrics["total_loss"])
        history["train_classification_loss"].append(train_metrics["classification_loss"])
        history["train_regression_loss"].append(train_metrics["regression_loss"])
        history["train_classification_acc"].append(train_metrics["classification_acc"])
        history["train_regression_mae"].append(train_metrics["regression_mae"])

        history["val_total_loss"].append(val_metrics["total_loss"])
        history["val_classification_loss"].append(val_metrics["classification_loss"])
        history["val_regression_loss"].append(val_metrics["regression_loss"])
        history["val_classification_acc"].append(val_metrics["classification_acc"])
        history["val_regression_mae"].append(val_metrics["regression_mae"])

        # Simple combined selection score
        selection_score = val_metrics["classification_acc"] - 0.05 * val_metrics["regression_mae"]

        if selection_score > best_val_score:
            best_val_score = selection_score
            best_model_wts = copy.deepcopy(model.state_dict())

        print(
            f"[{strategy_name}] Epoch {epoch+1}/{num_epochs} | "
            f"Train Total: {train_metrics['total_loss']:.4f} | "
            f"Train Cls: {train_metrics['classification_loss']:.4f} | "
            f"Train Reg: {train_metrics['regression_loss']:.4f} | "
            f"Train Acc: {train_metrics['classification_acc']:.4f} | "
            f"Train MAE: {train_metrics['regression_mae']:.4f} || "
            f"Val Total: {val_metrics['total_loss']:.4f} | "
            f"Val Cls: {val_metrics['classification_loss']:.4f} | "
            f"Val Reg: {val_metrics['regression_loss']:.4f} | "
            f"Val Acc: {val_metrics['classification_acc']:.4f} | "
            f"Val MAE: {val_metrics['regression_mae']:.4f}"
        )

    elapsed = time.time() - start_time
    print(f"[{strategy_name}] Training time: {elapsed:.2f} sec")

    model.load_state_dict(best_model_wts)
    return model, history


def make_optimizer(model, lr):
    return optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )


# =========================================================
# 9. STRATEGY RUNNERS
# =========================================================
def build_model_for_multitask():
    return MultiTaskResNet18(
        num_classes=NUM_CLASSES,
        use_pretrained=True,
        dropout_p=0.3
    )


def run_feature_extraction_multitask():
    """
    Freeze shared feature extractor, train both heads only.
    """
    model = build_model_for_multitask()

    freeze_all_feature_extractor(model)

    total, trainable = count_parameters(model)
    print(f"\n[Multi-task Feature Extraction] total params = {total:,}, trainable = {trainable:,}")

    model = model.to(device)

    classification_criterion = nn.CrossEntropyLoss()
    regression_criterion = nn.MSELoss()

    optimizer = make_optimizer(model, LEARNING_RATE)

    model, history = fit_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        classification_criterion=classification_criterion,
        regression_criterion=regression_criterion,
        device=device,
        num_epochs=NUM_EPOCHS,
        strategy_name="Multi-task Feature Extraction",
        classification_weight=CLASSIFICATION_LOSS_WEIGHT,
        regression_weight=REGRESSION_LOSS_WEIGHT
    )

    return model, history


def run_partial_fine_tuning_multitask():
    """
    Freeze most backbone, unfreeze last residual block + both heads.
    """
    model = build_model_for_multitask()

    freeze_all_feature_extractor(model)
    unfreeze_last_resnet_block(model)

    total, trainable = count_parameters(model)
    print(f"\n[Multi-task Partial Fine-Tuning] total params = {total:,}, trainable = {trainable:,}")

    model = model.to(device)

    classification_criterion = nn.CrossEntropyLoss()
    regression_criterion = nn.MSELoss()

    optimizer = make_optimizer(model, LEARNING_RATE)

    model, history = fit_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        classification_criterion=classification_criterion,
        regression_criterion=regression_criterion,
        device=device,
        num_epochs=NUM_EPOCHS,
        strategy_name="Multi-task Partial Fine-Tuning",
        classification_weight=CLASSIFICATION_LOSS_WEIGHT,
        regression_weight=REGRESSION_LOSS_WEIGHT
    )

    return model, history


def run_deeper_fine_tuning_multitask():
    """
    Freeze early backbone, unfreeze last two residual blocks + both heads.
    """
    model = build_model_for_multitask()

    freeze_all_feature_extractor(model)
    unfreeze_last_two_resnet_blocks(model)

    total, trainable = count_parameters(model)
    print(f"\n[Multi-task Deeper Fine-Tuning] total params = {total:,}, trainable = {trainable:,}")

    model = model.to(device)

    classification_criterion = nn.CrossEntropyLoss()
    regression_criterion = nn.MSELoss()

    optimizer = make_optimizer(model, LEARNING_RATE)

    model, history = fit_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        classification_criterion=classification_criterion,
        regression_criterion=regression_criterion,
        device=device,
        num_epochs=NUM_EPOCHS,
        strategy_name="Multi-task Deeper Fine-Tuning",
        classification_weight=CLASSIFICATION_LOSS_WEIGHT,
        regression_weight=REGRESSION_LOSS_WEIGHT
    )

    return model, history


def run_full_fine_tuning_multitask():
    """
    Train everything.
    """
    model = build_model_for_multitask()

    unfreeze_all(model)

    total, trainable = count_parameters(model)
    print(f"\n[Multi-task Full Fine-Tuning] total params = {total:,}, trainable = {trainable:,}")

    model = model.to(device)

    classification_criterion = nn.CrossEntropyLoss()
    regression_criterion = nn.MSELoss()

    optimizer = make_optimizer(model, LEARNING_RATE)

    model, history = fit_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        classification_criterion=classification_criterion,
        regression_criterion=regression_criterion,
        device=device,
        num_epochs=NUM_EPOCHS,
        strategy_name="Multi-task Full Fine-Tuning",
        classification_weight=CLASSIFICATION_LOSS_WEIGHT,
        regression_weight=REGRESSION_LOSS_WEIGHT
    )

    return model, history


# =========================================================
# 10. DEMO OF DYNAMIC GRAPH OUTPUT MODES
# =========================================================
def demo_dynamic_graph_behavior(model, loader, device):
    model.eval()

    images, class_labels, regression_targets = next(iter(loader))
    images = images[:4].to(device)

    with torch.no_grad():
        # normal forward
        class_logits, regression_output = model(images)

        print("\nDynamic Graph Demo:")
        print("Normal forward:")
        print("class_logits shape:", class_logits.shape)
        print("regression_output shape:", regression_output.shape)

        # return features too
        outputs = model(images, return_features=True, use_dropout_head=False)

        print("\nForward with return_features=True:")
        print("features shape:", outputs["features"].shape)
        print("class_logits shape:", outputs["class_logits"].shape)
        print("regression_output shape:", outputs["regression_output"].shape)


# =========================================================
# 11. MAIN
# =========================================================
if __name__ == "__main__":
    print("\n==============================================")
    print("MULTI-TASK LEARNING WITH DYNAMIC GRAPHS")
    print("Shared backbone + classification + regression")
    print("==============================================")

    multitask_feature_model, multitask_feature_history = run_feature_extraction_multitask()
    demo_dynamic_graph_behavior(multitask_feature_model, val_loader, device)

    multitask_partial_model, multitask_partial_history = run_partial_fine_tuning_multitask()

    multitask_deeper_model, multitask_deeper_history = run_deeper_fine_tuning_multitask()

    multitask_full_model, multitask_full_history = run_full_fine_tuning_multitask()
