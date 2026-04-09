# =========================================
# IMPORTS
# =========================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
from torchvision import transforms, models
import optuna

# =========================================
# DEVICE
# =========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================
# TARGET CLASSES
# =========================================
subset_target_classes = [
    'orchid', 'poppy', 'sunflower',
    'fox', 'raccoon', 'skunk',
    'butterfly', 'caterpillar', 'cockroach'
]

# =========================================
# TRANSFORMS (RESNET REQUIRES 224x224)
# =========================================
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(cifar100_mean, cifar100_std)
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(cifar100_mean, cifar100_std)
])

# =========================================
# LOAD DATA
# =========================================
train_dataset_full = torchvision.datasets.CIFAR100(
    root="./data", train=True, download=True, transform=train_transform
)

val_dataset_full = torchvision.datasets.CIFAR100(
    root="./data", train=False, download=True, transform=val_transform
)

# =========================================
# FILTER DATASET
# =========================================
class_to_idx = {cls: i for i, cls in enumerate(train_dataset_full.classes)}
target_indices = {class_to_idx[c] for c in subset_target_classes}

def filter_dataset(dataset, target_indices):
    indices = []
    for i, (_, label) in enumerate(dataset):
        if label in target_indices:
            indices.append(i)
    return Subset(dataset, indices)

train_subset = filter_dataset(train_dataset_full, target_indices)
val_subset = filter_dataset(val_dataset_full, target_indices)

# =========================================
# REMAP LABELS
# =========================================
label_map = {old: new for new, old in enumerate(target_indices)}

class RemappedDataset(Dataset):
    def __init__(self, subset, label_map):
        self.subset = subset
        self.label_map = label_map

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        return image, self.label_map[label]

train_dataset = RemappedDataset(train_subset, label_map)
val_dataset = RemappedDataset(val_subset, label_map)

NUM_CLASSES = len(subset_target_classes)

# =========================================
# MODEL BUILDER (TRANSFER LEARNING)
# =========================================
def build_model(trial):
    # Load pretrained model
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Replace classifier
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    # 🔥 Freeze strategy
    freeze_ratio = trial.suggest_float("freeze_ratio", 0.5, 1.0)

    total_layers = list(model.parameters())
    freeze_until = int(len(total_layers) * freeze_ratio)

    for i, param in enumerate(total_layers):
        param.requires_grad = False if i < freeze_until else True

    return model.to(device)

# =========================================
# TRAIN FUNCTION
# =========================================
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

# =========================================
# VALIDATION FUNCTION
# =========================================
def validate(model, loader, criterion):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

    return total_loss / len(loader)

# =========================================
# OBJECTIVE FUNCTION (OPTUNA CORE)
# =========================================
def objective(trial):

    # Hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = build_model(trial)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )

    criterion = nn.CrossEntropyLoss()

    num_epochs = 3

    # Training loop
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, criterion)
        val_loss = validate(model, val_loader, criterion)

        # 🔥 Pruning signal
        trial.report(val_loss, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_loss

# =========================================
# STUDY CREATION
# =========================================
study = optuna.create_study(
    direction="minimize",
    pruner=optuna.pruners.HyperbandPruner()
)

# =========================================
# RUN STUDY
# =========================================
study.optimize(objective, n_trials=10)

# =========================================
# RESULTS
# =========================================
print("\nBest Params:")
print(study.best_params)

print("\nBest Loss:")
print(study.best_value)

# Optional: Full trial analysis
df = study.trials_dataframe()
print(df.head())
