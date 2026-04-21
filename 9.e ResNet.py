import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
from torchvision import transforms

from torch.utils.data import DataLoader, random_split

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

# -----------------------------
# Config
# -----------------------------
SEED = 42
BATCH_SIZE = 64
NUM_WORKERS = 4
NUM_CLASSES = 100
MAX_EPOCHS = 10
LR = 1e-3

torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# DataModule
# -----------------------------
class CIFAR100DataModule(L.LightningDataModule):
    def __init__(self, batch_size=64):
        super().__init__()
        self.batch_size = batch_size

        self.mean = (0.5071, 0.4867, 0.4408)
        self.std = (0.2675, 0.2565, 0.2761)

        self.train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])

        self.val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])

    def setup(self, stage=None):
        full_train = torchvision.datasets.CIFAR100(
            root="./data",
            train=True,
            download=True,
            transform=self.train_transform
        )

        val_size = int(0.1 * len(full_train))
        train_size = len(full_train) - val_size

        self.train_dataset, self.val_dataset = random_split(
            full_train, [train_size, val_size]
        )

        self.test_dataset = torchvision.datasets.CIFAR100(
            root="./data",
            train=False,
            download=True,
            transform=self.val_transform
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=NUM_WORKERS)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=NUM_WORKERS)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=NUM_WORKERS)

# -----------------------------
# Plain CNN (Baseline)
# -----------------------------
class PlainCNN(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 32 -> 16
        x = self.pool(F.relu(self.conv2(x)))  # 16 -> 8
        x = self.pool(F.relu(self.conv3(x)))  # 8 -> 4

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# -----------------------------
# Residual Block
# -----------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# -----------------------------
# Simple ResNet
# -----------------------------
class SimpleResNet(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()

        self.conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(64)

        self.layer1 = ResidualBlock(64, 64)
        self.layer2 = ResidualBlock(64, 128, stride=2)
        self.layer3 = ResidualBlock(128, 256, stride=2)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# -----------------------------
# Lightning Module
# -----------------------------
class LitModel(L.LightningModule):
    def __init__(self, model, lr=1e-3):
        super().__init__()
        self.model = model
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def step(self, batch):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.step(batch)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.step(batch)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

# -----------------------------
# Train Function
# -----------------------------
def train_model(model_class, name):
    datamodule = CIFAR100DataModule(BATCH_SIZE)

    model = LitModel(model_class(NUM_CLASSES), lr=LR)

    checkpoint = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        filename=f"{name}-best"
    )

    trainer = L.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[checkpoint]
    )

    trainer.fit(model, datamodule)

# -----------------------------
# Run Experiments
# -----------------------------
if __name__ == "__main__":
    print("Training Plain CNN...")
    train_model(PlainCNN, "plain_cnn")

    print("Training ResNet...")
    train_model(SimpleResNet, "resnet")
