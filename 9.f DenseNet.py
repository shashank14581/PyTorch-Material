import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl

# -----------------------------
# DEVICE
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# CIFAR100 DATA
# -----------------------------
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408),
                         (0.2675, 0.2565, 0.2761))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408),
                         (0.2675, 0.2565, 0.2761))
])

train_dataset = torchvision.datasets.CIFAR100(
    root="./data", train=True, download=True, transform=transform_train
)

val_dataset = torchvision.datasets.CIFAR100(
    root="./data", train=False, download=True, transform=transform_test
)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)

# -----------------------------
# DENSE LAYER
# -----------------------------
class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()

        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        return torch.cat([x, out], dim=1)  # 🔥 concatenation

# -----------------------------
# DENSE BLOCK
# -----------------------------
class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate):
        super().__init__()

        layers = []
        for i in range(num_layers):
            layers.append(DenseLayer(
                in_channels + i * growth_rate,
                growth_rate
            ))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

# -----------------------------
# TRANSITION LAYER
# -----------------------------
class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.AvgPool2d(2)
        )

    def forward(self, x):
        return self.layer(x)

# -----------------------------
# DENSENET MODEL
# -----------------------------
class DenseNet(nn.Module):
    def __init__(self, growth_rate=32, num_classes=100):
        super().__init__()

        self.init_conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)

        # Dense Blocks
        self.block1 = DenseBlock(6, 64, growth_rate)
        self.trans1 = TransitionLayer(64 + 6 * growth_rate, 128)

        self.block2 = DenseBlock(12, 128, growth_rate)
        self.trans2 = TransitionLayer(128 + 12 * growth_rate, 256)

        self.block3 = DenseBlock(24, 256, growth_rate)
        self.trans3 = TransitionLayer(256 + 24 * growth_rate, 512)

        self.block4 = DenseBlock(16, 512, growth_rate)

        self.bn = nn.BatchNorm2d(512 + 16 * growth_rate)

        self.classifier = nn.Linear(512 + 16 * growth_rate, num_classes)

    def forward(self, x):
        x = self.init_conv(x)

        x = self.block1(x)
        x = self.trans1(x)

        x = self.block2(x)
        x = self.trans2(x)

        x = self.block3(x)
        x = self.trans3(x)

        x = self.block4(x)

        x = F.relu(self.bn(x))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)

        return self.classifier(x)

# -----------------------------
# LIGHTNING MODULE
# -----------------------------
class DenseNetLightning(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = DenseNet()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        acc = (logits.argmax(dim=1) == y).float().mean()

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        acc = (logits.argmax(dim=1) == y).float().mean()

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

# -----------------------------
# TRAIN
# -----------------------------
model = DenseNetLightning()

trainer = pl.Trainer(
    max_epochs=10,
    accelerator="auto",
    devices=1
)

trainer.fit(model, train_loader, val_loader)
