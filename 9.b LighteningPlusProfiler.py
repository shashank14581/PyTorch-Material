# =========================
# 1. Imports & Setup
# =========================
import os
import argparse
import warnings

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.optim as optim

from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import PyTorchProfiler

from torch.profiler import schedule
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchvision import datasets, transforms

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("medium")

pl.seed_everything(42)

device = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# 2. Config (CLI-driven)
# =========================
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_epochs", type=int, default=5)
    parser.add_argument("--model_type", type=str, default="heavy")  # heavy / efficient

    return parser.parse_args()


# =========================
# 3. DataModule
# =========================
class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, data_dir="./data", batch_size=128, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))
        ])

    def prepare_data(self):
        datasets.CIFAR10(self.data_dir, train=True, download=True)
        datasets.CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        self.train_dataset = datasets.CIFAR10(
            self.data_dir, train=True, transform=self.transform
        )
        self.val_dataset = datasets.CIFAR10(
            self.data_dir, train=False, transform=self.transform
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )


# =========================
# 4. Lightning Model
# =========================
class CIFAR10Model(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)

        if config["model_type"] == "heavy":
            conv_channels = (256, 512, 1024)
            linear_features = 2048
        else:
            conv_channels = (64, 128, 256)
            linear_features = 512

        layers = []
        in_channels = 3

        for out_channels in conv_channels:
            layers.append(nn.Conv2d(in_channels, out_channels, 3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))
            in_channels = out_channels

        self.conv = nn.Sequential(*layers)

        flattened_size = conv_channels[-1] * 4 * 4

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, linear_features),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(linear_features, 10)
        )

        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task="multiclass", num_classes=10)
        self.val_acc = Accuracy(task="multiclass", num_classes=10)

    def forward(self, x):
        return self.fc(self.conv(x))

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = self.train_acc(preds, y)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = self.val_acc(preds, y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return optim.AdamW(
            self.parameters(),
            lr=self.hparams["lr"]
        )


# =========================
# 5. Callbacks & Logger
# =========================
def get_callbacks():
    checkpoint = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        filename="best-model"
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=3,
        mode="min"
    )

    return [checkpoint, early_stop]


def get_logger():
    return TensorBoardLogger("logs", name="cifar10_exp")


# =========================
# 6. Profiler
# =========================
def get_profiler():
    return PyTorchProfiler(
        schedule=schedule(wait=1, warmup=1, active=3),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./log"),
        record_shapes=True,
        profile_memory=True
    )


# =========================
# 7. Main Pipeline
# =========================
def main():
    args = get_args()

    config = vars(args)

    # Data
    datamodule = CIFAR10DataModule(
        batch_size=config["batch_size"],
        num_workers=config["num_workers"]
    )

    # Model
    model = CIFAR10Model(config)

    # Trainer
    trainer = pl.Trainer(
        max_epochs=config["max_epochs"],
        accelerator="auto",
        devices="auto",

        precision="16-mixed",   # 🔥 huge performance boost

        callbacks=get_callbacks(),
        logger=get_logger(),

        profiler=get_profiler(),

        log_every_n_steps=50
    )

    # Train
    trainer.fit(model, datamodule)

    # Test (important)
    trainer.test(model, datamodule)


# =========================
# 8. Entry Point
# =========================
if __name__ == "__main__":
    main()
