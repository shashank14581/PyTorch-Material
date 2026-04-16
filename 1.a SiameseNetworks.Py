# =========================
# 1. Imports
# =========================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random

import pytorch_lightning as pl
from pytorch_lightning import Trainer

from torchvision import datasets, transforms


# =========================
# 2. Siamese Dataset
# =========================
class SiameseDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.targets = dataset.targets

        # Build class index mapping
        self.class_to_indices = {}
        for idx, label in enumerate(self.targets):
            self.class_to_indices.setdefault(label, []).append(idx)

    def __getitem__(self, index):
        img1, label1 = self.dataset[index]

        if random.random() < 0.5:
            # Positive pair
            idx2 = random.choice(self.class_to_indices[label1])
            label = 0.0
        else:
            # Negative pair
            label2 = random.choice(list(self.class_to_indices.keys()))
            while label2 == label1:
                label2 = random.choice(list(self.class_to_indices.keys()))
            idx2 = random.choice(self.class_to_indices[label2])
            label = 1.0

        img2, _ = self.dataset[idx2]

        return img1, img2, torch.tensor(label, dtype=torch.float32)

    def __len__(self):
        return len(self.dataset)


# =========================
# 3. Base CNN (Embedding Network)
# =========================
class BaseCNN(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )

    def forward(self, x):
        return self.fc(self.features(x))


# =========================
# 4. Lightning Module
# =========================
class SiameseLightning(pl.LightningModule):
    def __init__(self, lr=1e-3, margin=1.0):
        super().__init__()
        self.save_hyperparameters()

        self.model = BaseCNN()

    def forward(self, x1, x2):
        return self.model(x1), self.model(x2)

    def contrastive_loss(self, out1, out2, label):
        distance = F.pairwise_distance(out1, out2)

        loss = (1 - label) * torch.pow(distance, 2) + \
               label * torch.pow(torch.clamp(self.hparams.margin - distance, min=0.0), 2)

        return loss.mean(), distance

    def training_step(self, batch, batch_idx):
        x1, x2, label = batch

        out1, out2 = self(x1, x2)

        loss, distance = self.contrastive_loss(out1, out2, label)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_dist_mean", distance.mean())

        return loss

    def validation_step(self, batch, batch_idx):
        x1, x2, label = batch

        out1, out2 = self(x1, x2)

        loss, distance = self.contrastive_loss(out1, out2, label)

        preds = (distance > 0.5).float()
        acc = (preds == label).float().mean()

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer


# =========================
# 5. DataModule
# =========================
class SiameseDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=128):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage=None):
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        train_base = datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )
        val_base = datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform
        )

        self.train_dataset = SiameseDataset(train_base)
        self.val_dataset = SiameseDataset(val_base)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True
        )


# =========================
# 6. Main Training Script
# =========================
if __name__ == "__main__":
    pl.seed_everything(42)

    model = SiameseLightning(lr=1e-3, margin=1.0)
    data_module = SiameseDataModule(batch_size=128)

    trainer = Trainer(
        max_epochs=10,
        accelerator="auto",
        devices="auto",
        precision="16-mixed",
        log_every_n_steps=10
    )

    trainer.fit(model, data_module)
