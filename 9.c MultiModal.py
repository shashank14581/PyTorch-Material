import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import torchvision
from torchvision import transforms

import pytorch_lightning as pl

import optuna
from optuna.integration import PyTorchLightningPruningCallback

# =========================================================
# CONFIG
# =========================================================
NUM_CLASSES = 9
BATCH_SIZE = 64
MAX_EPOCHS = 5

subset_classes = [
    "orchid", "poppy", "sunflower",
    "fox", "raccoon", "skunk",
    "butterfly", "caterpillar", "cockroach"
]

# =========================================================
# DATASET
# =========================================================
class MultiInputCIFAR(Dataset):
    def __init__(self, train=True):
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])

        self.dataset = torchvision.datasets.CIFAR100(
            root="./data", train=train, download=True, transform=transform
        )

        self.class_to_idx = {cls: i for i, cls in enumerate(self.dataset.classes)}
        self.selected_idx = [self.class_to_idx[c] for c in subset_classes]

        self.samples = [
            (img, label)
            for img, label in self.dataset
            if label in self.selected_idx
        ]

        self.label_map = {old: i for i, old in enumerate(self.selected_idx)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, label = self.samples[idx]
        label = self.label_map[label]

        metadata = torch.randn(10)
        reg_target = img.mean()

        return img, metadata, label, reg_target


# =========================================================
# DATAMODULE
# =========================================================
class CIFARDataModule(pl.LightningDataModule):
    def setup(self, stage=None):
        self.train_ds = MultiInputCIFAR(train=True)
        self.val_ds = MultiInputCIFAR(train=False)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=BATCH_SIZE, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=BATCH_SIZE)


# =========================================================
# MODEL
# =========================================================
class DynamicLightningModel(pl.LightningModule):
    def __init__(self, n_blocks, lr):
        super().__init__()
        self.save_hyperparameters()

        self.encoder_blocks = nn.ModuleList()
        in_channels = 3

        for _ in range(n_blocks):
            block = nn.Sequential(
                nn.Conv2d(in_channels, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
            self.encoder_blocks.append(block)
            in_channels = 32

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.meta_net = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU()
        )

        self.shared = nn.Linear(64, 64)

        self.classifier = nn.Linear(64, NUM_CLASSES)
        self.regressor = nn.Linear(64, 1)

    def forward(self, image, metadata):
        x = image

        for block in self.encoder_blocks:
            x = block(x)

        x = self.pool(x)
        x = x.view(x.size(0), -1)

        meta_feat = self.meta_net(metadata)

        combined = torch.cat([x, meta_feat], dim=1)
        shared = torch.relu(self.shared(combined))

        return self.classifier(shared), self.regressor(shared)

    def training_step(self, batch, batch_idx):
        images, meta, labels, reg_targets = batch

        class_out, reg_out = self(images, meta)

        loss_cls = F.cross_entropy(class_out, labels)
        loss_reg = F.mse_loss(reg_out.squeeze(), reg_targets)

        loss = loss_cls + loss_reg

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, meta, labels, reg_targets = batch

        class_out, reg_out = self(images, meta)

        loss_cls = F.cross_entropy(class_out, labels)
        loss_reg = F.mse_loss(reg_out.squeeze(), reg_targets)

        loss = loss_cls + loss_reg

        preds = torch.argmax(class_out, dim=1)
        acc = (preds == labels).float().mean()

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr)


# =========================================================
# OPTUNA OBJECTIVE
# =========================================================
def objective(trial):

    # -------- SEARCH SPACE --------
    n_blocks = trial.suggest_int("n_blocks", 2, 5)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    # -------- DATA --------
    data_module = CIFARDataModule()

    # -------- MODEL --------
    model = DynamicLightningModel(n_blocks=n_blocks, lr=lr)

    # -------- TRAINER --------
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        devices="auto",
        callbacks=[
            PyTorchLightningPruningCallback(trial, monitor="val_loss")
        ],
        enable_progress_bar=False
    )

    trainer.fit(model, data_module)

    return trainer.callback_metrics["val_loss"].item()


# =========================================================
# STUDY
# =========================================================
if __name__ == "__main__":
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner()
    )

    study.optimize(objective, n_trials=10)

    print("Best Trial:")
    print(study.best_trial.params)
