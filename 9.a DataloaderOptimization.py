# ================================
# 1. IMPORTS
# ================================
import time
import gc
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ================================
# 2. DEVICE SETUP
# ================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ================================
# 3. DATA TRANSFORMS
# ================================
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ================================
# 4. DATASET
# ================================
train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

# ================================
# 5. SIMPLE MODEL
# ================================
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.net(x)

# ================================
# 6. TRAIN LOOP
# ================================
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0

    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

# ================================
# 7. MEASURE DATALOADER SPEED
# ================================
def measure_epoch_time(loader):
    start = time.time()

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device)

    end = time.time()
    return (end - start) * 1000  # ms

# ================================
# 8. EXPERIMENT: NUM_WORKERS
# ================================
def experiment_num_workers(dataset):
    results = {}

    workers_list = [0, 2, 4, 6, 8]

    for nw in workers_list:
        print(f"\nTesting num_workers={nw}")

        loader = DataLoader(
            dataset,
            batch_size=64,
            shuffle=True,
            num_workers=nw,
            pin_memory=True
        )

        try:
            # warmup
            for _ in range(2):
                measure_epoch_time(loader)

            # actual measurement
            times = []
            for _ in range(3):
                t = measure_epoch_time(loader)
                times.append(t)

            avg_time = sum(times) / len(times)
            results[nw] = avg_time

            print(f"Avg Time: {avg_time:.2f} ms")

        except RuntimeError as e:
            print(f"Error with workers={nw}")
            results[nw] = float("inf")

        # cleanup
        del loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results

# ================================
# 9. EXPERIMENT: BATCH SIZE
# ================================
def experiment_batch_size(dataset):
    results = {}

    batch_sizes = [32, 64, 128, 256]

    for bs in batch_sizes:
        print(f"\nTesting batch_size={bs}")

        loader = DataLoader(
            dataset,
            batch_size=bs,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        try:
            # warmup
            for _ in range(2):
                measure_epoch_time(loader)

            times = []
            for _ in range(3):
                t = measure_epoch_time(loader)
                times.append(t)

            avg_time = sum(times) / len(times)
            results[bs] = avg_time

            print(f"Avg Time: {avg_time:.2f} ms")

        except RuntimeError:
            print(f"OOM with batch_size={bs}")
            results[bs] = float("inf")

        del loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results

# ================================
# 10. FINAL OPTIMIZED LOADER
# ================================
def get_optimized_loader(dataset):
    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    return loader

# ================================
# 11. MAIN EXECUTION
# ================================
if __name__ == "__main__":

    print("\n===== Experiment: num_workers =====")
    worker_results = experiment_num_workers(train_dataset)
    print("Worker Results:", worker_results)

    print("\n===== Experiment: batch_size =====")
    batch_results = experiment_batch_size(train_dataset)
    print("Batch Results:", batch_results)

    print("\n===== Training with Optimized Loader =====")

    train_loader = get_optimized_loader(train_dataset)

    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    epochs = 5

    for epoch in range(epochs):
        start = time.time()

        loss = train_one_epoch(model, train_loader, optimizer, criterion)

        end = time.time()

        print(f"Epoch {epoch+1}: Loss={loss:.4f}, Time={(end-start):.2f}s")

    print("\nTraining Complete")
