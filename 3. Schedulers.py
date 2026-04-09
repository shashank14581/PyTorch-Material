import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------------
# Dummy Model
# -----------------------------
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(100, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.net(x)


# -----------------------------
# Config
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleModel().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 🔁 CHANGE THIS TO TEST DIFFERENT SCHEDULERS
scheduler_type = "step"  # options: step, plateau, cosine


# -----------------------------
# Scheduler Setup
# -----------------------------
if scheduler_type == "step":
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=5,   # every 5 epochs
        gamma=0.5      # halve LR
    )

elif scheduler_type == "plateau":
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',    # monitor loss
        factor=0.5,
        patience=2
    )

elif scheduler_type == "cosine":
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=10,
        eta_min=1e-4
    )

else:
    scheduler = None


# -----------------------------
# Dummy Data
# -----------------------------
def get_dummy_batch():
    x = torch.randn(32, 100).to(device)
    y = torch.randint(0, 10, (32,)).to(device)
    return x, y


# -----------------------------
# Training Loop
# -----------------------------
num_epochs = 15

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for _ in range(10):  # 10 batches
        inputs, labels = get_dummy_batch()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / 10

    # -----------------------------
    # Scheduler Step
    # -----------------------------
    if scheduler_type == "plateau":
        scheduler.step(avg_loss)   # needs metric
    else:
        if scheduler is not None:
            scheduler.step()

    # -----------------------------
    # Logging
    # -----------------------------
    current_lr = optimizer.param_groups[0]["lr"]

    print(f"Epoch {epoch+1}")
    print(f"Loss: {avg_loss:.4f}")
    print(f"LR: {current_lr:.6f}")
    print("-" * 30)
