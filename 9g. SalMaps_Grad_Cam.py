# =========================
# 0. Download Image (ROBUST)
# =========================
!wget -q -O img.jpg https://upload.wikimedia.org/wikipedia/commons/2/26/YellowLabradorLooking_new.jpg

# =========================
# 1. Imports
# =========================
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# 2. Load Model
# =========================
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model = model.to(device).eval()

labels = models.ResNet50_Weights.IMAGENET1K_V1.meta["categories"]

# =========================
# 3. Load Image
# =========================
img_pil = Image.open("img.jpg").convert("RGB")

# =========================
# 4. Preprocess
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
])

img_tensor = transform(img_pil).unsqueeze(0).to(device)

# =========================
# 5. Prediction
# =========================
with torch.no_grad():
    logits = model(img_tensor)
    probs = F.softmax(logits, dim=1)
    pred_prob, pred_class = torch.max(probs, dim=1)

print("Prediction:", labels[pred_class], pred_prob.item())

plt.imshow(img_pil)
plt.title(labels[pred_class])
plt.axis("off")
plt.show()

# =========================
# 6. SALIENCY MAP
# =========================
def get_saliency(model, x):
    x = x.clone().detach().requires_grad_(True)

    out = model(x)
    cls = out.argmax()

    out[0, cls].backward()

    grad = x.grad.abs()
    grad, _ = torch.max(grad, dim=1)

    return grad.squeeze().cpu().numpy()

saliency = get_saliency(model, img_tensor)

plt.imshow(saliency, cmap="hot")
plt.title("Saliency Map")
plt.axis("off")
plt.show()

# =========================
# 7. GRAD-CAM
# =========================
features = None
grads = None

def forward_hook(module, inp, out):
    global features
    features = out

def backward_hook(module, grad_in, grad_out):
    global grads
    grads = grad_out[0]

target_layer = model.layer4[-1]
target_layer.register_forward_hook(forward_hook)
target_layer.register_backward_hook(backward_hook)

# Forward + Backward
out = model(img_tensor)
cls = out.argmax()

model.zero_grad()
out[0, cls].backward()

# Compute CAM
weights = torch.mean(grads, dim=[2,3])
cam = torch.zeros(features.shape[2:], device=device)

for i, w in enumerate(weights[0]):
    cam += w * features[0, i]

cam = F.relu(cam)
cam = cam.cpu().detach().numpy()

# Normalize
cam = (cam - cam.min()) / (cam.max() + 1e-8)

# Resize
cam = cv2.resize(cam, (224, 224))

# =========================
# 8. Overlay
# =========================
img_np = np.array(img_pil.resize((224,224))) / 255.0

heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
heatmap = heatmap / 255.0

overlay = heatmap * 0.4 + img_np

plt.imshow(overlay)
plt.title("Grad-CAM")
plt.axis("off")
plt.show()
