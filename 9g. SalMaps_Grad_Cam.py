# =========================
# 0. Download Image
# =========================
!wget -q -O img.jpg https://upload.wikimedia.org/wikipedia/commons/2/26/YellowLabradorLooking_new.jpg

# =========================
# 1. Imports
# =========================
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

import matplotlib.pyplot as plt
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# 2. Load Model
# =========================
weights = models.ResNet50_Weights.IMAGENET1K_V1
model = models.resnet50(weights=weights)
model = model.to(device).eval()

labels = weights.meta["categories"]

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

print("Prediction:", labels[pred_class.item()], pred_prob.item())

plt.imshow(img_pil)
plt.title(labels[pred_class.item()])
plt.axis("off")
plt.show()

# =========================
# 6. Saliency Map
# =========================
def get_saliency(model, x):
    x = x.clone().detach().requires_grad_(True)

    out = model(x)
    cls = out.argmax()

    model.zero_grad()
    out[0, cls].backward()

    grad = x.grad.abs()
    grad, _ = torch.max(grad, dim=1)

    return grad.squeeze().detach().cpu()

saliency = get_saliency(model, img_tensor)

plt.imshow(saliency, cmap="hot")
plt.title("Saliency Map")
plt.axis("off")
plt.show()

# =========================
# 7. Grad-CAM Hooks
# =========================
features = None
grads = None

def forward_hook(module, inp, out):
    global features
    features = out

def backward_hook(module, grad_input, grad_output):
    global grads
    grads = grad_output[0]

target_layer = model.layer4[-1]

forward_handle = target_layer.register_forward_hook(forward_hook)
backward_handle = target_layer.register_full_backward_hook(backward_hook)

# =========================
# 8. Grad-CAM Forward + Backward
# =========================
out = model(img_tensor)
cls = out.argmax()

model.zero_grad()
out[0, cls].backward()

# weights: [1, 2048]
cam_weights = torch.mean(grads, dim=[2, 3])

# features: [1, 2048, 7, 7]
cam = torch.zeros(features.shape[2:], device=device)

for i, w in enumerate(cam_weights[0]):
    cam += w * features[0, i]

cam = F.relu(cam)

# Normalize
cam = cam - cam.min()
cam = cam / (cam.max() + 1e-8)

cam = cam.detach().cpu()

# =========================
# 9. Resize CAM using torchvision
# =========================
cam = cam.unsqueeze(0)                         # [1, 7, 7]
cam = TF.resize(cam, [224, 224], antialias=True)
cam = cam.squeeze(0)                           # [224, 224]

# =========================
# 10. Overlay using torchvision
# =========================
img_tensor_vis = TF.to_tensor(img_pil)         # [3, H, W]
img_tensor_vis = TF.resize(
    img_tensor_vis,
    [224, 224],
    antialias=True
)

heatmap = plt.get_cmap("jet")(cam.numpy())[:, :, :3]
heatmap = torch.tensor(heatmap).permute(2, 0, 1).float()

overlay = heatmap * 0.4 + img_tensor_vis * 0.6
overlay = overlay.clamp(0, 1)

plt.imshow(TF.to_pil_image(overlay))
plt.title("Grad-CAM")
plt.axis("off")
plt.show()

# =========================
# 11. Remove Hooks
# =========================
forward_handle.remove()
backward_handle.remove()
