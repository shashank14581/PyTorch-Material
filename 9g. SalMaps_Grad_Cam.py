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
model = models.resnet50(weights=weights).to(device).eval()
labels = weights.meta["categories"]

# =========================
# 3. Load + Preprocess Image
# =========================
img_pil = Image.open("img.jpg").convert("RGB")

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
# 4. Prediction
# =========================
with torch.no_grad():
    logits = model(img_tensor)
    probs = F.softmax(logits, dim=1)
    pred_prob, pred_class = probs.max(dim=1)

print("Prediction:", labels[pred_class.item()], pred_prob.item())

plt.imshow(img_pil)
plt.title(labels[pred_class.item()])
plt.axis("off")
plt.show()

# =========================
# 5. Saliency Map
# =========================
x = img_tensor.clone().detach().requires_grad_(True)

out = model(x)
cls = out.argmax()

model.zero_grad()
out[0, cls].backward()

saliency = x.grad.abs().max(dim=1)[0].squeeze().detach().cpu()

plt.imshow(saliency, cmap="hot")
plt.title("Saliency Map")
plt.axis("off")
plt.show()

# =========================
# 6. Grad-CAM
# =========================
features, grads = None, None

def forward_hook(module, inp, out):
    global features
    features = out

def backward_hook(module, grad_in, grad_out):
    global grads
    grads = grad_out[0]

target_layer = model.layer4[-1]

fh = target_layer.register_forward_hook(forward_hook)
bh = target_layer.register_full_backward_hook(backward_hook)

out = model(img_tensor)
cls = out.argmax()

model.zero_grad()
out[0, cls].backward()

weights_cam = grads.mean(dim=(2, 3))              # [1, 2048]
cam = (weights_cam[0, :, None, None] * features[0]).sum(dim=0)

cam = F.relu(cam)
cam = cam - cam.min()
cam = cam / (cam.max() + 1e-8)
cam = cam.detach().cpu()

fh.remove()
bh.remove()

# =========================
# 7. Grad-CAM Visualization
# =========================
cam = TF.resize(cam.unsqueeze(0), [224, 224], antialias=True).squeeze(0)

img = TF.resize(TF.to_tensor(img_pil), [224, 224], antialias=True)
heatmap = torch.tensor(plt.cm.jet(cam.numpy())[:, :, :3]).permute(2, 0, 1).float()

plt.imshow(TF.to_pil_image((0.6 * img + 0.4 * heatmap).clamp(0, 1)))
plt.title("Grad-CAM")
plt.axis("off")
plt.show()
