import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import ast

device = torch.device("cpu")

NUM_CLASSES = 315

# load model
model = models.efficientnet_b0(weights=None)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)

model.load_state_dict(torch.load("model/coin_model_best.pth", map_location=device))
model.eval()

# load classes
with open("classes.txt") as f:
    classes = ast.literal_eval(f.read())

# transform
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

def predict_image(img_path):

    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        probs = F.softmax(outputs, dim=1)
        pred = torch.argmax(probs,1)

    coin = classes[pred.item()]
    value, currency, country = coin.split(",")

    confidence = probs[0][pred.item()].item() * 100

    return f"{value} ({currency}) - {country}", round(confidence, 2)