from transformers import AutoModelForImageClassification
import torchvision.transforms as transforms
from PIL import Image
import torch

model = AutoModelForImageClassification.from_pretrained("yusuf802/Leaf-Disease-Predictor")

def get_leaf_disease(img_path):
    if img_path == "": return None

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(img_path)
    inputs = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(inputs)
        prediction_idx = torch.argmax(outputs.logits, dim=1).item()

    return model.config.id2label[prediction_idx]