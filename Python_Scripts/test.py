import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO
from PIL import Image
import time

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ['fake', 'real']  # Make sure this order is correct for ResNet/VGG

# --- 1. Load All Models (Done once on startup) ---
print("Loading all models, this may take a moment...")

# Load YOLOv8-cls model
yolo_model = YOLO(r"D:\TrueScan\models\yolov8m-cls_best.pt")
print("✅ YOLOv8-cls model loaded.")

# --- Load ResNet50 Model ---
resnet_model = models.resnet50()
num_ftrs_resnet = resnet_model.fc.in_features
resnet_model.fc = nn.Linear(num_ftrs_resnet, len(CLASS_NAMES))
resnet_model.load_state_dict(torch.load(r"D:\TrueScan\models\resnet50_best.pt", map_location=DEVICE))
resnet_model = resnet_model.to(DEVICE)
resnet_model.eval()
print("✅ ResNet50 model loaded.")

# --- Load VGG19 Model ---
vgg_model = models.vgg19_bn()
num_ftrs_vgg = vgg_model.classifier[6].in_features
vgg_model.classifier[6] = nn.Linear(num_ftrs_vgg, len(CLASS_NAMES))
vgg_model.load_state_dict(torch.load(r"D:\TrueScan\models\vgg19_BN_best.pt", map_location=DEVICE))
vgg_model = vgg_model.to(DEVICE)
vgg_model.eval()
print("✅ VGG19 model loaded.")

# Define image transformations for PyTorch models
data_transforms =transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) #Still standardizing
])


# --- 2. Individual Prediction Functions ---

def predict_yolo(image_path):
    start_time = time.time()
    results = yolo_model(image_path, verbose=False)
    result = results[0]
    processing_time = f"{(time.time() - start_time):.2f}s"

    if result.probs is not None:
        prediction = result.names[result.probs.top1]
        confidence = float(result.probs.top1conf)
    else:
        prediction, confidence = 'No Prediction', 0.0

    return {
        'model_name': 'YOLOv8-cls',
        'prediction': prediction,
        'confidence': round(confidence * 100, 2),
        'inference_time': processing_time
    }


def predict_pytorch_model(image_path, model, model_name):
    start_time = time.time()
    img = Image.open(image_path).convert('RGB')
    img_t = data_transforms(img)
    batch_t = torch.unsqueeze(img_t, 0).to(DEVICE)

    with torch.no_grad():
        output = model(batch_t)

    processing_time = f"{(time.time() - start_time):.2f}s"

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    confidence, predicted_idx = torch.max(probabilities, 0)

    prediction = CLASS_NAMES[predicted_idx.item()]
    confidence_score = confidence.item()

    return {
        'model_name': model_name,
        'prediction': prediction,
        'confidence': round(confidence_score * 100, 2),
        'inference_time': processing_time
    }


# --- 3. Main Scalable Functions ---

def run_all_models(image_path):
    """
    Runs all three models and returns a list of their results.
    """
    results_list = []
    results_list.append(predict_yolo(image_path))
    results_list.append(predict_pytorch_model(image_path, resnet_model, 'ResNet50'))
    results_list.append(predict_pytorch_model(image_path, vgg_model, 'VGG19'))
    return results_list


def calculate_overall_prediction(model_results):
    """
    Calculates the simple average of all model confidence scores.
    """
    if not isinstance(model_results, list) or not model_results:
        return 0.0

    total_confidence = 0
    for result in model_results:
        total_confidence += result.get('confidence', 0.0)

    return round(total_confidence / len(model_results), 2)

