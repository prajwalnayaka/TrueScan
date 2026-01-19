import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import os
import time

# --- 1. HYPERPARAMETERS & SETUP ---
DATA_DIR = r'D:\Medical_Deepfake_2\dataset_classification'
# Model hyperparameters
NUM_CLASSES = 2  # We have two classes: 'real' and 'fake'
EPOCHS = 100
IMG_SIZE = 224
BATCH_SIZE = 8
SEED = 42

# Set seed for reproducibility
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Check if GPU is available and set the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 2. DATA TRANSFORMS & LOADING ---
# Define the transformations for the images.
# For training, we add data augmentation (RandomHorizontalFlip).
# For validation, we only resize and normalize.

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) #btw transform.Normalize is STANDARDIZATION
    ]),
    'val': transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])#btw transform.Normalize is STANDARDIZATION
    ]),
}

# Create ImageFolder datasets. This automatically finds classes from folder names.
print("Loading data...")
image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x]) for x in ['train', 'val']}

# Create DataLoader objects to load data in batches.
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=4) for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
print(f"Classes found: {class_names}")

# --- 3. MODEL SETUP ---
# Load a pre-trained ResNet50 model.
print("Loading pre-trained ResNet50 model...")
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

# ResNet50's final layer is called 'fc'. We need to replace it with a new
# layer that has the correct number of outputs for our problem (2 classes).
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, NUM_CLASSES)

# Send the model to the GPU
model = model.to(device)

# --- 4. TRAINING SETUP ---
# Define the loss function and the optimizer.
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)

# --- 5. THE TRAINING LOOP ---
def train_model(model, criterion, optimizer, num_epochs=100):
    since = time.time()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                # Track history only if in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model if it's the best one so far
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                # Save the model's state dictionary
                torch.save(model.state_dict(), 'resnet50_best.pt')
                print("New best model saved!")

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

# --- Start Training! ---
if __name__ == '__main__':
    train_model(model, criterion, optimizer, num_epochs=EPOCHS)