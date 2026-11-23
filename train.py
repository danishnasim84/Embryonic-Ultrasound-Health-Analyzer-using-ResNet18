"""
train.py  -  Fetal Ultrasound Classifier
==========================================
Trains a ResNet-18 model on the ultrasound images and saves model.pth.

Classes: benign | malignant | normal

HOW TO RUN:
    pip install torch torchvision pillow
    python train.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

# ------------------------------------------------------------------ #
DATA_DIR   = r"Ultrasound Fetus Dataset\Ultrasound Fetus Dataset\Data\Data"
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
MODEL_SAVE_PATH = "model.pth"
# ------------------------------------------------------------------ #

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),         # simple data augmentation
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---- Step 2: Load datasets from the train/ and validation/ folders ----
train_folder = os.path.join(DATA_DIR, "train")
val_folder   = os.path.join(DATA_DIR, "validation")

train_dataset = datasets.ImageFolder(train_folder, transform=train_transform)
val_dataset   = datasets.ImageFolder(val_folder,   transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,num_workers=0)

# The class names are taken directly from the subfolder names
class_names = train_dataset.classes
NUM_CLASSES  = len(class_names)

print(f"Classes found: {class_names}")
print(f"Training images  : {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")
print()

# ---- Step 3: Build the model ----
# We use ResNet-18 with pre-trained weights (transfer learning).
# Only the final layer is replaced to match our 3-class problem.
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, NUM_CLASSES)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

best_val_accuracy = 0.0

print("Starting training...")
print("-" * 50)

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss    = 0.0
    correct_train   = 0
    total_train     = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()          # clear old gradients
        outputs = model(images)        # forward pass
        loss    = criterion(outputs, labels)
        loss.backward()                # backpropagation
        optimizer.step()               # update weights

        running_loss  += loss.item()
        _, predicted   = torch.max(outputs, 1)
        correct_train += (predicted == labels).sum().item()
        total_train   += labels.size(0)

    train_accuracy = 100.0 * correct_train / total_train

    model.eval()
    correct_val = 0
    total_val   = 0

    with torch.no_grad():              # no gradients needed for validation
        for images, labels in val_loader:
            images  = images.to(device)
            labels  = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct_val += (predicted == labels).sum().item()
            total_val   += labels.size(0)

    val_accuracy = 100.0 * correct_val / total_val
    avg_loss     = running_loss / len(train_loader)

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}]  "
          f"Loss: {avg_loss:.4f}  "
          f"Train Acc: {train_accuracy:.2f}%  "
          f"Val Acc: {val_accuracy:.2f}%")

    # Save the model only when validation accuracy improves
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save({
            "model_state_dict": model.state_dict(),
            "class_names":      class_names,
            "num_classes":      NUM_CLASSES,
        }, MODEL_SAVE_PATH)
        print(f"  --> Best model saved (val acc = {best_val_accuracy:.2f}%)")

print()
print("=" * 50)
print(f"Training complete!")
print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
print(f"Model saved to: {MODEL_SAVE_PATH}")
print("=" * 50)
