import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import models
import os
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet152_Weights, Wide_ResNet50_2_Weights, ResNet101_Weights

'''
This script is a template for training a deep learning model on a dataset of harmonic radar signals,
under the form of IQ samples.

'''
# Define transformations for training and validation
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 for ResNet, only useful is IQ samples are not preprocessed
    transforms.ToTensor(),          # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization
])

# Load training and validation datasets
root_dir = os.getcwd()
train_dir = os.path.join(root_dir, 'train')
val_dir = os.path.join(root_dir, 'val')
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)
test_dir = os.path.join(root_dir, 'test')
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

# Create the test data loader
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Load pre-trained ResNet
#model = models.resnet50(pretrained=True)
model = models.resnet34(weights=ResNet34_Weights.DEFAULT)
num_classes = len(train_dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Move the model to GPU if available, useful in particular with the GPU cluster
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()  # For multi-class classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer

'''
This part of the script trains the model on the training dataset and evaluates it on the validation dataset.

'''
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=32):
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Track accuracy and loss
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        print(f"Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")

        # Evaluate on the validation set
        model.eval()
        val_corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)

        val_acc = val_corrects.double() / len(val_loader.dataset)
        print(f"Validation Acc: {val_acc:.4f}")

        # Save the model if it improves
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), 'best_model.pth')

    print(f"Best Validation Accuracy: {best_accuracy:.4f}")

# Train the model
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50)

# Evaluate the model
def evaluate_model(model, test_loader):
    model.eval()
    test_corrects = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            test_corrects += torch.sum(preds == labels.data)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_acc = test_corrects.double() / len(test_loader.dataset)
    print(f"Test Accuracy: {test_acc:.4f}")

    # Print classification report
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))

# Load the best model
model.load_state_dict(torch.load('best_model.pth'))

evaluate_model(model, test_loader)

# Save the entire model
torch.save(model, 'resnet_corrosion.pth')

# Load the entire model for inference
model = torch.load('resnet_corrosion.pth')
model.eval()

# Quick check that all classes were used and the model is not doing random stuff due to path issues
# This test is done due to suspiciously high accuracy on the test set

# Get the targets (labels) for all images in the validation dataset
val_labels = [label for _, label in val_dataset]

# Count the number of occurrences of each class
class_counts = Counter(val_labels)

# Get the counts
class_indices = list(class_counts.keys())
class_names = [val_dataset.classes[i] for i in class_indices]
class_samples = [class_counts[i] for i in class_indices]

# Generate the confusion matrix
def plot_confusion_matrix(model, test_loader, class_names):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute the confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=range(len(class_names)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    # Plot the confusion matrix
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
    plt.title('Confusion Matrix')
    plt.show()

    return all_preds, all_labels

# Class names
class_names = test_dataset.classes

# Plot the confusion matrix
all_preds, all_labels = plot_confusion_matrix(model, test_loader, class_names)

# Print classification report
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))
