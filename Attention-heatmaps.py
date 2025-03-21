import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torchvision.datasets as datasets
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
from PIL import Image
from timm import create_model
from sklearn.model_selection import train_test_split

# Define Device (Use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define Transformations (Data Augmentation & Normalization)
transform = T.Compose([
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(10),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Define dataset path
dataset_path = "animals/animals"

# Auto-split dataset into train (80%) and test (20%)
train_dir = "animals/train"
test_dir = "animals/test"

if not os.path.exists(train_dir):
    os.makedirs(train_dir)
    os.makedirs(test_dir)
    
    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            images = os.listdir(class_path)
            train_images, test_images = train_test_split(images, test_size=0.2, random_state=42)

            os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
            os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

            for img in train_images:
                shutil.copy(os.path.join(class_path, img), os.path.join(train_dir, class_name, img))
            
            for img in test_images:
                shutil.copy(os.path.join(class_path, img), os.path.join(test_dir, class_name, img))

# Load the dataset dynamically
train_data = datasets.ImageFolder(root=train_dir, transform=transform)
test_data = datasets.ImageFolder(root=test_dir, transform=transform)

# Create Data Loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)

# Load Pretrained Vision Transformer Model (ViT)
model = create_model("vit_base_patch16_224", pretrained=True, num_classes=len(train_data.classes))
model = model.to(device)

# Define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=3e-4)

# Training Loop
num_epochs = 1
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")

# Save Trained Model
torch.save(model.state_dict(), "vit_animal_model.pth")

# Load a Sample Image for Attention Visualization
def visualize_attention(image_path, model):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model.forward_features(img_tensor)  # Extract ViT features

        # ViT outputs (batch_size, num_patches+1, hidden_dim)
        attn_weights = features[:, 1:, :].mean(dim=-1).cpu().numpy().reshape(1, -1)  # Remove class token

    # Determine patch grid size (assuming 16x16 patches for 224x224 input)
    grid_size = int(np.sqrt(attn_weights.shape[1]))
    attn_map = attn_weights.reshape(grid_size, grid_size)  # Reshape to grid

    # Normalize attention map
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
    attn_map = np.array(Image.fromarray(attn_map * 255).resize(image.size))

    # Plot the original image and attention overlay
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")

    plt.subplot(1, 2, 2)
    plt.imshow(image)
    plt.imshow(attn_map, cmap="jet", alpha=0.5)
    plt.title("Attention Heatmap")
    plt.show()

# Test with an image from your dataset
visualize_attention("animals/animals/leopard/3c2367666e.jpg", model)

    
