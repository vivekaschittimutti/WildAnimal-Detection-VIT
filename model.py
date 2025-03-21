import os
import zipfile
import torch
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from torch.utils.data import DataLoader, Dataset
from transformers import ViTForImageClassification, ViTFeatureExtractor
import requests
from PIL import Image
from io import BytesIO
from torchvision import transforms
import matplotlib.pyplot as plt

zip_path = os.path.join(os.getcwd(), 'archive (1).zip')

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall("dataset1")

data_dir = "dataset1/animals/animals"


IMG_SIZE = 224  
BATCH_SIZE = 32
EPOCHS = 30

datagen = ImageDataGenerator(
    rescale=1.0/255.0,  
    validation_split=0.2,  
    rotation_range=20,  
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)


train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)


feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
vit_model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=len(train_generator.class_indices),
    ignore_mismatched_sizes=True
)


for param in vit_model.parameters():
    param.requires_grad = False


for param in vit_model.classifier.parameters():
    param.requires_grad = True


def create_custom_classifier(num_labels):
    return torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(vit_model.config.hidden_size, 128),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(128, num_labels)
    )

vit_model.classifier = create_custom_classifier(len(train_generator.class_indices))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vit_model.to(device)


optimizer = torch.optim.Adam(vit_model.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()


transform = transforms.Compose([
    transforms.ToPILImage(),  
    transforms.Resize((IMG_SIZE, IMG_SIZE)), 
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class CustomImageDataset(Dataset):
    def __init__(self, generator, transform=None):
        self.generator = generator
        self.image_paths = []
        self.labels = []
        self.class_indices = generator.class_indices
        self.transform = transform

        
        for class_name, label in self.class_indices.items():
            class_path = os.path.join(generator.directory, class_name)
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                self.image_paths.append(img_path)
                self.labels.append(label)  # Store the integer label

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = img_array / 255.0 
        label = self.labels[idx]

        
        if self.transform:
            img_array = self.transform(img_array) 

        return img_array, torch.tensor(label)


train_dataset = CustomImageDataset(train_generator, transform=transform)
val_dataset = CustomImageDataset(val_generator, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


def train_model(model, train_loader, val_loader, epochs):
    training_losses = []
    validation_accuracies = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images).logits
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        training_losses.append(avg_train_loss)

        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss}")

        # Validation step
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images).logits
                val_loss += loss_fn(outputs, labels).item()
                correct += (outputs.argmax(1) == labels).type(torch.float).sum().item()
        avg_val_loss = val_loss / len(val_loader)
        accuracy = correct / len(val_loader.dataset)
        validation_accuracies.append(accuracy)

        print(f"Validation Loss: {avg_val_loss}, Accuracy: {accuracy}")

    
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs+1), training_losses, label="Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs+1), validation_accuracies, label="Validation Accuracy", color="green")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy Over Epochs")
    plt.legend()

    plt.show()


train_model(vit_model, train_loader, val_loader, EPOCHS)
