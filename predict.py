import cv2
import torch
import numpy as np
from PIL import Image
import winsound  # Windows-only for sound alert
from torchvision import transforms

# Device Configuration (Use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Wild animals that require an alert
WILD_ANIMALS = {
    "Tiger", "Lion", "Leopard", "Bear", "Wolf", "Fox", "Coyote", "Hippopotamus",
    "Hyena", "Elephant", "Rhinoceros", "Jaguar", "Panther"
}

# ✅ Class indices based on dataset (update if needed)
CLASS_INDICES = [
    "Antelope", "Badger", "Bat", "Bear", "Bee", "Beetle", "Bison", "Boar", "Butterfly", "Cat",
    "Caterpillar", "Chimpanzee", "Cockroach", "Cow", "Coyote", "Crab", "Crow", "Deer", "Dog", "Dolphin",
    "Donkey", "Dragonfly", "Duck", "Eagle", "Elephant", "Flamingo", "Fly", "Fox", "Goat", "Goldfish",
    "Goose", "Gorilla", "Grasshopper", "Hamster", "Hare", "Hedgehog", "Hippopotamus", "Hornbill", "Horse", "Hummingbird",
    "Hyena", "Jellyfish", "Kangaroo", "Koala", "Ladybugs", "Leopard", "Lion", "Lizard", "Lobster", "Mosquito",
    "Moth", "Mouse", "Octopus", "Okapi", "Orangutan", "Otter", "Owl", "Ox", "Oyster", "Panda",
    "Parrot", "Pelecaniformes", "Penguin", "Pig", "Pigeon", "Porcupine", "Possum", "Raccoon", "Rat", "Reindeer",
    "Rhinoceros", "Sandpiper", "Seahorse", "Seal", "Shark", "Sheep", "Snake", "Sparrow", "Squid", "Squirrel",
    "Starfish", "Swan", "Tiger", "Turkey", "Turtle", "Whale", "Wolf", "Wombat", "Woodpecker", "Zebra"
]

# ✅ Define Image Transformation Pipeline
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # Resize to ViT input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ✅ Load Pre-trained ViT Model (Ensure it's the trained model)
vit_model.to(device)
vit_model.eval()

# 🚀 Function to Predict Animal from Image
def predict_animal(image):
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = vit_model(image_tensor)  # Direct model output
        
        # Handle cases where the model returns raw tensor
        if isinstance(outputs, torch.Tensor):
            outputs = outputs  # Use directly
        else:
            outputs = outputs.logits  # Extract logits if available

        predicted_idx = torch.argmax(outputs, dim=1).item()

    predicted_label = CLASS_INDICES[predicted_idx]
    return predicted_label

# 🚨 Function to Issue Alert for Wild Animals
def issue_alert(animal_name):
    if animal_name in WILD_ANIMALS:
        print(f"🚨 ALERT: {animal_name} detected! Stay safe! 🚨")
        winsound.Beep(1000, 500)  # Beep sound (Windows only)
    else:
        print(f"✅ No threat detected. Animal: {animal_name}")

# 🎥 OpenCV Live Camera Feed for Animal Detection
def live_animal_detection():
    cap = cv2.VideoCapture(0)  # Open the camera
    
    if not cap.isOpened():
        print("Error: Camera not accessible.")
        return

    print("📸 Press 's' to capture an image, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Show the camera feed
        cv2.imshow("Wild Animal Detection - Press 's' to Capture", frame)

        # Wait for the 's' key to capture an image
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            # 🧠 Predict the animal in the captured frame
            predicted_animal = predict_animal(frame)

            # 🚨 Issue Alert if the detected animal is wild
            issue_alert(predicted_animal)

            # 🏷️ Display the prediction on the frame
            cv2.putText(frame, f"Animal: {predicted_animal}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Show the captured frame with prediction
            cv2.imshow("Captured Frame with Prediction", frame)
            cv2.waitKey(0)  # Wait for key press to close the frame

        # Press 'q' to exit
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 🚀 Run the Live Detection System
live_animal_detection()
