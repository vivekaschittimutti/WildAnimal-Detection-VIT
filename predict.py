import cv2
import numpy as np
from PIL import Image
import winsound  # For audio alert (Windows systems only)

# List of classes that trigger an alert
ALERT_CLASSES = ["Tiger","Lion","Jaguar","Cheetah","Leopard"]  # Replace with actual class names that require alerts

# Function to capture image from the system camera
def capture_image_from_camera():
    cap = cv2.VideoCapture(0)  # Open the default camera

    if not cap.isOpened():
        print("Error: Camera could not be opened.")
        return None

    print("Press 's' to capture an image, or 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        cv2.imshow("Camera", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):  # Press 's' to save the frame
            cap.release()
            cv2.destroyAllWindows()
            return frame
        elif key == ord('q'):  # Press 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()
    return None

# Function to preprocess the captured image for the model
def preprocess_image(image):
    # Convert BGR (OpenCV format) to RGB
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert the NumPy array to a PIL Image
    pil_img = Image.fromarray(img)
    
    # Apply the transform
    img_tensor = transform(pil_img).unsqueeze(0)  # Apply transform and add batch dimension
    return img_tensor.to(device)

# Function to make predictions
def predict_image(model, image_tensor):
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor).logits
        predicted_class = torch.argmax(outputs, dim=1).item()

        class_indices = {v: k for k, v in train_generator.class_indices.items()}  # Reverse the class indices
        predicted_label = class_indices[predicted_class]
        return predicted_label

# Function to issue an alert
def issue_alert(predicted_label):
    if predicted_label in ALERT_CLASSES:
        print(f"ALERT: {predicted_label} detected!")
        
        # Play a sound alert (optional, Windows-specific)
        winsound.Beep(1000, 500)  # Beep sound with frequency and duration
        
        # Additional actions like sending a notification can be added here
    else:
        print(f"No alert. Detected: {predicted_label}")

# Capture an image from the camera
captured_frame = capture_image_from_camera()

if captured_frame is not None:
    # Preprocess the image
    image_tensor = preprocess_image(captured_frame)

    # Predict the class
    prediction = predict_image(vit_model, image_tensor)
    print(f"Predicted Class: {prediction}")

    # Issue an alert if necessary
    issue_alert(prediction)

    # Display the captured frame with the prediction
    cv2.putText(captured_frame, f"Prediction: {prediction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Captured Image", captured_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
