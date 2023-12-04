import os
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from sklearn.metrics import precision_score
from sklearn.preprocessing import LabelEncoder
import cv2

# Load the trained model
model = load_model('trained_model.h5')

# Set the path to the test images folder
test_data_dir = 'test_images'

# Set the target image size
target_size = (224, 224)

# Define the class names
class_names = ['bobcat', 'deer', 'opossum', 'raccoon']

# Get the list of image files in the test folder
image_files = os.listdir(test_data_dir)

# Initialize lists to store predicted classes and ground truth labels
predicted_classes = []
ground_truth_labels = []

# Iterate over the image files
for file in image_files:
    # Load and preprocess the image
    img_path = os.path.join(test_data_dir, file)
    img = image.load_img(img_path, target_size=target_size)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize the image

    # Make predictions
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)

    # Extract the ground truth label from the image file name
    ground_truth_label = file.split('-')[0]

    # Append predicted class and ground truth label to the lists
    predicted_classes.append(predicted_class)
    ground_truth_labels.append(ground_truth_label)

    # Print the predicted class and the corresponding image file name
    print(f"Image: {file} - Predicted Class: {class_names[predicted_class]}")

    # Print the probabilities of all predicted classes
    for i, prob in enumerate(predictions[0]):
        print(f"Probability of {class_names[i]}: {prob:.2f}")

    # Resize the image for display
    img_display = cv2.resize(cv2.imread(img_path), (400, 400))

    # Draw the accuracy in the upper-left corner of the image
    accuracy_text = f"Accuracy: {predictions[0][predicted_class]:.2f}"
    cv2.putText(img_display, accuracy_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    prediction_text = "Predicted Class:" + class_names[predicted_class]
    cv2.putText(img_display, prediction_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the image
    cv2.imshow('Test Image', img_display)
    cv2.waitKey(0)

# Encode the ground truth labels as numeric values
label_encoder = LabelEncoder()
ground_truth_labels_encoded = label_encoder.fit_transform(ground_truth_labels)

# Calculate the precision rate
precision = precision_score(ground_truth_labels_encoded, predicted_classes, average='macro')

# Print the precision rate
print(f"Precision Rate: {precision}")
