from keras.applications import VGG16
import os

# Set the path to the base_model folder
base_model_folder = 'base_model'

# Create the base_model folder if it doesn't exist
if not os.path.exists(base_model_folder):
    os.makedirs(base_model_folder)

# Set the path to save the VGG16 model weights
weights_path = os.path.join(base_model_folder, 'vgg16_weights.h5')

# Download the VGG16 model weights and save them
VGG16(weights='imagenet').save_weights(weights_path)