import os
from keras.applications import VGG16
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
from visualization_utils import plot_confusion_matrix, plot_loss
import numpy as np

# Set the path to the base_model folder
base_model_folder = 'base_model'

# Set the path to the saved VGG16 model weights
weights_path = os.path.join(base_model_folder, 'vgg16_weights.h5')

# Load the pre-trained VGG16 model
base_model = VGG16(weights=None, include_top=False, input_shape=(224, 224, 3))

# Freeze the layers of the pre-trained model
for layer in base_model.layers:
    layer.trainable = False

# Set the number of classes
num_classes = 4  # 4 additional classes + 1000 classes from VGG16

# Add your own classification layers on top of the pre-trained model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Load the compatible layers from the saved weights file
model.load_weights(weights_path, by_name=True)

# Set the path to your training images folder
train_data_dir = 'training_images'

# Set the batch size and number of epochs
batch_size = 32
epochs = 10

# Set up data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Generate batches of augmented data from the training set
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Create a TensorBoard callback
tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1)
# Train the model using the generated data
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    callbacks=[tensorboard_callback]
)

# Save the trained model
model.save('trained_model_with_imagenet_2.keras')
# trained_model is weights = None
# trained_model_with_imagenet -> weights = imagenet, layers = False
# trained_model_with_imagenet_2 -> weights = imagenet, layers = True
y_true = train_generator.classes
y_pred = model.predict(train_generator)
y_pred = np.argmax(y_pred, axis=1)

# Plot confusion matrix
plot_confusion_matrix(y_true, y_pred, classes=['bobcat', 'deer', 'opossum', 'raccoon'])

# Plot loss
plot_loss(history)
model.summary()
