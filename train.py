import os
import numpy as np
from keras.applications import VGG16
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, EarlyStopping
from visualization_utils import plot_loss, plot_confusion_matrix

# Set the path to the base_model folder
base_model_folder = 'base_model'

# Set the path to the saved VGG16 model weights
weights_path = os.path.join(base_model_folder, 'vgg16_weights.h5')

# Set the path to your training images folder
train_data_dir = 'training_images'

# Set the batch size and number of epochs
batch_size = 32
epochs = 10

# Set the target image size
target_size = (224, 224)

# Set the number of classes
num_classes = 4   # 4 additional classes + 1000 classes from VGG16

# Set up data augmentation for training
train_datagen = ImageDataGenerator(
    shear_range=0.2,
    # rescale=1. / 255,
    zoom_range=0.1,
    horizontal_flip=True,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.9, 1.1]
)

# Generate batches of augmented data from the training set
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)


# Load the pre-trained VGG16 model
base_model = VGG16(weights=None, include_top=False, input_shape=(224, 224, 3))

# Freeze the layers of the pre-trained model
for layer in base_model.layers:
    layer.trainable = False

# Add your own classification layers on top of the pre-trained model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)  # Add dropout for regularization
predictions = Dense(num_classes, activation='softmax')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Load the compatible layers from the saved weights file
model.load_weights(weights_path, by_name=True)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Create a TensorBoard callback
tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1)

# Create an EarlyStopping callback to prevent overfitting
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model using the generated data
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    callbacks=[tensorboard_callback, early_stopping_callback]
)

# Save the trained model
model.save('trained_model_3.h5')
# Save the trained model

# trained_model is weights = None
# trained_model_with_imagenet -> weights = imagenet, layers = False
# trained_model_with_imagenet_2 -> weights = imagenet, layers = True
# trained_model_2 -> include_top = True
# trained_model_3 -> without data augmentation
y_true = train_generator.classes
y_pred = model.predict(train_generator)
y_pred = np.argmax(y_pred, axis=1)

# Plot confusion matrix
plot_confusion_matrix(y_true, y_pred, classes=['bobcat', 'deer', 'opossum', 'raccoon'])

# Plot loss
plot_loss(history)
model.summary()
