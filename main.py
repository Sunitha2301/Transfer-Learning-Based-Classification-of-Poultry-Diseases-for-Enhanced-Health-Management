# STEP 1: Install Required Libraries
!pip install -q tensorflow matplotlib kaggle

# STEP 2: Import Required Libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os
import shutil

# STEP 3: Upload kaggle.json
from google.colab import files
files.upload()  # Upload your kaggle.json file here

# STEP 4: Configure Kaggle API
!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# STEP 5: Download and unzip the dataset
!kaggle datasets download -d allandclive/chicken-disease-1
!unzip -o chicken-disease-1.zip -d /content/poultry_dataset

# STEP 6: Fix Folder Structure if all images are dumped in one folder
src_folder = '/content/poultry_dataset/Train'
for filename in os.listdir(src_folder):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
    label = filename.split('.')[0].lower()
    class_dir = os.path.join(src_folder, label)
    os.makedirs(class_dir, exist_ok=True)
    shutil.move(os.path.join(src_folder, filename), os.path.join(class_dir, filename))

# STEP 7: Data Preprocessing
dataset_path = '/content/poultry_dataset/Train'
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2
)

train_gen = datagen.flow_from_directory(
    dataset_path,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    dataset_path,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# STEP 8: Build Transfer Learning Model
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
predictions = Dense(train_gen.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the base model
for layer in base_model.layers:
    layer.trainable = False

# STEP 9: Compile the Model
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# STEP 10: Train the Model
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10
)

# STEP 11: Plot Accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

