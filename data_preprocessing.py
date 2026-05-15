import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os

IMG_SIZE   = 224
BATCH_SIZE = 32


# ── Data Generators ───────────────────────────────────────────
def get_data_generators(dataset_path='dataset'):

    # Augmentation only for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )

    # Only rescale for val and test
    val_test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(dataset_path, 'train'),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )

    val_gen = val_test_datagen.flow_from_directory(
        os.path.join(dataset_path, 'val'),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    test_gen = val_test_datagen.flow_from_directory(
        os.path.join(dataset_path, 'test'),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    return train_gen, val_gen, test_gen


# ── Preprocess single image for prediction ───────────────────
def preprocess_image(pil_image):
    """
    Input  : PIL Image
    Output : numpy array (1, 224, 224, 3) normalized 0-1
    """
    img = pil_image.convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# ── Class weights for imbalanced data ────────────────────────
def compute_class_weights(train_gen):
    classes = np.unique(train_gen.classes)
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=train_gen.classes
    )
    return dict(zip(classes, weights))
