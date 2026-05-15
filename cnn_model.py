import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# ── Config ────────────────────────────────────────────────────
CLASS_NAMES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
NUM_CLASSES = 4
IMG_SIZE    = 224
BATCH_SIZE  = 32
EPOCHS      = 30


# ── Build VGG16 Model ─────────────────────────────────────────
def build_model():
    """
    Phase 1: VGG16 base frozen, only classification head trains
    """
    base_model = VGG16(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base_model.trainable = False  # freeze VGG16

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)

    # Custom CNN classification head
    x = layers.Flatten()(x)
    x = layers.Dense(4096, activation='relu', name='fc1')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(4096, activation='relu', name='fc2')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(1024, activation='relu', name='fc3')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax', name='predictions')(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model, base_model


# ── Fine-tune VGG16 block5 ────────────────────────────────────
def fine_tune_model(model, base_model):
    """
    Phase 2: Unfreeze last conv block (block5) of VGG16
    block5_conv1, block5_conv2, block5_conv3 trainable
    """
    base_model.trainable = True
    for layer in base_model.layers:
        if layer.name in ['block5_conv1', 'block5_conv2', 'block5_conv3']:
            layer.trainable = True
        else:
            layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=1e-5),  # very low LR for fine-tuning
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ── Callbacks ─────────────────────────────────────────────────
def get_callbacks(save_path='model/best_model.h5'):
    return [
        EarlyStopping(
            monitor='val_accuracy',
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=save_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,
            patience=3,
            min_lr=1e-8,
            verbose=1
        )
    ]
