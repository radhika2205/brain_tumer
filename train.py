import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os, json

from model.cnn_model import build_model, fine_tune_model, get_callbacks, EPOCHS
from utils.data_preprocessing import get_data_generators, compute_class_weights

os.makedirs('model', exist_ok=True)

print("=" * 50)
print("   Brain Tumor Detection — VGG16 Training")
print("=" * 50)


# ── Step 1: Load Data ─────────────────────────────────────────
print("\n[1] Loading dataset...")
train_gen, val_gen, test_gen = get_data_generators('dataset')
print(f"    Train : {train_gen.samples} images")
print(f"    Val   : {val_gen.samples} images")
print(f"    Test  : {test_gen.samples} images")
print(f"    Classes: {train_gen.class_indices}")

# Save class indices for Streamlit app
with open('model/class_indices.json', 'w') as f:
    json.dump(train_gen.class_indices, f, indent=2)
print("    class_indices.json saved.")


# ── Step 2: Class Weights ─────────────────────────────────────
print("\n[2] Computing class weights...")
class_weights = compute_class_weights(train_gen)
print(f"    Class weights: {class_weights}")


# ── Step 3: Phase 1 — Train head only ────────────────────────
print("\n[3] Phase 1: Training classification head (VGG16 base frozen)...")
model, base_model = build_model()
model.summary()

history1 = model.fit(
    train_gen,
    epochs=15,
    validation_data=val_gen,
    callbacks=get_callbacks('model/best_model.h5'),
    class_weight=class_weights,
    verbose=1
)


# ── Step 4: Phase 2 — Fine-tune block5 ───────────────────────
print("\n[4] Phase 2: Fine-tuning VGG16 block5 (block5_conv1/2/3)...")
model = fine_tune_model(model, base_model)

history2 = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=get_callbacks('model/best_model.h5'),
    class_weight=class_weights,
    verbose=1
)


# ── Step 5: Evaluate ──────────────────────────────────────────
print("\n[5] Evaluating on test set...")
model.load_weights('model/best_model.h5')
test_loss, test_acc = model.evaluate(test_gen, verbose=0)
print(f"\n    Test Accuracy : {test_acc * 100:.2f}%")
print(f"    Test Loss     : {test_loss:.4f}")


# ── Step 6: Save Training Plot ────────────────────────────────
print("\n[6] Saving training history plot...")

acc   = history1.history['accuracy']   + history2.history['accuracy']
val_a = history1.history['val_accuracy'] + history2.history['val_accuracy']
loss  = history1.history['loss']       + history2.history['loss']
val_l = history1.history['val_loss']   + history2.history['val_loss']
split = len(history1.history['accuracy'])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('VGG16 Brain Tumor Detection — Training History', fontsize=13)

axes[0].plot(acc,   label='Train Accuracy', color='#534AB7', linewidth=2)
axes[0].plot(val_a, label='Val Accuracy',   color='#1D9E75', linewidth=2)
axes[0].axvline(x=split - 1, color='gray', linestyle='--', label='Fine-tune start')
axes[0].set_title('Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(loss,  label='Train Loss', color='#D85A30', linewidth=2)
axes[1].plot(val_l, label='Val Loss',   color='#BA7517', linewidth=2)
axes[1].axvline(x=split - 1, color='gray', linestyle='--', label='Fine-tune start')
axes[1].set_title('Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('model/training_history.png', dpi=150, bbox_inches='tight')
print("    Saved: model/training_history.png")

print("\n" + "=" * 50)
print(f"   Training Complete!  Accuracy: {test_acc*100:.2f}%")
print("   Model saved: model/best_model.h5")
print("=" * 50)
