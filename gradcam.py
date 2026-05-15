import numpy as np
import tensorflow as tf
import cv2


# ── Grad-CAM Heatmap ──────────────────────────────────────────
def make_gradcam_heatmap(img_array, model, last_conv_layer_name='block5_conv3'):
    """
    VGG16 last conv layer = 'block5_conv3'
    img_array : (1, 224, 224, 3) normalized
    Returns   : heatmap numpy array (0 to 1)
    """
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index   = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads       = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap      = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap      = tf.squeeze(heatmap)
    heatmap      = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()


# ── Overlay heatmap on original MRI ──────────────────────────
def overlay_gradcam(original_img_array, heatmap, alpha=0.45):
    """
    original_img_array : numpy (224, 224, 3) values 0-255
    heatmap            : numpy (h, w) values 0-1
    Returns            : superimposed image uint8
    """
    heatmap_resized = cv2.resize(heatmap, (original_img_array.shape[1], original_img_array.shape[0]))
    heatmap_uint8   = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    superimposed = heatmap_colored * alpha + original_img_array * (1 - alpha)
    superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)
    return superimposed


# ── Full pipeline: PIL image → original + overlay + heatmap ──
def get_gradcam_overlay(pil_image, model, last_conv_layer='block5_conv3'):
    """
    pil_image : PIL Image uploaded by user
    Returns   : (original_np, overlay_np, heatmap_np)
    """
    img         = pil_image.convert('RGB').resize((224, 224))
    original_np = np.array(img)

    img_array   = original_np.astype('float32') / 255.0
    img_array   = np.expand_dims(img_array, axis=0)

    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer)
    overlay = overlay_gradcam(original_np, heatmap, alpha=0.45)

    return original_np, overlay, heatmap
