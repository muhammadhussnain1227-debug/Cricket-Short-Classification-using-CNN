import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2


def build_model(num_classes):
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1)
    ])

    resize_and_rescale = tf.keras.Sequential([
        layers.Resizing(256, 256),
        layers.Rescaling(1./255)
    ])

    base_model = MobileNetV2(
        input_shape=(256,256,3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = True  # fine-tune last layers

    # freeze initial layers
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model = tf.keras.Sequential([
        data_augmentation,
        resize_and_rescale,
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu",
                     kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax")
    ])
    return model