import tensorflow as tf
from src.data_loader import load_dataset, split_dataset
from src.model import build_model


DATA_PATH = r"C:\Users\User\Desktop\Projects\Cricket Short Classification using CNN\data" 
MODEL_PATH = "models/mobilenetv2.keras"


def train():

    dataset, class_names = load_dataset(DATA_PATH)
    train_ds, val_ds, test_ds = split_dataset(dataset)

    model = build_model(len(class_names))

    model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),  # lower learning rate
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),  # if labels integer
    metrics=["accuracy"]
)

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        callbacks=[early_stop]
    )

    model.evaluate(test_ds)

    model.save(MODEL_PATH)

    print("Model saved successfully!")


if __name__ == "__main__":
    train()