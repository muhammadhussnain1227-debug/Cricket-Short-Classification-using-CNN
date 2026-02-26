import tensorflow as tf

def load_dataset(data_path, img_size=(256,256), batch_size=32):

    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        data_path,
        shuffle=True,
        image_size=img_size,
        batch_size=batch_size
    )

    class_names = dataset.class_names

    dataset = dataset.cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)

    return dataset, class_names


def split_dataset(ds, train_split=0.8, val_split=0.1):

    ds_size = len(ds)
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)

    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size + val_size)

    return train_ds, val_ds, test_ds