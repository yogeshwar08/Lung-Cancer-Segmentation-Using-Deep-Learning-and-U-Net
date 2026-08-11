import os
import cv2
import numpy as np
import tensorflow as tf

from glob import glob

from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping
)


MODEL_DIR = "models"

IMG_SIZE = (256, 256)

BATCH_SIZE = 8
EPOCHS = 10

IMG_DIR = "processed"
MASK_DIR = "masks_auto"

CHECKPOINT = os.path.join(
    MODEL_DIR,
    "unet_best.keras"
)

FINAL_MODEL = os.path.join(
    MODEL_DIR,
    "final_unet.keras"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


def conv_block(x, filters):

    x = layers.Conv2D(
        filters,
        3,
        padding="same"
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(
        filters,
        3,
        padding="same"
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    return x


def modified_unet(
    input_shape=(256, 256, 3),
    base_filters=32
):

    inputs = layers.Input(
        shape=input_shape
    )

    # Encoder
    c1 = conv_block(
        inputs,
        base_filters
    )

    p1 = layers.MaxPooling2D()(c1)

    c2 = conv_block(
        p1,
        base_filters * 2
    )

    p2 = layers.MaxPooling2D()(c2)

    c3 = conv_block(
        p2,
        base_filters * 4
    )

    p3 = layers.MaxPooling2D()(c3)

    c4 = conv_block(
        p3,
        base_filters * 8
    )

    p4 = layers.MaxPooling2D()(c4)

    # Bottleneck
    bn = conv_block(
        p4,
        base_filters * 16
    )

    # Decoder
    u4 = layers.Conv2DTranspose(
        base_filters * 8,
        2,
        strides=2,
        padding="same"
    )(bn)

    u4 = layers.Concatenate()(
        [u4, c4]
    )

    c5 = conv_block(
        u4,
        base_filters * 8
    )

    u3 = layers.Conv2DTranspose(
        base_filters * 4,
        2,
        strides=2,
        padding="same"
    )(c5)

    u3 = layers.Concatenate()(
        [u3, c3]
    )

    c6 = conv_block(
        u3,
        base_filters * 4
    )

    u2 = layers.Conv2DTranspose(
        base_filters * 2,
        2,
        strides=2,
        padding="same"
    )(c6)

    u2 = layers.Concatenate()(
        [u2, c2]
    )

    c7 = conv_block(
        u2,
        base_filters * 2
    )

    u1 = layers.Conv2DTranspose(
        base_filters,
        2,
        strides=2,
        padding="same"
    )(c7)

    u1 = layers.Concatenate()(
        [u1, c1]
    )

    c8 = conv_block(
        u1,
        base_filters
    )

    outputs = layers.Conv2D(
        1,
        1,
        activation="sigmoid"
    )(c8)

    return models.Model(
        inputs,
        outputs,
        name="Modified_U-Net"
    )


def load_pairs():

    image_paths = []
    mask_paths = []

    classes = [
        d for d in os.listdir(IMG_DIR)
        if os.path.isdir(
            os.path.join(
                IMG_DIR,
                d
            )
        )
    ]

    for cls in classes:

        imgs = []

        imgs.extend(
            glob(
                os.path.join(
                    IMG_DIR,
                    cls,
                    "*.jpg"
                )
            )
        )

        imgs.extend(
            glob(
                os.path.join(
                    IMG_DIR,
                    cls,
                    "*.jpeg"
                )
            )
        )

        for img_path in imgs:

            filename = os.path.basename(
                img_path
            )

            mask_path = os.path.join(
                MASK_DIR,
                cls,
                filename
            )

            if os.path.exists(
                mask_path
            ):

                image_paths.append(
                    img_path
                )

                mask_paths.append(
                    mask_path
                )

    return (
        image_paths,
        mask_paths
    )


def load_image(path):

    image = cv2.imread(
        path
    )

    image = cv2.resize(
        image,
        IMG_SIZE
    )

    image = (
        image.astype(
            np.float32
        ) / 255.0
    )

    return image


def load_mask(path):

    mask = cv2.imread(
        path,
        cv2.IMREAD_GRAYSCALE
    )

    mask = cv2.resize(
        mask,
        IMG_SIZE
    )

    mask = (
        mask > 127
    ).astype(
        np.float32
    )

    return mask[..., None]


def data_generator(
    image_paths,
    mask_paths,
    batch_size
):

    indices = np.arange(
        len(image_paths)
    )

    while True:

        np.random.shuffle(
            indices
        )

        for start in range(
            0,
            len(indices),
            batch_size
        ):

            batch_idx = indices[
                start:start + batch_size
            ]

            images = [
                load_image(
                    image_paths[i]
                )
                for i in batch_idx
            ]

            masks = [
                load_mask(
                    mask_paths[i]
                )
                for i in batch_idx
            ]

            yield (
                np.array(images),
                np.array(masks)
            )


def dice_loss(
    y_true,
    y_pred
):

    smooth = 1e-6

    y_true_f = tf.reshape(
        y_true,
        [-1]
    )

    y_pred_f = tf.reshape(
        y_pred,
        [-1]
    )

    intersection = tf.reduce_sum(
        y_true_f * y_pred_f
    )

    dice = (
        2.0 * intersection + smooth
    ) / (
        tf.reduce_sum(y_true_f)
        +
        tf.reduce_sum(y_pred_f)
        +
        smooth
    )

    return 1.0 - dice


def combined_loss(
    y_true,
    y_pred
):

    bce = tf.keras.losses.binary_crossentropy(
        y_true,
        y_pred
    )

    return (
        tf.reduce_mean(bce)
        +
        dice_loss(
            y_true,
            y_pred
        )
    )


def main():

    image_paths, mask_paths = load_pairs()

    if len(image_paths) < 2:

        raise RuntimeError(
            "Not enough JPEG images with matching masks."
        )

    indices = np.arange(
        len(image_paths)
    )

    np.random.seed(42)

    np.random.shuffle(
        indices
    )

    split = int(
        len(indices) * 0.8
    )

    train_idx = indices[:split]
    val_idx = indices[split:]

    train_imgs = [
        image_paths[i]
        for i in train_idx
    ]

    train_masks = [
        mask_paths[i]
        for i in train_idx
    ]

    val_imgs = [
        image_paths[i]
        for i in val_idx
    ]

    val_masks = [
        mask_paths[i]
        for i in val_idx
    ]

    model = modified_unet()

    model.compile(
        optimizer=Adam(
            learning_rate=1e-4
        ),
        loss=combined_loss,
        metrics=["accuracy"]
    )

    callbacks = [

        ModelCheckpoint(
            CHECKPOINT,
            monitor="val_loss",
            save_best_only=True
        ),

        EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        )
    ]

    train_steps = max(
        1,
        len(train_imgs) // BATCH_SIZE
    )

    val_steps = max(
        1,
        len(val_imgs) // BATCH_SIZE
    )

    model.fit(

        data_generator(
            train_imgs,
            train_masks,
            BATCH_SIZE
        ),

        validation_data=data_generator(
            val_imgs,
            val_masks,
            BATCH_SIZE
        ),

        steps_per_epoch=train_steps,

        validation_steps=val_steps,

        epochs=EPOCHS,

        callbacks=callbacks
    )

    model.save(
        FINAL_MODEL
    )

    print(
        "U-Net trained and saved:",
        FINAL_MODEL
    )


if __name__ == "__main__":
    main()