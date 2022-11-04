import random

import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Input
from pathlib import Path
from tensorflow.keras.optimizers import Adam

from utils.losses import LossFunctions
from utils.model_utils import get_unet, IOU_coef
from utils.utils import (
        add_path, get_paths_by_chip, get_images, plot_loss, transform_images
    )


if __name__ == '__main__':
    DATA_PATH = Path.cwd() / "training_data"
    MODEL_NAME = "unet_model_diceloss" 
    MODEL_NAME_SQ = "unet_model_diceloss_square" 
    train_metadata = pd.read_csv(
        DATA_PATH / "flood-training-metadata.csv",
        parse_dates=["scene_start"]
    )

    random.seed(123)
    img_size = 512

    train_metadata = add_path(train_metadata, DATA_PATH)

    # Sample 3 random floods for validation set
    flood_ids = train_metadata.flood_id.unique().tolist()
    val_flood_ids = random.sample(flood_ids, 3)
    test = train_metadata[train_metadata.flood_id.isin(val_flood_ids)]
    train = train_metadata[~train_metadata.flood_id.isin(val_flood_ids)]

    # Separate features from labels
    test_meta_x = get_paths_by_chip(test)
    test_meta_y = test[
            ["chip_id", "label_path"]
            ].drop_duplicates().reset_index(drop=True)
    train_meta_x = get_paths_by_chip(train)
    train_meta_y = train[
            ["chip_id", "label_path"]
            ].drop_duplicates().reset_index(drop=True)
    train_x, train_y = get_images(train_meta_x, train_meta_y)
    test_x, test_y = get_images(test_meta_x, test_meta_y)
    train_x_aug, train_y_aug = transform_images(train_x, train_y)

    train_x_final = np.concatenate((train_x, train_x_aug))
    train_y_final = np.concatenate((train_y, train_y_aug))

    input_img = Input((img_size, img_size, 9), name='img')
    model_diceloss = get_unet(
            input_img,
            n_filters=16,
            dropout=0.05,
            batchnorm=True
        )
    model_diceloss.compile(
            optimizer=Adam(),
            loss=LossFunctions.DiceLoss,
            metrics=[IOU_coef]
        )
    model_diceloss_sq = get_unet(
            input_img,
            n_filters=16,
            dropout=0.05,
            batchnorm=True
        )
    model_diceloss_sq.compile(
            optimizer=Adam(),
            loss=LossFunctions.DiceLoss_square,
            metrics=[IOU_coef]
        )

    callbacks = [
        EarlyStopping(patience=10, verbose=1),
        ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1)
    ]

    results = model_diceloss.fit(
            train_x_final,
            train_y_final,
            batch_size=8,
            epochs=100,
            callbacks=callbacks,
            validation_data=(test_x, test_y))

    plot_loss(results, MODEL_NAME)
    plot_loss(results, MODEL_NAME_SQ)
