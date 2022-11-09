import os

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import pandas as pd
import rasterio


def add_path(df, path):
    '''
    Adds the path to the images and labels in the dataframe
    '''
    sources = [
        'change', 'extent', 'occurrence', 'recurrence',
        'seasonality', 'transitions', 'nasadem'
        ]
    
    df['feature_path'] = (
            str(path / "train_features")
            / df.image_id.path.with_suffix(".tif").path
        )

    for i in os.listdir('training_data/train_features'):
        for source in sources:
            if source in i:
                df[source] = path /'train_features' / i
    df["label_path"] = (
            str(path / "train_labels")
            / df.chip_id.path.with_suffix(".tif").path
        )

    return df

def get_paths_by_chip(image_level_df):
    """
    Function that takes as input the meta_dataframe
    and return a dataframe with the chip id and both path for vv and vh.
    """
    paths = []
    for chip, group in image_level_df.groupby("chip_id"):
        vv_path = group[group.polarization == "vv"]["feature_path"].values[0]
        vh_path = group[group.polarization == "vh"]["feature_path"].values[0]
        nasadem_path = group["nasadem"].values[0]
        change_path = group["change"].values[0]
        extent_path = group["extent"].values[0]
        occurrence_path = group["occurrence"].values[0]
        recurrence_path = group["recurrence"].values[0]
        seasonality_path = group["seasonality"].values[0]
        transitions_path = group["transitions"].values[0]
        paths.append([
                chip, vv_path,
                vh_path,
                nasadem_path,
                change_path,
                extent_path,
                occurrence_path,
                recurrence_path,
                seasonality_path,
                transitions_path
            ])
    return pd.DataFrame(
            paths,
            columns=["chip_id",
                "vv_path",
                "vh_path",
                "nasadem",
                "change",
                "extent",
                "occurrence",
                "recurrence",
                "seasonality",
                "transitions"
                ]
            )

def get_images(feature_path, label_path):
    features = []
    labels = []
    masks = []

    paths = label_path['label_path'].to_list()
    nb_cols = len(paths)
    #load labels
    for i in range(nb_cols):
        with rasterio.open(paths[i]) as lp:
            img = lp.read(1)

        #create a list of mask for missing pixels
        mask = np.zeros(img.shape, dtype=np.uint8)
        mask[np.where(img == 255)] = 0

        labels.append(ma.array(img.astype('float32'), mask = mask))
        masks.append(mask)

    #load features
    cols = [
            "vv_path", "vh_path", "nasadem", "change", "extent",
            "seasonality", "occurrence", "recurrence", "transitions"
        ]
    nb_cols = len(feature_path)
    for row in range(nb_cols) :
        images = []
        for col in cols:
            with rasterio.open(feature_path.loc[row, col]) as img:
          #load the tif file
                if(col in ["vv_path", "vh_path"]):
                    #apply transformation: clip values out of -30;0 range and map them to 0; 255 range then convert to uint8
                    images.append(
                            ma.array(np.uint8(
                                np.clip(img.read(1), -30, 0)*(-8.5)
                            ), mask = masks[row])
                        )
                elif col == "nasadem":
                    #clip values > 255 and converto to uint8
                    images.append(
                            ma.array(np.uint8(
                                np.clip(img.read(1), 0, 255)
                            ), mask = masks[row])
                        )
                else:
                #no transformation, values are already between 0 and 255 and in uint8 format
                    images.append(ma.array(img.read(1), mask = masks[row]))
        features.append(np.stack(images, axis=-1))  

    return np.array(features), np.array(labels)

def transform_images(train_images, label_images):
    transform = A.Compose([
        A.RandomRotate90(),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.OneOf([
            A.MotionBlur(p=0.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.2)]
    )
    train_x_aug = []
    train_y_aug = []
    for i in range(len(train_images)):
        t = transform(image=train_images[i], mask=label_images[i])
        train_x_aug.append(t['image'])
        train_y_aug.append(t['mask'])

    train_x_aug = np.array(train_x_aug)
    train_y_aug = np.array(train_y_aug)
    return train_x_aug, train_y_aug

def plot_loss(results, model_name):
    plt.figure(figsize=(8, 8))
    iou = round(max(results.history['IOU_coef']), 4)
    plt.title(f"{model_name} Learning curve - IoU: {iou}")
    plt.plot(results.history["loss"], label="loss")
    plt.plot(results.history["val_loss"], label="val_loss")
    plt.plot( np.argmin(
            results.history["val_loss"]),
            np.min(results.history["val_loss"]),
            marker="x",
            color="r",
            label="best model"
        )
    plt.xlabel("Epochs")
    plt.ylabel("log_loss")
    plt.legend()
    plt.savefig(f'../output/{model_name}_loss.png')
    plt.show()
