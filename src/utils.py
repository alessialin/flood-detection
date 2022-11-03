import os
import random

import albumentations as A
import cv2
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import pandas as pd
import rasterio
import tensorflow as tf

from keras.models import Model
from keras.layers import Input, BatchNormalization, Activation, Dropout, concatenate
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from pandas_path import path
from pathlib import Path
from skimage.morpholoy import label
from tensorflow.keras.optimizers import Adam