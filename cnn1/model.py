path_to_training = "/home/kcsmta/Desktop/WORKS/SPECT/code/data/data_training"
path_to_testing="/home/kcsmta/Desktop/WORKS/SPECT/code/data/data_testing"

import os, cv2, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns
#  if use Jupyter notebook
# %matplotlib inline 

from keras import backend as K
from keras.models import Sequential
from keras.layers import Input, Dropout, Flatten, Conv2D, MaxPool2D, Dense, Activation
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils import np_utils

TRAIN_DIR='/home/kcsmta/Desktop/WORKS/SPECT/code/data/data_training'
TEST_DIR='/home/kcsmta/Desktop/WORKS/SPECT/code/data/data_testing'
ROWS = 256
COLS = 256
ROWS2 = 64
COLS2 = 64
CHANNELS = 3

train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)]

for train_image in train_images:
    print train_image
