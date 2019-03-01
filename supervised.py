# the SSGANN model 
# based on the tutorial: https://medium.com/@birla.deepak26/single-image-super-resolution-using-gans-keras-aca310f33112
import random as rd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import numpy as np

from tensorflow.keras.layers import Activation, BatchNormalization, Input, Flatten, Dense
from tensorflow.keras.layers import UpSampling2D, Conv2D, add
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

