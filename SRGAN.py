# the SSGANN model 
# based on the tutorial: https://medium.com/@birla.deepak26/single-image-super-resolution-using-gans-keras-aca310f33112
# change paths for you're data base

# for animation during the training session
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# for data extraction
import numpy as np
from PIL import Image
import os

# for batch creation
import random as rd

# for creating the NN
from tensorflow.keras.layers import Activation, BatchNormalization, Input, Flatten, Dense
from tensorflow.keras.layers import UpSampling2D, Conv2D, add
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

# for a custom loss function
from tensorflow.keras.applications.vgg19 import VGG19
import tensorflow.keras.backend as K

# Global Vars
#############

# get my dataset folder path
DATASET_PATH = os.environ["DATASETS"] + "/place_images/"

# images sizes
HR_IMAGE_SIZE = 128, 128 # size for resize after open
DOWNGRADE_FACTOR = 4 # the downgrade factor between the LR and HR image (HR size must be a multiple of DOWNGRADE_FACTor)
LR_IMAGE_SIZE = HR_IMAGE_SIZE[0] // DOWNGRADE_FACTOR, HR_IMAGE_SIZE[1] // DOWNGRADE_FACTOR # CALCULATE THE LR SIZE

# Neuralnet inputs shape
GENE_INPUT_SHAPE = None, None, 3 # the input shape of the generator (NONE = the size don't matter and the 3 = number of channels)
DISC_INPUT_SHAPE = HR_IMAGE_SIZE[0], HR_IMAGE_SIZE[1], 3 # the input shape of the discriminator
GAN_INPUT_SHAPE = LR_IMAGE_SIZE[0], LR_IMAGE_SIZE[1], 3 # the input shape of the gan

# the save path for trained models
SAVE_PATH = "E:/users/romain/Dekstop/pgm/Python/machine learning/keras/models/SRGAN/" 

# SAVE AND LOAD INFO
LOAD = False
SAVE = True
SAVR_RATE = 10 # save every how many epochs

# Functions
###########

# generate the x_train data
def get_one_image():
    # get a random image path
    path = DATASET_PATH
    path += rd.choice(os.listdir(path)) + "/"
    path += rd.choice(os.listdir(path))

    # extract the image and resize
    img = Image.open(path)
    imgy = img.resize(HR_IMAGE_SIZE)
    imgx = img.resize(LR_IMAGE_SIZE)

    # create the arrays and map the value form 0-255 to 0-1
    x = np.array(imgx)
    x = x.astype('float32') / 255

    y = np.array(imgy)
    y = y.astype('float32') / 255

    # check the image format and size must be: (X, Y, 3)
    if len(y.shape) != 3 or y.shape[2] != 3:
        return get_one_image()

    # return a batch with only one image
    return np.array([x,]), np.array([y,])

# a custom loss
def vgg_loss(y_true, y_pred):
    
    vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=(128, 128,3))
    vgg19.trainable = False
    # Make trainable as False
    for l in vgg19.layers:
        l.trainable = False
    model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)
    model.trainable = False
    
    return K.mean(K.square(model(y_true) - model(y_pred)))


# Creat the models graphs
######################### 

# the generator

# the input layer (who extract the data from image)
# one convulution form 3 chanels to 64 chanels
input_layer = Input(shape=GENE_INPUT_SHAPE)
generator = Conv2D(filters = 64, kernel_size = 9, strides = 1, padding = "same")(input_layer)
generator = LeakyReLU(alpha=.1)(generator)

gene_model = generator

# the resudial blocks (who treaat the data from image to extract the most usefull)
# tow convolution + add the input image to the generated image
for i in range(16):
    model = generator

    model = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same")(model)
    model = LeakyReLU(alpha=.1)(model)
    model = BatchNormalization(momentum = 0.5)(model)

    model = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same")(model)
    model = LeakyReLU(alpha=.1)(model)
    model = BatchNormalization(momentum = 0.5)(model)

    generator = add([model, generator])

# the end of the residual blocks
# one convolution + add the image befor residuals blocks
generator = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same")(generator)
generator = BatchNormalization(momentum = 0.5)(generator)
generator = add([gene_model, generator])

# the upsampling blocks (who upscale the image by 2)
# one convolution + upscalling
for i in range(2):
    generator = Conv2D(filters = 256, kernel_size = 3, strides = 1, padding = "same")(generator)
    generator = UpSampling2D(size = 2)(generator)
    generator = LeakyReLU(alpha = 0.2)(generator)

# the final layer (who creat the new image)
# one convolution from 64 chanels to 3 chanels
generator = Conv2D(filters = 3, kernel_size = 9, strides = 1, padding = "same")(generator)
generator = Activation('sigmoid')(generator)

# end of the graph creation
generator = Model(inputs = input_layer, outputs = generator)


# the discriminator

# the discriminator input
# extract featurs from the image
# convolution form a chanels to 64 chanels
dis_input = Input(shape = DISC_INPUT_SHAPE)

discriminator = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same")(dis_input)
discriminator = LeakyReLU(alpha = 0.2)(discriminator)

# the discrominator blocks data
# chanels, kernel size, strides (jump of the convolution)
size = [[64,3,2],
        [128,3,1],
        [128,3,2],
        [256,3,1],
        [256,3,2],
        [512,3,1],
        [512,3,2]]

# add the blocks layer to the graph
# one convolution, one batch normalization, one activation layer
for data in size:
    discriminator = Conv2D(filters = data[0], kernel_size = data[1], strides = data[2], padding = "same")(discriminator)
    discriminator = BatchNormalization(momentum = 0.5)(discriminator)
    discriminator = LeakyReLU(alpha = 0.2)(discriminator)

# the dense layers (1024 nodes) (1 nodes)
# say if the image is a FAKE or a TRUE image
discriminator = Flatten()(discriminator)
discriminator = Dense(1024)(discriminator)
discriminator = LeakyReLU(alpha = 0.2)(discriminator)
       
discriminator = Dense(1)(discriminator)
discriminator = Activation('sigmoid')(discriminator) 

# end of the graph creation
discriminator = Model(inputs = dis_input, outputs = discriminator)

# Compilation
#############

# generator
# the optimizer 
optimizer=Adam(lr=2E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

# load & compilation
if LOAD: 
    generator.load_weights(SAVE_PATH + 'generator_model.h5')

_gnerator = generator # save an uncompile version of the generator
generator.compile(loss=vgg_loss, optimizer=optimizer)

# discriminator (use the same opti then the generator)
# load & compile
if LOAD:
    discriminator.loss_weights(SAVE_PATH + 'discriminator_model.h5')

discriminator.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])


# gann
# the optimizer
optimizer=Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

# creat the graph based on generator and discriminator
gan_input = Input(shape=GAN_INPUT_SHAPE)
x = generator(gan_input)
gan_output = discriminator(x)
gan = Model(inputs=gan_input, outputs=[x,gan_output])

# compile
gan.compile(loss=[vgg_loss, "binary_crossentropy"], loss_weights=[1., 1e-3], optimizer=optimizer, metrics=["accuracy"])


def animate(e):
    x, y = get_one_image()
    fake_img = generator.predict(x)

    inp = np.concatenate([fake_img, y])
    out = np.array([0,1])

    data1 = discriminator.train_on_batch(inp, out)

    x, y = get_one_image()
    out = np.array([1,])
    data2 = gan.train_on_batch(x, [y,out])

    print(e, data1, data2)

    images = generator.predict(X)

    for j in range(images.shape[0]):
        curr = sub_plots[j]
        curr.clear()

        im = images[j, :, :, :]

        curr.imshow(im)
        curr.axis('off')

    for j in range(4):
        curr = sub_plots[j + 4]
        curr.clear()

        im = X[j, :, :, :]

        curr.imshow(im)
        curr.axis('off')
    
    if e % 10 == 0:
        generator.save_weights("./" + 'gen_model.h5')
        discriminator.save_weights("./"+ 'dis_model.h5')


# setup for matplotlib
fig = plt.figure(figsize=(10,10))
sub_plots = []
x1,y1 = get_one_image()
x2,y2 = get_one_image()
x3,y3 = get_one_image()
x4,y4 = get_one_image()

X = np.concatenate((x1,x2,x3,x4))


for i in range(8):
    sub_plots.append(plt.subplot(2, 4, i+1))

ani = animation.FuncAnimation(fig, animate, interval=1)

plt.show()


x, y = get_one_image()
# show a upscalle image 
new = generator.predict(x)
plt.imshow(x[0])
plt.show()
plt.imshow(new[0])
plt.show()
plt.imshow(y[0])
plt.show()