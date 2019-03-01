# the SSGANN model 
# based on the tutorial: https://medium.com/@birla.deepak26/single-image-super-resolution-using-gans-keras-aca310f33112
import random as rd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import numpy as np

# get my dataset folder path
DATASET_PATH = os.environ["DATASETS"]

# import the data for creating the NN
from tensorflow.keras.layers import Activation, BatchNormalization, Input, Flatten, Dense
from tensorflow.keras.layers import UpSampling2D, Conv2D, add
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam


# generate the x_train data
def get_one_image():
    path = DATASET_PATH + "/place_images/"
    path += rd.choice(os.listdir(path)) + "/"
    path += rd.choice(os.listdir(path))

    img = Image.open(path)
    img = img.resize((128,128))
    y = np.array(img)
    y = y.astype('float32') / 255
    size = y.shape

    if len(y.shape) != 3 or y.shape[2] != 3 or y.shape[1] > 2000 or y.shape[0] > 2000:
        return get_one_image()

    new_image = np.zeros((size[0]//4, size[1]//4, size[2]))    

    # creat the low quality images
    for i in range(new_image.shape[0]):
        for j in range(new_image.shape[1]):
            new_image[i][j] = y[i*4][j*4]

    y_size = new_image.shape
    y = y[0:y_size[0]*4]
    new_y = []
    
    for row in y:
        new_y.append(row[0:y_size[1]*4])

    return np.array([new_image,]), np.array([new_y,])

print("training data loaded and generate")

# genarator

# the input and generator layer
input_layer = Input(shape=(None, None, 3))
generator = Conv2D(filters = 64, kernel_size = 9, strides = 1, padding = "same")(input_layer)
generator = LeakyReLU(alpha=.1)(generator)

gene_model = generator

# the resudial blocks
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
generator = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same")(generator)
generator = BatchNormalization(momentum = 0.5)(generator)
generator = add([gene_model, generator])

# the upsampling blocks
for i in range(2):
    generator = Conv2D(filters = 256, kernel_size = 3, strides = 1, padding = "same")(generator)
    generator = UpSampling2D(size = 2)(generator)
    generator = LeakyReLU(alpha = 0.2)(generator)

# the final layer
generator = Conv2D(filters = 3, kernel_size = 9, strides = 1, padding = "same")(generator)
generator = Activation('sigmoid')(generator)
    
generator = Model(inputs = input_layer, outputs = generator)
generator.summary()

# the discriminator
dis_input = Input(shape = (128, 128, 3))
        
discriminator = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same")(dis_input)
discriminator = LeakyReLU(alpha = 0.2)(discriminator)

# the discrominator blocks
size = [[64,3,2],
        [128,3,1],
        [128,3,2],
        [256,3,1],
        [256,3,2],
        [512,3,1],
        [512,3,2]]

# add the blocks layer
for data in size:
    discriminator = Conv2D(filters = data[0], kernel_size = data[1], strides = data[2], padding = "same")(discriminator)
    discriminator = BatchNormalization(momentum = 0.5)(discriminator)
    discriminator = LeakyReLU(alpha = 0.2)(discriminator)

# the dense layer and final layer
discriminator = Flatten()(discriminator)
discriminator = Dense(1024)(discriminator)
discriminator = LeakyReLU(alpha = 0.2)(discriminator)
       
discriminator = Dense(1)(discriminator)
discriminator = Activation('sigmoid')(discriminator) 
        
discriminator = Model(inputs = dis_input, outputs = discriminator)

# a custom loss for gann
from tensorflow.keras.applications.vgg19 import VGG19
import tensorflow.keras.backend as K


def vgg_loss(y_true, y_pred):
    
    vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=(128, 128,3))
    vgg19.trainable = False
    # Make trainable as False
    for l in vgg19.layers:
        l.trainable = False
    model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)
    model.trainable = False
    
    return K.mean(K.square(model(y_true) - model(y_pred)))

optimizer=Adam(lr=2E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

# compilation
generator.load_weights("./" + 'gen_model.h5')
generator.save("./" + 'gen.h5')
generator.compile(loss=vgg_loss, optimizer=optimizer)
discriminator.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
discriminator.load_weights("./" + 'dis_model.h5')


optimizer=Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
# the gann 
gan_input = Input(shape=(32,32,3))
x = generator(gan_input)
gan_output = discriminator(x)
gan = Model(inputs=gan_input, outputs=[x,gan_output])
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
        #im = np.reshape(im, [self.img_rows, self.img_cols)

        curr.imshow(im)
        curr.axis('off')

    for j in range(4):
        curr = sub_plots[j + 4]
        curr.clear()

        im = X[j, :, :, :]
        #im = np.reshape(im, [self.img_rows, self.img_cols)

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