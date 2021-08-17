# The Model structure is inspired by Jason Brownlee's blogs "How to Develop a GAN for Generating MNIST Handwritten Digits" and
# "How to Identify and Diagnose GAN Failure Modes" 
# https://machinelearningmastery.com/practical-guide-to-gan-failure-modes/
# https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-an-mnist-handwritten-digits-from-scratch-in-keras/

from matplotlib import pyplot
from keras.datasets.mnist import load_data
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LeakyReLU
import random
from keras.layers import Conv2DTranspose
from keras.layers import Reshape
import numpy as np
np.random.seed = 3
random.seed(3)
import tensorflow as tf
configuration = tf.compat.v1.ConfigProto()
configuration.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=configuration)


def Discriminator(in_shape=(28,28,1)):
    model = Sequential()
    model.add(Conv2D(64, (4,4), strides=(2, 2), padding='same', input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Conv2D(64, (4,4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
   
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


def Generator(latent_dim):
    model = Sequential()
    n_nodes = 128 * 7 * 7
    model.add(Dense(n_nodes, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((7, 7, 128)))
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(1, (7,7), activation='tanh', padding='same'))
    return model


def GAN(generator,discriminator):
    
    model = Sequential()
    discriminator.trainable = False
    model.add(generator)
    model.add(discriminator)
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


def load_dataset():
    (trainX, _), (_, _) = load_data() 
    X = np.expand_dims(trainX, axis=-1)
    X = X.astype('float32')
    X = (X-127.5) /127.5   # transform pixle value into value between [-1,1]
    return X



# create a batch of randomized input vectors
def rand_seq(dim,batch):
    X = np.random.randn(batch, dim)
    return X



#randomly choose real images, label as 1
def generate_real(dataset,batch):
    i= random.choices(range(dataset.shape[0]),k=batch)
    
    X = dataset[i]
    y = np.ones((batch,1))
    return X, y

# use generator to generate images, label as 0
def generate_fake(generator,dataset,batch):
    x_input = rand_seq(dim,batch)
    X = generator.predict(x_input)
    y = np.zeros((batch,1))
    return X,y


def generate_data(generator,dataset,batch):
    X_real,y_real = generate_real(dataset,batch//2)
    X_fake,y_fake = generate_fake(generator,dataset,batch//2)

    X, y = np.vstack((X_real, X_fake)), np.vstack((y_real, y_fake))
    return X,y

# plot loss
def plot_history(d_list,g_list):
    pyplot.subplot(2, 1, 1)
    pyplot.plot(d_list, label='d_loss')
    pyplot.legend()

    pyplot.subplot(2, 1, 2)
    pyplot.plot(g_list, label='g_loss')
    pyplot.legend()

    pyplot.savefig('loss.png')
    pyplot.close()



def train():
    dataset = load_dataset()
    bat_per_epo = int(dataset.shape[0] / batch)
    generator = Generator(dim)
    discriminator = Discriminator()
    GAN_model = GAN(generator,discriminator)
    b = 0
    d_list = []
    g_list = []
    for i in range(n_epoc):
        for j in range(bat_per_epo):
            X,y = generate_data(generator,dataset,batch)
            dloss,_ = discriminator.train_on_batch(X,y)
            X_g = rand_seq(dim,batch)
            y_g = np.ones((batch, 1))
            gloss,_ = GAN_model.train_on_batch(X_g,y_g)
            d_list.append(dloss)
            g_list.append(gloss)
            if dloss<=0.002:
                b+=1
            
                
            if j%5 == 0:
                print('>%d, %d/%d, d=%.3f, g=%.3f' % (i+1, j+1, n_epoc, dloss, gloss))
        if b>=20:
            break
    return generator,discriminator,d_list,g_list

def draw_result(generator,n_epoc):
    h = w = 28
    num_gen = 25

    X = rand_seq(50,25)

    X = generator.predict(X)
    X = (X + 1) / 2.0
    # plot of generation
    n = np.sqrt(num_gen).astype(np.int32)
    I_generated = np.empty((h*n, w*n))
    d = 0
    for i in range(n):
        for j in range(n):
            I_generated[i*h:(i+1)*h, j*w:(j+1)*w] = X[d,:,:,0]
            d+=1
    pyplot.figure(figsize=(4, 4))
    pyplot.axis("off")
    pyplot.imshow(I_generated, cmap='gray_r')
    pyplot.savefig('result_{}.png'.format(n_epoc))
    pyplot.show()







if __name__ == '__main__':
    batch = 256
    n_epoc = 400
    dim = 50

    generator,discriminator,d_list,g_list = train()
    plot_history(d_list,g_list)

    generator_json = generator.to_json()
    with open("generator.json", "w") as json_file:
        json_file.write(generator_json)

    generator.save_weights("generator.h5")

    discriminator_json=discriminator.to_json()
    with open("discriminator.json", "w") as json_file:
        json_file.write(discriminator_json)

    discriminator.save_weights("discriminator.h5")

    draw_result(generator,n_epoc)