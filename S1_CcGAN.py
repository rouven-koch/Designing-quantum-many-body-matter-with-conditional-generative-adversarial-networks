# Author: Rouven Koch
# general GAN structure adapted from:
# https://gist.github.com/apozas/38d4640d9e6525b43db62dac846f1c19 (author: Alejandro Pozas-Kerstjens)

#-----------------------------------------------------------------------------
# spin-1 model - continuous conditional GAN 
#-----------------------------------------------------------------------------

# import libraries and packages
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import LeakyReLU, Activation, Input, Dense, Dropout, Concatenate, BatchNormalization, GaussianNoise
from keras.models import Model
from keras.regularizers import l1_l2
from keras.optimizers import Adam, SGD
from keras.utils import to_categorical
from keras import activations
from tensorflow.keras.models import load_model
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib


#-----------------------------------------------------------------------------
# CcGAN parameters
#------------------------------------------------------------------------------

# general
n = 18 # length of 1d-chain
noise_dim = 10 # latent space vector
omega_dim = 50 # frequency points
image_dim = 900 # total input parameter (18*50)
label_dim = 2 # number of conditional paramters

# for plotting
xs = [] 
xs = range(n)
ys_spin = np.linspace(-0.0,2,50)
ys_hubbard = np.linspace(-0.0,5,50)


#------------------------------------------------------------------------------
# data scaling
#------------------------------------------------------------------------------

# scaling of spectra and conditional parameter separately 
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1,1))
scaler_alpha = MinMaxScaler(feature_range=(0,1))
scaler_beta = MinMaxScaler(feature_range=(0,1))


#------------------------------------------------------------------------------
# data preparation
#------------------------------------------------------------------------------

# # load enhanced data 
# n_data = 2200
# x_train = np.load("/u/11/kochr1/unix/Rouven/Python/Project_2/results_paper/S1/x_data_all.npy")
# x_train = scaler.fit_transform(x_train)  
# l_train = np.load("/u/11/kochr1/unix/Rouven/Python/Project_2/results_paper/S1/label_all.npy")
# l_train_scale = l_train[:,:] 
# l_train_scale_1 = scaler_alpha.fit_transform(l_train_scale[:,0:1]) 
# l_train_scale_2 = scaler_beta.fit_transform(l_train_scale[:,1:]) 
# l_train_scale = np.concatenate((l_train_scale_1,l_train_scale_2), axis=1)

# if enhanced data does not exist:
dos_S1 = np.load("data/spectra_S1.npy", allow_pickle=True)
label_1_S1 = np.load("data/cparam_S1_1.npy", allow_pickle=True)
label_2_S1 = np.load("data/cparam_S1_2.npy", allow_pickle=True)
n_data = 2200

# format to 3D
d3_plot = np.zeros((n_data, omega_dim, 18))
for k in range(n_data):
    for j in range(18):
        for i in range(omega_dim):
            d3_plot[k,i,j]=dos_S1[ 900*k +  (j*omega_dim+i)]

label_all = np.zeros((n_data,2))
dos_all = d3_plot
label_all[:,0] = label_1_S1[:2200]
label_all[:,1] = label_2_S1[:2200]
dos_mirrow = np.flip(dos_all, axis=2)
label_all = np.concatenate((label_all, label_all))
dos_all_new = np.concatenate((dos_all, dos_mirrow))

# flatten zs
dos_format = np.zeros((n_data*2, 50*n))
for k in range(n_data*2):
    for m in range(18):
        for l in range(50):
            dos_format[k,m*50+l] = dos_all_new[k,l,m] 

# add "noise" to data
def rescale_dos(x,d1,d2):
    return x*(1 + d1 * np.cos(d2*x))

# enhance dataset without new calculations
data_all = np.zeros((20*n_data,900))
original_images = n_data*2
new_dos = np.zeros((10,900))
for k in range(original_images):
    data_all[k,:] = dos_format[k,:]
    for l in range(10):       
        # rescaling
        d1 = np.random.uniform(.07, .2)  # amplitude
        d2 = (2*np.pi * np.random.uniform(1, 5) ) / 2 # 2.5  # phase
        for i in range(18):
            for j in range(50):
                    # new_dos[l,i*100 + j] = rescale_dos(output_save[k,(i*100+j)],d1,d2)
                    new_dos[l,i*50 + j] = rescale_dos(dos_format[k,(i*50+j)],d1,d2)    
    data_all[original_images + k, : ] = new_dos[0,:]
    data_all[original_images*2 + k, : ] = new_dos[1,:]
    data_all[original_images*3 + k, : ] = new_dos[2,:]
    data_all[original_images*4 + k, : ] = new_dos[3,:]
    data_all[original_images*5 + k, : ] = new_dos[4,:]    
    data_all[original_images*6 + k, : ] = new_dos[5,:]
    data_all[original_images*7 + k, : ] = new_dos[6,:]
    data_all[original_images*8 + k, : ] = new_dos[7,:]
    data_all[original_images*9 + k, : ] = new_dos[8,:]

x_train_all = scaler.fit_transform(data_all)  
x_train = x_train_all[:, :]

# enhance labels
l_train = np.zeros((n_data*20,2))
for i in range(original_images):
    l_train[i,0:2] = label_all[i,0:2]
    l_train[original_images+i,0:2] = label_all[i,0:2]
    l_train[original_images*2+i,0:2] = label_all[i,0:2]
    l_train[original_images*3+i,0:2] = label_all[i,0:2]
    l_train[original_images*4+i,0:2] = label_all[i,0:2]
    l_train[original_images*5+i,0:2] = label_all[i,0:2]
    l_train[original_images*6+i,0:2] = label_all[i,0:2]
    l_train[original_images*7+i,0:2] = label_all[i,0:2]
    l_train[original_images*8+i,0:2] = label_all[i,0:2]
    l_train[original_images*9+i,0:2] = label_all[i,0:2]

l_train_scale = l_train[:,:] 
l_train_scale_1 = scaler_alpha.fit_transform(l_train_scale[:,0:1]) 
l_train_scale_2 = scaler_beta.fit_transform(l_train_scale[:,1:]) 
l_train_scale = np.concatenate((l_train_scale_1,l_train_scale_2), axis=1)


#------------------------------------------------------------------------------
# define  CcGAN  
#------------------------------------------------------------------------------

def build_gan(generator, discriminator, name="gan"):
    # building the cGAN
    yfake = Activation("linear", name="yfake")(discriminator(generator(generator.inputs)))
    yreal = Activation("linear", name="yreal")(discriminator(discriminator.inputs))
    model = Model(generator.inputs + discriminator.inputs, [yfake, yreal], name=name)
    return model

def disc(image_dim, label_dim, layer_dim=4048, reg=lambda: l1_l2(1e-5, 1e-5)):
    # discriminator
    x      = (Input(shape=(image_dim,)))
    label  = (Input(shape=(label_dim,)))
    inputs = (Concatenate(name='input_concatenation'))([x, label])
    a = (GaussianNoise(0.033))(inputs)
    a = (Dense(layer_dim, kernel_regularizer=reg()))(inputs)
    a = (LeakyReLU(0.2))(a)
    a = (Dense(int(layer_dim / 2), kernel_regularizer=reg()))(a)
    a = (LeakyReLU(0.2))(a)
    a = (Dense(1, kernel_regularizer=reg()))(a)
    a = (Activation('sigmoid'))(a)
    model = Model(inputs=[x, label], outputs=a, name="discriminator")
    return model
 
def gen(noise_dim, label_dim, image_dim, layer_dim=4048, reg=lambda: l1_l2(1e-5, 1e-5)):
    # generator
    z      = (Input(shape=(noise_dim,)))
    label  = (Input(shape=(label_dim,)))
    inputs = (Concatenate(name='input_concatenation'))([z, label])
    a = (Dense(int(layer_dim / 2), kernel_regularizer=reg()))(inputs)
    a = (LeakyReLU(0.2))(a)
    a = (Dense(int(layer_dim), kernel_regularizer=reg()))(a)
    a = (LeakyReLU(0.2))(a)
    a = (Dense(np.prod(image_dim), kernel_regularizer=reg()))(a)
    a = (Activation(activations.tanh))(a)    
    model = Model(inputs=[z, label], outputs=[a, label], name="generator")
    return model

def make_trainable(net, val):
    # change trainability status of individual networks
    net.trainable = val
    for l in net.layers:
        l.trainable = val


# ------------------------------------------------------------------------------
# create network
# ------------------------------------------------------------------------------

# create generator 
generator_S1 = gen(noise_dim, label_dim, image_dim)
adam = Adam(lr=0.001, beta_1=0.5)
generator_S1.compile(loss='mean_squared_error', optimizer=adam)    

# create discriminator 
discriminator_S1 = disc(image_dim, label_dim)
discriminator_S1.compile(loss='binary_crossentropy', optimizer='SGD')

# create CcGAN
make_trainable(discriminator_S1, False)
gan_S1 = build_gan(generator_S1, discriminator_S1)
gan_S1.compile(loss='binary_crossentropy', optimizer=adam)


# ------------------------------------------------------------------------------
# training (if already trained --> skip or load model in the next step)
# ------------------------------------------------------------------------------

# training parameters
batch_size  = 100
num_batches = int(x_train.shape[0] / batch_size)
num_epochs  = 10

for epoch in range(num_epochs):
    print("Epoch {}/{}".format(epoch + 1, num_epochs))
    for index in range(num_batches):
        # train discriminator
        make_trainable(discriminator_S1, True)
        # train discriminator on real data
        batch       = np.random.randint(0, x_train.shape[0], size=batch_size)
        image_batch = x_train[batch]
        label_batch = l_train_scale[batch]
        y_real      = np.ones(batch_size) + 0.1 * np.random.uniform(-1, 1, size=batch_size)
        discriminator_S1.train_on_batch([image_batch, label_batch], y_real)
        # train discriminator on fake data
        noise_batch      = np.random.normal(0, 1, (batch_size, noise_dim))    
        generated_images = generator_S1.predict([noise_batch, label_batch])
        y_fake           = np.zeros(batch_size) + 0.1 * np.random.uniform(0, 1, size=batch_size)
        d_loss = discriminator_S1.train_on_batch(generated_images, y_fake)  
        make_trainable(discriminator_S1, False)
        # train GAN
        gan_loss = gan_S1.train_on_batch([noise_batch, label_batch, image_batch, label_batch], [y_real, y_fake])
        print("Batch {}/{}: Discriminator loss = {}, GAN loss = {}".format(index + 1, num_batches, d_loss, gan_loss))


# ------------------------------------------------------------------------------
# saving & loading the model
# ------------------------------------------------------------------------------

# generator_S1.save_weights('generator_CcGAN_MB_18_S1.h5')
# discriminator_S1.save_weights('discriminator_CcGAN_MB_18_S1.h5')
# gan_S1.save_weights('gan_CcGAN_MB_18_S1.h5')

# generator_S1.load_weights('generator_CcGAN_MB_18_S1.h5')
# discriminator_S1.load_weights('discriminator_CcGAN_MB_18_S1.h5')
# gan_S1.load_weights('gan_CcGAN_MB_18_S1.h5')


# ------------------------------------------------------------------------------
# Generate samples (generator)
# ------------------------------------------------------------------------------

# initilize sampling 
noise_batch = np.random.normal(0, 1, (batch_size, noise_dim)) 
noise_batch[0] = np.random.uniform(0, 1, (1, noise_dim))
test_label_batch = np.zeros((batch_size,2))

# conditional parameter
test_label_batch[0,0] = 0.0  # insert here N_y scaled in [0.0, 1.0]  
test_label_batch[0,1] = 0.0  # insert here B_y scaled in [0.0, 1.0] 

# make new prediction
pred = generator_S1.predict([noise_batch, test_label_batch])[0]
pred_scale = scaler.inverse_transform(pred)

# print real conditional parameter values
N_real = scaler_alpha.inverse_transform(test_label_batch)
print('conditional parameter:')
print('N_y =',  N_real[0,0])
B_real = scaler_beta.inverse_transform(test_label_batch)
print('B_x =', B_real[0,1])

# 3D plot
d3_plot_gan = np.zeros((omega_dim,18))
for j in range(18):
    for i in range(omega_dim):
        d3_plot_gan[i,j]=pred_scale[0,j*omega_dim+i]

matplotlib.rcParams['font.family'] = "Bitstream Vera Serif"
fig = plt.figure()
fig.subplots_adjust(0.2,0.2)
plt.contourf(xs,ys_spin,d3_plot_gan,100)
plt.ylabel("frequency [J]")
plt.xlabel("Site")
plt.show()

#------------------------------------------------------------------------------
#  GAN architecture
#------------------------------------------------------------------------------
generator_S1.summary()
discriminator_S1.summary()
gan_S1.summary()
