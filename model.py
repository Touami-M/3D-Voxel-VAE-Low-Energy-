import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv3D, Conv3DTranspose, BatchNormalization, ReLU, Flatten, Reshape, Lambda, Dense
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    
    epsilon = K.random_normal(shape=(batch, dim))
    
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def build_vae(input_shape, latent_dim=32, learning_rate=0.00001):
    # Encoder
    inputs = Input(shape=input_shape)
    
    x = Conv3D(16, kernel_size=4, strides=1, padding='valid')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    x = Conv3D(32, kernel_size=4, strides=1, padding='valid')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    x = Conv3D(64, kernel_size=4, strides=1, padding='valid')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    x = Conv3D(128, kernel_size=4, strides=1, padding='valid')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    x = Flatten()(x)
    
    z_mean = Dense(latent_dim)(x)
    z_log_var = Dense(latent_dim)(x)

    # Sampling
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    #  encoder model
    encoder = Model(inputs, [z_mean, z_log_var, z])

    # Decoder
    latent_inputs = Input(shape=(latent_dim,))
    x = Dense(128 * 10 * 4 * 23)(latent_inputs)  #  based on the final flattened shape from the encoder
    x = Reshape((10, 4, 23, 128))(x)  # also
    
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    
    x = Conv3DTranspose(64, kernel_size=4, strides=1, padding='valid')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    x = Conv3DTranspose(32, kernel_size=4, strides=1, padding='valid')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    x = Conv3DTranspose(16, kernel_size=4, strides=1, padding='valid')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    outputs = Conv3DTranspose(input_shape[3], kernel_size=4, strides=1, padding='valid', activation='sigmoid')(x)

    #  decoder model
    decoder = Model(latent_inputs, outputs)

    # VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs)

    optimizer = Adam(learning_rate=0.00001)

    # Loss
    reconstruction_loss = tf.reduce_mean(tf.square(inputs - outputs))
    kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
    vae_loss = reconstruction_loss + kl_loss
    vae.add_loss(vae_loss)
    vae.add_metric(reconstruction_loss, name='reconstruction_loss', aggregation='mean')
    vae.add_metric(kl_loss, name='kl_loss', aggregation='mean')
    vae.compile(optimizer=optimizer)

    return vae, encoder, decoder

input_shape = (22, 16, 35, 1)  
vae, encoder, decoder = build_vae(input_shape, latent_dim=32, learning_rate=0.00001)

def train_vae(vae, data, epochs=100, batch_size=8):
    history = vae.fit(data, data, epochs=epochs, batch_size=batch_size, validation_split=0.15)
    return history
