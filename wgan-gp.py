from __future__ import print_function, division

import os
import tensorflow as tf
from keras.layers.merge import _Merge
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Conv2DTranspose, SpatialDropout2D, GaussianNoise
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from functools import partial
from keras.constraints import max_norm

import keras.backend as K

import matplotlib.pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys

import numpy as np
from keras_preprocessing.image import img_to_array, load_img
# tf.config.optimizer.set_jit(True)
tf.config.optimizer.set_experimental_options(options={'arithmetic_optimizatio':True, 'shape_optimization':True})

class RandomWeightedAverage(_Merge):
    def _merge_function(self, inputs):
        alpha = K.random_uniform((32, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

class WGANGP():
    def __init__(self):
        self.img_rows = 32
        self.img_cols = 32
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        optimizer = RMSprop(lr=0.00005)

        # Build the generator and critic
        self.generator = self.build_generator()
        self.critic = self.build_critic()

        #-------------------------------
        # Construct Computational Graph
        #       for the Critic
        #-------------------------------

        # Freeze generator's layers while training critic
        self.generator.trainable = False

        # Image input (real sample)
        real_img = Input(shape=self.img_shape)

        # Noise input
        z_disc = Input(shape=(self.latent_dim,))
        # Generate image based of noise (fake sample)
        fake_img = self.generator(z_disc)

        # Discriminator determines validity of the real and fake images
        fake = self.critic(fake_img)
        valid = self.critic(real_img)

        # Construct weighted average between real and fake images
        interpolated_img = RandomWeightedAverage()([real_img, fake_img])
        # Determine validity of weighted sample
        validity_interpolated = self.critic(interpolated_img)

        # Use Python partial to provide loss function with additional
        # 'averaged_samples' argument
        partial_gp_loss = partial(self.gradient_penalty_loss,
                          averaged_samples=interpolated_img)
        partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names

        self.critic_model = Model(inputs=[real_img, z_disc],
                            outputs=[valid, fake, validity_interpolated])
        self.critic_model.compile(loss=[self.wasserstein_loss,
                                              self.wasserstein_loss,
                                              partial_gp_loss],
                                        optimizer=optimizer,
                                        loss_weights=[1, 1, 10])
        #-------------------------------
        # Construct Computational Graph
        #         for Generator
        #-------------------------------

        # For the generator we freeze the critic's layers
        self.critic.trainable = False
        self.generator.trainable = True

        # Sampled noise for input to generator
        z_gen = Input(shape=(self.latent_dim,))
        # Generate images based of noise
        img = self.generator(z_gen)
        # Discriminator determines validity
        valid = self.critic(img)
        # Defines generator model
        self.generator_model = Model(z_gen, valid)
        self.generator_model.compile(loss=self.wasserstein_loss, optimizer=optimizer)


    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        gradients = K.gradients(y_pred, averaged_samples)[0]
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)


    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(256 * 2 * 2, activation="relu", input_dim=self.latent_dim, use_bias=False))
        model.add(Reshape((2, 2, 256)))
        # print(model.output_shape)
        model.add(Conv2DTranspose(filters=256, kernel_size=(2, 2),
                                  strides=(2, 2),
                                  padding='same',
                                  name="FIRST_UP_CONV"))
        model.add(BatchNormalization(momentum=0.8, name="B1"))
        model.add(Activation("relu", name="A1"))

        model.add(Conv2DTranspose(filters=128, kernel_size=(4, 4),
                                  strides=(2, 2),
                                  padding='same',
                                  name="SECOND_UP_CONV"))
        model.add(BatchNormalization(momentum=0.8, name="B2"))
        model.add(Activation("relu", name="A2"))

        model.add(Conv2DTranspose(filters=64, kernel_size=(4, 4),
                                  strides=(2, 2),
                                  padding='same',
                                  name="THIRD_UP_CONV"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu", name="A4"))

        model.add(Conv2DTranspose(filters=32, kernel_size=(4, 4),
                                  strides=(2, 2),
                                  padding='same',
                                  name="Fourth_UP_CONV"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu", name="A5"))

        model.add(Conv2DTranspose(filters=16, kernel_size=(4, 4),
                                  strides=(1, 1),
                                  padding='same',
                                  name="Fifth_UP_CONV"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu", name="A6"))

        model.add(Dense(3, activation='tanh', kernel_initializer='glorot_uniform'))

        model.summary()
        # print(model.input.dtype)

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_critic(self):
        # const = self.ClipConstraint(0.01)
        model = Sequential()

        model.add(Conv2D(16, kernel_size=4, input_shape=self.img_shape))
        model.add(GaussianNoise(1))
        model.add(LeakyReLU())
        model.add(BatchNormalization(momentum=0.8))
        model.add(SpatialDropout2D(0.25))

        model.add(Conv2D(32, kernel_size=4, padding='same', strides=2))
        model.add(GaussianNoise(1))
        # model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU())
        model.add(SpatialDropout2D(0.25))

        model.add(Conv2D(64, kernel_size=4, padding='same', strides=2))
        # model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU())
        model.add(SpatialDropout2D(0.25))

        model.add(Conv2D(64, kernel_size=4, padding='same', strides=2))
        model.add(GaussianNoise(1))
        # model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU())
        model.add(SpatialDropout2D(0.25))

        model.add(Conv2D(128, kernel_size=4, padding='same', strides=2))
        model.add(GaussianNoise(1))
        # model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU())
        model.add(SpatialDropout2D(0.25))

        model.add(Conv2D(256, kernel_size=4, padding='same', strides=1))
        model.add(GaussianNoise(1))
        # model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU())
        model.add(SpatialDropout2D(0.25))

        model.add(Flatten())

        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)
    # def build_generator(self):
    #
    #     model = Sequential()
    #
    #     model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim))
    #     model.add(Reshape((7, 7, 128)))
    #     model.add(UpSampling2D())
    #     model.add(Conv2D(128, kernel_size=4, padding="same"))
    #     model.add(BatchNormalization(momentum=0.8))
    #     model.add(Activation("relu"))
    #     model.add(UpSampling2D())
    #     model.add(Conv2D(64, kernel_size=4, padding="same"))
    #     model.add(BatchNormalization(momentum=0.8))
    #     model.add(Activation("relu"))
    #     model.add(Conv2D(self.channels, kernel_size=4, padding="same"))
    #     model.add(Activation("tanh"))
    #
    #     model.summary()
    #
    #     noise = Input(shape=(self.latent_dim,))
    #     img = model(noise)
    #
    #     return Model(noise, img)
    #
    # def build_critic(self):
    #
    #     model = Sequential()
    #
    #     model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
    #     model.add(LeakyReLU(alpha=0.2))
    #     model.add(Dropout(0.25))
    #     model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
    #     model.add(ZeroPadding2D(padding=((0,1),(0,1))))
    #     model.add(BatchNormalization(momentum=0.8))
    #     model.add(LeakyReLU(alpha=0.2))
    #     model.add(Dropout(0.25))
    #     model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    #     model.add(BatchNormalization(momentum=0.8))
    #     model.add(LeakyReLU(alpha=0.2))
    #     model.add(Dropout(0.25))
    #     model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
    #     model.add(BatchNormalization(momentum=0.8))
    #     model.add(LeakyReLU(alpha=0.2))
    #     model.add(Dropout(0.25))
    #     model.add(Flatten())
    #     model.add(Dense(1))
    #
    #     model.summary()
    #
    #     img = Input(shape=self.img_shape)
    #     validity = model(img)
    #
    #     return Model(img, validity)

    def train(self, epochs, batch_size, sample_interval=50, xT=None):

        # Load the dataset
        X_train = xT

        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))
        dummy = np.zeros((batch_size, 1)) # Dummy gt for gradient penalty
        for epoch in range(epochs):

            for _ in range(self.n_critic):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                imgs = X_train[idx]
                # Sample generator input
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                # Train the critic
                d_loss = self.critic_model.train_on_batch([imgs, noise],
                                                                [valid, fake, dummy])

            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.generator_model.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch, batch_size)

    def sample_images(self, epoch, batch_size):
        # with tf.device("CPU:0"):
        noise = tf.Session().run(tf.random.normal((batch_size, self.latent_dim), 0, 1, seed=1337))
        noise1 = tf.Session().run(tf.random.normal((batch_size, self.latent_dim), 0, 1, seed=99))
        noise2 = tf.Session().run(tf.random.normal((batch_size, self.latent_dim), 0, 1, seed=154124124))
        noise3 = tf.Session().run(tf.random.normal((batch_size, self.latent_dim), 0, 1, seed=1))

        # noise = tf.compat.v1.make_ndarray(noise)
        gen_imgs = self.generator.predict(noise)
        gen_imgs1 = self.generator.predict(noise1)
        gen_imgs2 = self.generator.predict(noise2)
        gen_imgs3 = self.generator.predict(noise3)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5
        gen_imgs1 = 0.5 * gen_imgs1 + 0.5
        gen_imgs2 = 0.5 * gen_imgs2 + 0.5
        gen_imgs3 = 0.5 * gen_imgs3 + 0.5

        # plt.figure()
        fig, axs = plt.subplots(4, 4)
        cnt = 0
        axs[0, 0].imshow(gen_imgs[0, :, :, :])
        axs[0, 1].imshow(gen_imgs1[0, :, :, :])
        axs[0, 2].imshow(gen_imgs2[0, :, :, :])
        axs[0, 3].imshow(gen_imgs3[0, :, :, :])
        axs[1, 0].imshow(gen_imgs[1, :, :, :])
        axs[1, 1].imshow(gen_imgs1[1, :, :, :])
        axs[1, 2].imshow(gen_imgs2[1, :, :, :])
        axs[1, 3].imshow(gen_imgs3[1, :, :, :])
        axs[2, 0].imshow(gen_imgs[2, :, :, :])
        axs[2, 1].imshow(gen_imgs1[2, :, :, :])
        axs[2, 2].imshow(gen_imgs2[2, :, :, :])
        axs[2, 3].imshow(gen_imgs3[2, :, :, :])
        axs[3, 0].imshow(gen_imgs[3, :, :, :])
        axs[3, 1].imshow(gen_imgs1[3, :, :, :])
        axs[3, 2].imshow(gen_imgs2[3, :, :, :])
        axs[3, 3].imshow(gen_imgs3[3, :, :, :])

        for i in range(4):
            for j in range(4):
                # axs[i, j].imshow(gen_imgs[i+j, :, :, :])
                axs[i, j].axis('off')
                # cnt += 1
        format = 'png'
        fig.savefig(("images/gan_%s." + str(format)) % int(epoch / 10), format=str(format))
        # plt.imsave("brownian_noise_gan_samples/noise%d.png" % epoch, concat_noise)
        plt.close()


if __name__ == '__main__':
    input("Process IMG? (Enter)")
    # new_images = []
    # root_dir = os.walk('image/')
    # counter = 0
    # for path, subdirs, files in root_dir:
    #     # print(subdirs)
    #     for name in files:
    #         if name.endswith('.jpg'):
    #             print(counter)
    #             counter = counter + 1
    #             new_images.append(img_to_array(load_img(os.path.join(path, name), target_size=(32, 32), interpolation='lanczos')))
    #         if counter == 20000:
    #             break
    #     if counter == 20000:
    #         break
    # images = np.array(new_images, dtype=np.float16)
    # np.save("img_npy_arr_lanczos64.npy", images)
    # exit(1)
    wgan = WGANGP()
    input("Cont?")
    X_train = np.load("img_npy_arr_lanczos64.npy")
    X_train = X_train / 127.5 - 1.

    wgan.train(epochs=30000, batch_size=32, sample_interval=10, xT=X_train)
