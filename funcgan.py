import datetime
import os
import tensorflow as tf
from keras_preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, SpatialDropout2D
from tensorflow.keras.layers import BatchNormalization, Activation, Concatenate, concatenate, GaussianDropout
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.losses import BinaryCrossentropy

# set_session(session)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib.pyplot as plt
import numpy as np

tf.config.optimizer.set_jit(True)


class DCGAN():
    def __init__(self, dataset):
        # Input shape
        self.global_total = 0
        self.img_rows = 64
        self.img_cols = 64
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100
        self.dataset = dataset
        bce_logits = BinaryCrossentropy(from_logits=True, )

        optimizer_D = Nadam(0.0001, 0.5)
        optimizer_G = Nadam(0.0001, 0.5)
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        # self.discriminator = multi_gpu_model(self.discriminator, gpus=1)
        self.discriminator.summary()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer_D,
                                   metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()
        self.generator.summary()
        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        # self.combined = multi_gpu_model(self.combined, gpus=1)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer_G, metrics=['accuracy'])

    def build_generator(self):
        outputs = []

        z_in = tf.keras.Input(shape=(self.latent_dim,))
        x = Dense(2 * 2 * 256)(z_in)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Reshape((2, 2, 256))(x)

        for i in range(5):
            if i == 0:
                x = Conv2DTranspose(128, (4, 4), strides=(2, 2),
                                    padding='same')(x)
                x = BatchNormalization()(x)
                x = LeakyReLU()(x)
            else:
                x = Conv2DTranspose(128, (4, 4), strides=(2, 2),
                                    padding='same')(x)
                x = BatchNormalization()(x)
                x = LeakyReLU()(x)

            x = Conv2D(64, (5, 5), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = LeakyReLU()(x)

            outputs.append(Conv2DTranspose(3, (4, 4), strides=(1, 1),
                                           padding='same', activation='tanh')(x))

        model = tf.keras.Model(inputs=z_in, outputs=outputs)
        return model

    def build_discriminator(self):

        inputs = [
            Input(shape=(64, 64, 3)),
            Input(shape=(32, 32, 3)),
            Input(shape=(16, 16, 3)),
            Input(shape=(8, 8, 3)),
            Input(shape=(4, 4, 3)),

        ]

        x = None
        for image_in in inputs:
            if x is None:
                # for the first input we don't have features to append to
                x = Conv2D(64, (5, 5), strides=(2, 2), padding='same')(image_in)
                x = LeakyReLU()(x)
                x = Dropout(0.3)(x)
            else:
                # every additional input gets its own conv layer then appended
                x = Conv2D(64, (5, 5), strides=(2, 2), padding='same')(x)
                x = LeakyReLU()(x)
                x = Dropout(0.3)(x)
                y = Conv2D(64, (5, 5), strides=(2, 2), padding='same')(image_in)
                y = LeakyReLU()(y)
                y = Dropout(0.3)(y)
                x = concatenate([x, y])

            x = Conv2D(128, (5, 5), strides=(1, 1), padding='same')(x)
            x = LeakyReLU()(x)
            x = Dropout(0.3)(x)

            x = Conv2D(256, (5, 5), strides=(1, 1), padding='same')(x)
            x = LeakyReLU()(x)
            x = Dropout(0.3)(x)

        x = Flatten()(x)
        out = Dense(1)(x)
        inputs = inputs[::-1]  # reorder the list to be smallest resolution first
        model = tf.keras.Model(inputs=inputs, outputs=out)
        return model

    def train(self, epochs, batch_size=128, save_interval=50, X_train=None, global_total=0, dataset=None):
        input("Errthing good?")
        # # Adversarial ground truths
        # valid = np.ones((batch_size, 1))
        # fake = np.zeros((batch_size, 1))
        d_race_counter = 0
        g_race_counter = 0
        d_race = 1
        g_race = 1
        dataset = dataset
        for epoch in range(epochs):
            for batch in dataset:
                for _ in range(d_race):
                    # ---------------------
                    #  Train Discriminator
                    # ---------------------

                    noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                    gen_imgs = self.generator.predict(noise)

                    # Train the discriminator (real classified as ones and generated as zeros)
                    d_loss_real = self.discriminator.train_on_batch(batch, np.ones((batch_size, 1)))
                    d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((batch_size, 1)))
                    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                    # if(int(d_loss_fake[1] * 100) > 90):
                    #     d_race_counter = d_race_counter + 1

                # ---------------------
                #  Train Generator
                # ---------------------
                if d_race_counter > 100:
                    d_race_counter = 0
                    d_race = 1
                    g_race = 2

                for idx in range(g_race):
                    # Train the generator (wants discriminator to mistake images as real)
                    g_loss = self.combined.train_on_batch(noise, np.ones((batch_size, 1)))
                    # if(int(g_loss[1] * 100) > 95):
                    # if idx > 0:
                    #     noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                    # if(g_loss[1] * 100 > 70):
                    #     g_race_counter = g_race_counter + 1

                if g_race_counter > 50:
                    g_race_counter = 0
                    d_race = 2
                    g_race = 1

                # Plot the progress
                print(
                    "%d [D loss: %f, acc real: %-.2f%%, acc fake.: %.2f%% acc comb: %.2f%%] [G loss: %f G acc: %.2f%%] [G: %d Dn: %d]" %
                    (self.global_total, d_loss[0], (100 * d_loss_real[1]), 100 * d_loss_fake[1], (100 * d_loss[1]),
                     g_loss[0], g_loss[1] * 100, g_race, d_race))
                self.global_total = self.global_total + 1
                # If at save interval => save generated image samples
                if epoch % save_interval == 0:
                    sample = self.generator.predict(noise)
                    self.sample_images(self.global_total, sample)

        # use matplotlib to plot a given tensor sample

    def sample_images(self, epoch, sample):
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        num_samples = 4
        # sample = np.array([i for i in sample])
        grid = gridspec.GridSpec(5, 4)
        grid.update(left=0, bottom=0, top=1, right=1, wspace=0.01, hspace=0.01)
        fig = plt.figure(figsize=[8, 10])
        # print(sample.shape)
        for x in range(5):
            # images = np.squeeze(images)
            for y in range(num_samples):
                ax = fig.add_subplot(grid[x, y])
                ax.set_axis_off()
                ax.imshow((sample[x][y] + 1.0) / 2)
        fig.savefig(("imags/skippedouts/gan_%s." + 'png') % int(epoch / 2), format='png')
        plt.close(fig)
        # plt.show()

def preprocessData():
    def image_reshape(x):
        return [
            tf.image.resize(x, (4, 4)),
            tf.image.resize(x, (8, 8)),
            tf.image.resize(x, (16, 16)),
            tf.image.resize(x, (32, 32)),
            x
        ]

    input("Process IMG? (Enter)")
    new_images = []
    root_dir = os.walk('image/')
    counter = 0
    dataset = None
    def plot_sample(epoch, sample):
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        num_samples = 4
        sample = sample[::-1]
        grid = gridspec.GridSpec(7, 4)
        grid.update(left=0, bottom=0, top=1, right=1, wspace=0.01, hspace=0.01)
        fig = plt.figure(figsize=[4, 7])
        for x in range(7):
            images = sample[x] # .numpy()  # this converts the tensor to a numpy array
            # images = np.squeeze(images)
            for y in range(num_samples):
                ax = fig.add_subplot(grid[x, y])
                ax.set_axis_off()
                #images = resolution, dim_x, dim_y, dim_z, RGB
                #images[x] = batch_size, dim_x, dim_y, dim_z, RGB
                #images[x][y] = dim_x, dim_y, dim_z, RGB
                ax.imshow((images[x][y].astype(np.int)))
        # fig.savefig(("imags/skippedouts/gan_%s." + 'svg') % int(epoch / 100), format='svg')
        plt.show()
    for path, subdirs, files in root_dir:
        # print(subdirs)
        for name in files:
            if name.endswith('.jpg'):
                print(counter)
                counter = counter + 1
                new_images.append(np.array([img_to_array(load_img(os.path.join(path, name), target_size=(64, 64), interpolation='lanczos'))]))
                # if counter % 500 == 0 or counter == 1:
                #     plot_sample(5421, new_images[counter-1])
                k = 0
                if counter == 1000:
                    imags = np.array(new_images)
                    imags = np.squeeze(imags)
                    dataset = tf.data.Dataset.from_tensor_slices(imags)
                    dataset = dataset.map(image_reshape)
                    dataset = dataset.cache()
                    dataset = dataset.shuffle(len(imags))
                    dataset = dataset.batch(32, drop_remainder=True)
                    dataset = dataset.prefetch(1)
                    return dataset
                    #print(bruh.shape)
                    #np.save("img_npy_arr_lanczos64_skip_connections.npy", bruh)
                    #exit()

if __name__ == '__main__':


    # bruh = np.array(new_images, dtype=np.float16)
    # np.save("img_npy_arr_lanczos64_skip_connections.npy", bruh)
    # exit(1)
    # Rescale -1 to 1
    # 136725
    dataset = preprocessData()
    dcgan = DCGAN(dataset)
    # exit(1)
    input("Cont?")
    X_train = np.load("img_npy_arr_lanczos64_skip_connections.npy", allow_pickle=True)
    X_train = X_train / 127.5 - 1.
    global_total = 0
    new = 1
    while new != 0:
        # new = int(input("Epochs?: "))
        # bsize = int(input("Batch size "))
        dcgan.train(epochs=100, batch_size=32, save_interval=2, X_train=X_train, global_total=global_total, dataset=dataset)
