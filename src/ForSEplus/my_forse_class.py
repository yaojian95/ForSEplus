from .nn_tools import load_training_set
import re

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import UpSampling2D, Conv2D, Activation, BatchNormalization
from tensorflow.keras.layers import Reshape, Dense, Input
from tensorflow.keras.layers import LeakyReLU, Dropout, Flatten, ZeroPadding2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import losses
import numpy as np
import os
from tensorflow.keras import backend as K

import threading
from datetime import datetime
import time

class forse_my(object):
    '''
    forse of tensorflow version
    '''
    
    def __init__(self, model_dir):
        self.dirs = model_dir
        self.img_shape = (320, 320, 1); 
        self.kernel_size = 5; 
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True); 
        self.channels = 1;
        
    def build_generator(self):

        model = Sequential()
        model.add(Conv2D(64, kernel_size=self.kernel_size, padding="same", input_shape=self.img_shape)) # 64x64x64
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.5))
        model.add(Conv2D(128, kernel_size=self.kernel_size, padding="same", strides=2)) #32x32x128
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.5))
        model.add(Conv2D(256, kernel_size=self.kernel_size, padding="same", strides=2)) #16x16x256
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.5))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=self.kernel_size, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.5))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=self.kernel_size, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.5))
        model.add(Conv2D(self.channels, kernel_size=self.kernel_size, padding="same"))
        model.add(Activation("tanh"))
        # img_in = Input(shape=img_shape)
        # img_out = model(img_in)
        # return Model(img_in, img_out)
        return model

    def build_discriminator(self):

        model = Sequential()
        model.add(Conv2D(64, kernel_size=self.kernel_size, strides=1, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.5))
        model.add(Conv2D(128, kernel_size=self.kernel_size, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.5))
        model.add(Conv2D(256, kernel_size=self.kernel_size, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Flatten())
        model.add(Dropout(0.25))
        model.add(Dense(1, activation='sigmoid'))
        # img = Input(shape=img_shape)
        # validity = model(img)
        # return Model(img, validity)
        return model


    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)
    
    def build_gan(self):

        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

        self.generator_optimizer = tf.keras.optimizers.Adam(0.00005, 0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(0.00005, 0.5)
    
    def check_point(self, best_epoch_dir = False):

        checkpoint_dir = self.dirs + 'training_checkpoints'
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer, discriminator_optimizer=self.discriminator_optimizer, generator=self.generator, discriminator=self.discriminator)
        
        ### recover from the best_epoch
        if best_epoch_dir:
            self.checkpoint.restore( best_epoch_dir).expect_partial()
            print('restore from checkpoint:%s'% best_epoch_dir)
            return;
            
        ###  or recover from the latest training epoch if best_epoch non-stated
        self.latest = tf.train.latest_checkpoint(checkpoint_dir);
        if self.latest is not None:
            self.epoch_latest = np.int(re.findall(r'\d+', self.latest)[-1])*500
            print('restore from checkpoint:%s'% self.latest)
            self.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    
    @tf.function
    def train_step(self, noise, images):
        # noise = tf.random.normal([BATCH_SIZE, noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
    
    def train(self, epochs, patches_file, batch_size=32, save_interval=100, seed=4324):
        start = time.time()
        
        self.build_gan()
        self.check_point()
        
        X_train, X_test, Y_train, Y_test = load_training_set(patches_file, part_train = 1, part_test = 0, seed=seed); # X-input large scales; Y-real small scales
        print("Training Data Shape: ", X_train.shape)
        half_batch = batch_size // 2
        accs = []

        if self.latest:
            epochs_range = range(self.epoch_latest, epochs)
            np.random.randint(0, X_train.shape[0], batch_size*(self.epoch_latest)) #To let the random start from lastest state, not the 0-state, which is state 4324
        else:
            epochs_range = range(epochs)

        for epoch in epochs_range:    
            now = time.time()

            ind_batch = np.random.randint(0, X_train.shape[0], batch_size)

            idxX = np.random.randint(0, X_train.shape[0], half_batch)
            idxY = np.random.randint(0, X_train.shape[0], half_batch)

            self.train_step(X_train[ind_batch], Y_train[idxY])
            if (epoch + 1) % save_interval == 0:
                self.checkpoint.save(file_prefix = self.checkpoint_prefix)
                message = 'You are at epoch %s ! Time cost is %0.2f mins! ETA: %0.2f hours!'%(epoch, (now-start)/60, (epochs - epoch)*(now-start)/60/60/epoch)
                print(message)