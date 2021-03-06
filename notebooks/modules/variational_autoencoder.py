import time
from IPython import display

import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Conv2D, Conv2DTranspose, Reshape, Flatten, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras import metrics
import tensorflow_probability as tfp

from vae_plotter import Plotter

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

#######################################################################################
#    ENCODER                                                                          #
#######################################################################################


class Encoder(Layer):
    '''Encoder class:

        Parameters:
        inp_shape (tuple): input shape
        latent_dim (int): Dimension of the latent space
        net (string): neural network architecture
        info (bool): print info at initialization
    '''

    def __init__(self,
                 inp_shape=(28, 28, 1),
                 latent_dim=2,
                 net='conv',
                 info=False,
                 **kwargs):
        super(Encoder, self).__init__(**kwargs)
        # Initialize Shape & Architecture
        self.inp_shape = inp_shape
        self.latent_dim = latent_dim
        self.net = net
        assert self.net == 'conv' or self.net == 'mlp', print(
            'Choose net from {"conv", "mlp"}')

        # Initialize Layers
        self.flatten = Flatten()
        if self.net == 'conv':
            self.conv2d1 = Conv2D(filters=32, kernel_size=3,
                                  strides=(2, 2), activation='relu')
            self.conv2d2 = Conv2D(filters=64, kernel_size=3,
                                  strides=(2, 2), activation='relu')
        else:
            self.dense_hidden = Dense(512, activation='relu')
        self.dense_mu = Dense(self.latent_dim, name='mu')
        self.dense_log_var = Dense(self.latent_dim, name='log_var')

        if info == True:
            self.print_info()

    def __call__(self, x):
        if self.net == 'conv':
            cx = self.conv2d1(x)
            cx = self.conv2d2(cx)
            x = self.flatten(cx)
        else:
            x = self.flatten(x)
            x = self.dense_hidden(x)
        mu = self.dense_mu(x)
        log_var = self.dense_log_var(x)
        return mu, log_var

    def print_info(self):
        print()
        print(self.net + '-Encoder')
        print(' - Input Shape: {}'.format(self.inp_shape))
        print(' - Latent Variable Space: {}'.format(self.latent_dim))


#######################################################################################
#    DECODER                                                                          #
#######################################################################################
class Decoder(Layer):
    '''Decoder class:

        Parameters:
        latent_dim (int): Dimension of the latent space
        outp_shape (tuple): input shape
        net (string): neural network architecture
        info (bool): print info at initialization
    '''

    def __init__(self,
                 latent_dim=2,
                 outp_shape=(28, 28, 1),
                 net='conv',
                 info=False,
                 **kwargs):
        super(Decoder, self).__init__(**kwargs)
        # Initialize Shape & Architecture
        self.outp_shape = outp_shape
        self.latent_dim = latent_dim
        self.net = net
        assert self.net == 'conv' or self.net == 'mlp', print(
            'Choose net from {"conv", "mlp"}')

        # Initialize Layers
        if self.net == 'conv':
            self.dense = Dense(7*7*32, activation='relu')
            self.reshape = Reshape((7, 7, 32))
            self.conv2dT1 = Conv2DTranspose(
                filters=64, kernel_size=3, strides=2, padding='same', activation='relu')
            self.conv2dT2 = Conv2DTranspose(
                filters=32, kernel_size=3, strides=2, padding='same',  activation='relu')
            self.out = Conv2DTranspose(
                filters=1, kernel_size=3, padding='same', name='output')
        else:
            digit_len = self.outp_shape[0] * \
                self.outp_shape[1]*self.outp_shape[2]
            self.dense1 = Dense(512, activation='relu')
            self.dense2 = Dense(digit_len)
            self.out = Reshape(self.outp_shape)

        # Print Encoder-Decoder Info
        if info == True:
            self.print_info()

    def __call__(self, z):
        if self.net == 'conv':
            z = self.dense(z)
            z = self.reshape(z)
            cz = self.conv2dT1(z)
            cz = self.conv2dT2(cz)
            x = self.out(cz)
        else:
            z = self.dense1(z)
            z = self.dense2(z)
            x = self.out(z)
        return x

    def print_info(self):
        print()
        print(self.net + '-Decoder')
        print(' - Latent Variable Space: {}'.format(self.latent_dim))
        print(' - Output Shape: {}'.format(self.outp_shape))


#######################################################################################
#    REPARAMETRIZATION TRICK                                                          #
#######################################################################################
class Sampling(Layer):
    ''' Reparametrization Trick:
    Apply reparametrization trick to lower the variance during training phase. Sample latent variable from normal distribution definde by the mean and the (log-)variance.

    Inputs
    inputs (list): list containing mean and the log variance
    '''

    def __call__(self, inputs):
        mean, log_var = inputs
        eps = tf.random.normal(shape=tf.shape(mean))
        return mean + tf.multiply(tf.exp(0.5 * log_var), eps)


#######################################################################################
#    VARIATIONAL AUTOENCODER                                                          #
#######################################################################################
class VAE(Model):
    ''' Variational Autoencoder

    Parameters:
    encoder (Encoder): encoder network
    decoder (Decoder): decoder network
    loss_fn (func): vae loss function
    '''

    def __init__(self, encoder, decoder, loss_fn):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        assert encoder.inp_shape == decoder.outp_shape, print(
            'Encoder input shape and decoder output shape need to be equal!')
        assert encoder.latent_dim == decoder.latent_dim, print(
            'Encoder and decoder latent space dimension need to be equal!')
        self.inp_shape = encoder.inp_shape
        self.latent_dim = encoder.latent_dim

        self.loss_fn = loss_fn

        self.print_vae_info()

        self.init_metrics()

        # Initialize Output
        self.plotter = Plotter()
        self.train_summary_writer = None
        self.test_summary_writer = None

        self.progress = 'Progress - Runtime {:.2f} s:\n'
        self.progress += 'Epoch {}/{}, '
        self.progress += 'Loss: {:.2f}, Accuracy: {:.2f}, '
        self.progress += 'Test Loss: {:.2f}, '
        self.progress += 'Test Accuracy: {:.2f}, '
        self.progress += 'Time: {:.2f} s'

    def __call__(self, x, logits=True):
        z, mean, log_var = self.encode(x)
        x_out = self.decode(z, logits=logits)
        return x_out, mean, log_var

    def print_vae_info(self):
        print()
        print('++++ VARIATIONAL AUTOENCODER ++++')
        self.encoder.print_info()
        self.decoder.print_info()

    def print_fit_info(self, epochs, lr):
        print()
        print('++++ FITTING ++++')
        print()
        print('- Epochs: {}'.format(epochs))
        print('- Learning Rate: {}'.format(lr))
        print('- Optimizer: Adam')
        print()
        print('Start Training...')

    def encode(self, x):
        mean, log_var = self.encoder(x)
        z = Sampling()([mean, log_var])
        return z, mean, log_var

    def decode(self, z, logits=True):
        x_logit = self.decoder(z)
        if logits:
            return x_logit
        return tf.math.sigmoid(x_logit)

    def init_metrics(self):
        print()
        print("Initialize Metrics:")
        self.history = []
        self.metrics_ = {}
        self.metrics_['train_loss'] = tf.keras.metrics.Mean(
            'NELBO_train', dtype=tf.float32)
        self.metrics_['train_accuracy'] = tf.keras.metrics.BinaryAccuracy(
            'BinaryAccuracy_train', threshold=0.5, dtype=tf.float32)
        self.metrics_['test_loss'] = tf.keras.metrics.Mean(
            'NELBO_test', dtype=tf.float32)
        self.metrics_['test_accuracy'] = tf.keras.metrics.BinaryAccuracy(
            'BinaryAccuracy_test', threshold=0.5, dtype=tf.float32)
        for key in self.metrics_.keys():
            print(" - " + key)

    def reset_metrics(self):
        print()
        print("Reset Metrics")
        for key in self.metrics_.keys():
            self.metrics_[key].reset_states()

    @tf.function
    def train_step(self, x, optimizer):
        with tf.GradientTape() as tape:
            x_out, mean, log_var = self.__call__(x, logits=True)
            loss = self.loss_fn(x, x_out, mean, log_var)
        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.metrics_['train_loss'](loss)
        self.metrics_['train_accuracy'].update_state(x, tf.math.sigmoid(x_out))

    def fit(self, train_ds, test_ds, epochs=20, lr=1e-4, save=True):
        self.optimizer = tf.keras.optimizers.Adam(lr)

        num_samples = 16
        for x_test_batch, _ in test_ds.take(1):
            test_sample = x_test_batch[0:num_samples, :, :, :]

        self.print_fit_info(epochs, lr)

        elapsed_time = 0
        for epoch in range(1, epochs+1):
            start_time = time.time()
            for train_x, _ in train_ds:
                self.train_step(train_x, self.optimizer)
            end_time = time.time()
            if self.train_summary_writer != None:
                with self.train_summary_writer.as_default():
                    tf.summary.scalar('loss', self.metrics_[
                                      'train_loss'].result(), step=epoch)
                    tf.summary.scalar('accuracy', self.metrics_[
                                      'train_accuracy'].result(), step=epoch)
            for test_x, _ in test_ds:
                x_out, mean, log_var = self.__call__(test_x, logits=True)
                loss = self.loss_fn(test_x, x_out, mean, log_var)
                self.metrics_['test_loss'](loss)
                self.metrics_['test_accuracy'].update_state(
                    test_x, tf.math.sigmoid(x_out))
            if self.test_summary_writer != None:
                with self.test_summary_writer.as_default():
                    tf.summary.scalar('loss', self.metrics_[
                                      'test_loss'].result(), step=epoch)
                    tf.summary.scalar('accuracy', self.metrics_[
                                      'test_accuracy'].result(), step=epoch)

            display.clear_output(wait=False)
            elapsed_time += end_time - start_time
            print(self.progress.format(elapsed_time,
                                       epoch,
                                       epochs,
                                       self.metrics_['train_loss'].result(),
                                       self.metrics_[
                                           'train_accuracy'].result()*100,
                                       self.metrics_['test_loss'].result(),
                                       self.metrics_[
                                           'test_accuracy'].result()*100,
                                       end_time - start_time))
            self.history.append([self.metrics_['train_loss'].result(), self.metrics_['train_accuracy'].result(
            )*100, self.metrics_['test_loss'].result(), self.metrics_['test_accuracy'].result()*100])

            if epoch > 1:
                self.plotter.plot_losses(self)
            self.plotter.generate_plot_write(
                self, epoch, test_sample, save=save)

            self.reset_metrics()
        print()
        print('---- FINISHED ----')


#######################################################################################
#    CONDITIONAL VARIATIONAL AUTOENCODER                                              #
#######################################################################################
class CVAE(Model):
    ''' Conditional Variational Autoencoder

    Parameters:
    encoder (Encoder): encoder network
    decoder (Decoder): decoder network
    loss_fn (func): vae loss function
    '''

    def __init__(self, encoder, decoder, loss_fn):
        super(CVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        # assert encoder.inp_shape == decoder.outp_shape, print('Encoder input shape and decoder output shape need to be equal!')
        # assert encoder.latent_dim == decoder.latent_dim, print('Encoder and decoder latent space dimension need to be equal!')
        self.inp_shape = encoder.inp_shape
        self.latent_dim = encoder.latent_dim

        self.loss_fn = loss_fn
        self.optimizer = tf.keras.optimizers.Adam()

        self.print_info()

        self.init_metrics()

        # Initialize Output
        self.plotter = Plotter()
        self.train_summary_writer = None
        self.test_summary_writer = None

        self.progress = 'Progress - Runtime {:.2f} s:\n'
        self.progress += 'Epoch {}/{}, '
        self.progress += 'Loss: {:.2f}, Accuracy: {:.2f}, '
        self.progress += 'Test Loss: {:.2f}, '
        self.progress += 'Test Accuracy: {:.2f}, '
        self.progress += 'Time: {:.2f} s'

    def __call__(self, x, y, logits=True):
        X = Flatten()(x)
        enc_inp = concatenate([X, y])
        z, mean, log_var = self.encode(enc_inp)
        dec_inp = concatenate([z, y])
        x_out = self.decode(dec_inp, logits=logits)
        return x_out, mean, log_var

    def print_info(self):
        print()
        print('++++ VARIATIONAL AUTOENCODER ++++')
        self.encoder.print_info()
        self.decoder.print_info()

    def print_fit_info(self, epochs, lr):
        print()
        print('++++ FITTING ++++')
        print()
        print('- Epochs: {}'.format(epochs))
        print('- Learning Rate: {}'.format(lr))
        print('- Optimizer: Adam')
        print()
        print('Start Training...')

    def encode(self, x):
        mean, log_var = self.encoder(x)
        z = Sampling()([mean, log_var])
        return z, mean, log_var

    def decode(self, z, logits=True):
        x_logit = self.decoder(z)
        if logits:
            return x_logit
        return tf.math.sigmoid(x_logit)

    def init_metrics(self):
        print()
        print("Initialize Metrics:")
        self.history = []
        self.metrics_ = {}
        self.metrics_['train_loss'] = tf.keras.metrics.Mean(
            'NELBO_train', dtype=tf.float32)
        self.metrics_['train_accuracy'] = tf.keras.metrics.BinaryAccuracy(
            'BinaryAccuracy_train', threshold=0.5, dtype=tf.float32)
        self.metrics_['test_loss'] = tf.keras.metrics.Mean(
            'NELBO_test', dtype=tf.float32)
        self.metrics_['test_accuracy'] = tf.keras.metrics.BinaryAccuracy(
            'BinaryAccuracy_test', threshold=0.5, dtype=tf.float32)
        for key in self.metrics_.keys():
            print(" - " + key)

    def reset_metrics(self):
        print()
        print("Reset Metrics")
        for key in self.metrics_.keys():
            self.metrics_[key].reset_states()

    @tf.function
    def train_step(self, x, y, optimizer):
        with tf.GradientTape() as tape:
            x_out, mean, log_var = self.__call__(x, y, logits=True)
            loss = self.loss_fn(x, x_out, mean, log_var)
        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.metrics_['train_loss'](loss)
        self.metrics_['train_accuracy'].update_state(x, tf.math.sigmoid(x_out))

    def fit(self, train_ds, test_ds, epochs=20, lr=1e-4, save=True):
        self.optimizer = tf.keras.optimizers.Adam(lr)

        num_samples = 16
        for x_test_batch, y_test_batch in test_ds.take(1):
            x_test_sample = x_test_batch[0:num_samples, :, :, :]
            y_test_sample = y_test_batch[0:num_samples, :]

        self.print_fit_info(epochs, lr)

        elapsed_time = 0
        for epoch in range(1, epochs+1):
            start_time = time.time()
            for train_x, train_y in train_ds:
                self.train_step(train_x, train_y, self.optimizer)
            end_time = time.time()
            if self.train_summary_writer != None:
                with self.train_summary_writer.as_default():
                    tf.summary.scalar('loss', self.metrics_[
                                      'train_loss'].result(), step=epoch)
                    tf.summary.scalar('accuracy', self.metrics_[
                                      'train_accuracy'].result(), step=epoch)
            for test_x, test_y in test_ds:
                x_out, mean, log_var = self.__call__(
                    test_x, test_y, logits=True)
                loss = self.loss_fn(test_x, x_out, mean, log_var)
                self.metrics_['test_loss'](loss)
                self.metrics_['test_accuracy'].update_state(
                    test_x, tf.math.sigmoid(x_out))
            if self.test_summary_writer != None:
                with self.test_summary_writer.as_default():
                    tf.summary.scalar('loss', self.metrics_[
                                      'test_loss'].result(), step=epoch)
                    tf.summary.scalar('accuracy', self.metrics_[
                                      'test_accuracy'].result(), step=epoch)

            display.clear_output(wait=False)
            elapsed_time += end_time - start_time
            print(self.progress.format(elapsed_time,
                                       epoch,
                                       epochs,
                                       self.metrics_['train_loss'].result(),
                                       self.metrics_[
                                           'train_accuracy'].result()*100,
                                       self.metrics_['test_loss'].result(),
                                       self.metrics_[
                                           'test_accuracy'].result()*100,
                                       end_time - start_time))
            self.history.append([self.metrics_['train_loss'].result(), self.metrics_['train_accuracy'].result(
            )*100, self.metrics_['test_loss'].result(), self.metrics_['test_accuracy'].result()*100])

            self.plotter.plot_losses(self)
            self.plotter.generate_plot_write(
                self, epoch, x_test_sample, y_test_sample, save=save)

            self.reset_metrics()
        print()
        print('---- FINISHED ----')


if __name__ == "__main__":
    pass
