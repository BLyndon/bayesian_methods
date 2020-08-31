import tensorflow as tf
print("TensorFlow Version: " + str(tf.__version__))

from IPython import display

from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import metrics

from vae_plotter import Plotter

import time

class Sampling(Layer):
  ''' Reparametrization Trick
  The the speed of lerning is enormously increased by a reparametrization, that is done here.

  Inputs
  inputs:     tf.tensors of for the mean and the log variance. Used to define a normal distribution
              where a new z is sampled from.
  '''
  def call(self, inputs):
    mean, log_var = inputs
    eps = tf.random.normal(shape=tf.shape(mean))
    return mean + tf.multiply(tf.exp(0.5 * log_var), eps)

class VAE(Model, Plotter):
    ''' Variational Autoencoder
    '''
    def __init__(self, encoder=None, decoder=None, loss_fn=None):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.loss_fn = loss_fn
        self.optimizer = tf.keras.optimizers.Adam()

        self.train_summary_writer = None
        self.writers_summary_writer = None

        self.init_metrics()

    def encode(self, x):
        mean, log_var = self.encoder(x)
        z = Sampling()([mean, log_var])
        return z, mean, log_var

    def decode(self, z, logits=True):
        x_logit = self.decoder(z)
        if logits:
            return x_logit
        return tf.math.sigmoid(x_logit)

    def feedforward(self, x, logits=True):
        z, mean, log_var = self.encode(x)
        x_out = self.decode(z, logits=logits)
        return x_out, mean, log_var

    def init_metrics(self):
        print()
        print("Initialize Metrics:")
        self.history = []
        self.metrics_ = {}
        self.metrics_['train_loss'] = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        self.metrics_['train_accuracy'] = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
        self.metrics_['test_loss'] = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
        self.metrics_['test_accuracy'] = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')
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
            x_out, mean, log_var = self.feedforward(x, logits=True)
            loss = self.loss_fn(x, x_out, mean, log_var)
        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.metrics_['train_loss'](-loss)
        self.metrics_['train_accuracy'](x, x_out)

    def fit(self, train_ds, test_ds, epochs=20, lr=1e-4, save=True):
        print()
        print("Set Adam optimizer")
        self.optimizer = tf.keras.optimizers.Adam(lr)

        num_samples = 16
        for test_batch, _ in test_ds.take(1):
            test_sample = test_batch[0:num_samples, :, :, :]

        print('Start Training')
        for epoch in range(1, epochs+1):
            start_time = time.time()
            for train_x, _ in train_ds:
                self.train_step(train_x, self.optimizer)      
            end_time = time.time()
            if self.train_summary_writer.as_default!=None:
                with self.train_summary_writer.as_default():
                    tf.summary.scalar('loss', self.metrics_['train_loss'].result(), step=epoch)
                    tf.summary.scalar('accuracy', self.metrics_['train_accuracy'].result(), step=epoch)
            for test_x, _ in test_ds:
                x_out, mean, log_var = self.feedforward(test_x, logits=True)
                loss = self.loss_fn(test_x, x_out, mean, log_var)
                self.metrics_['test_loss'](-loss)
                self.metrics_['test_accuracy'](test_x, x_out)
            if self.test_summary_writer.as_default!=None:
                with self.test_summary_writer.as_default():
                    tf.summary.scalar('loss', self.metrics_['test_loss'].result(), step=epoch)
                    tf.summary.scalar('accuracy', self.metrics_['test_accuracy'].result(), step=epoch)

            display.clear_output(wait=False)
            template = 'Epoch {}, Loss: {:.2f}, Accuracy: {:.2f}, Test Loss: {:.2f}, Test Accuracy: {:.2f}, Time: {:.2f}'
            print(template.format(epoch,
                        self.metrics_['train_loss'].result(), 
                        self.metrics_['train_accuracy'].result()*100,
                        self.metrics_['test_loss'].result(), 
                        self.metrics_['test_accuracy'].result()*100,
                        end_time - start_time))
            self.history.append([self.metrics_['train_loss'].result(), self.metrics_['train_accuracy'].result()*100, self.metrics_['test_loss'].result(), self.metrics_['test_accuracy'].result()*100])
            if save==True:
                self.generate_plot_write(epoch, test_sample)
                print('Save and Plot')
            self.reset_metrics()