from IPython import display

import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Conv2D, Conv2DTranspose, Reshape, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras import metrics
import tensorflow_probability as tfp

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

import time

#######################################################################################
#    ENCODER                                                                          #
#######################################################################################
class Encoder(Layer):
    def __init__(self,
                inp_shape=(28,28,1),
                latent_dim=2,
                net='conv',
                info=False, 
                **kwargs): 
        super(Encoder, self).__init__(**kwargs)
        # Initialize Shape & Architecture
        self.inp_shape = inp_shape
        self.latent_dim = latent_dim
        self.net = net
        assert self.net=='conv' or self.net=='mlp', print('Choose net from {"conv", "mlp"}')

        # Initialize Layers
        self.flatten = Flatten()
        if self.net=='conv':
            self.conv2d1 = Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu')
            self.conv2d2 = Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu')
        else:
            self.dense_hidden = Dense(512, activation='relu')
        self.dense_mu = Dense(self.latent_dim, name='mu')
        self.dense_log_var = Dense(self.latent_dim, name='log_var')
    
        if info==True:
            self.print_info()
    
    def __call__(self, x):
        if self.net=='conv':
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
        assert self.net=='conv' or self.net=='mlp', print('Choose net from {"conv", "mlp"}')

        # Initialize Layers
        if self.net=='conv':
            self.dense = Dense(7*7*32, activation='relu')
            self.reshape = Reshape((7, 7, 32))
            self.conv2dT1 = Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation='relu')
            self.conv2dT2 = Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same',  activation='relu')
            self.out = Conv2DTranspose(filters=1, kernel_size=3, padding='same', name='output')
        else:
            digit_len = self.outp_shape[0]*self.outp_shape[1]*self.outp_shape[2]
            self.dense1 = Dense(512, activation='relu')
            self.dense2 = Dense(digit_len)
            self.out = Reshape(self.outp_shape)
        
        if info==True:
            self.print_info()

    def __call__(self, z):
        if self.net=='conv':
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
    ''' Reparametrization Trick
    The the speed of lerning is enormously increased by a reparametrization, that is done here.

    Inputs
    inputs:     tf.tensors of for the mean and the log variance. Used to define a normal distribution
                where a new z is sampled from.
    '''
    def __call__(self, inputs):
        mean, log_var = inputs
        eps = tf.random.normal(shape=tf.shape(mean))
        return mean + tf.multiply(tf.exp(0.5 * log_var), eps)




#######################################################################################
#    PLOTTER                                                                          #
#######################################################################################
class Plotter:
    def __init__(self, model):
        self.model = model

    def plot_losses(self, figsize=(4,3)):
        '''Plot losses
        Plots the recorded losses and metrics.

        Inputs
        figsize:    size of plot
        '''
        fig, axs = plt.subplots(1, 2, figsize=(8, 4), sharey=False)

        fig.suptitle('Encoder: {} -- Decoder: {}'.format(self.model.encoder.net, self.model.decoder.net))
        fig.tight_layout(pad=3.0)

        axs[0].set_xlabel('epochs')
        axs[1].set_xlabel('epochs')

        axs[0].set_ylabel('NELBO')
        axs[1].set_ylabel('Accuracy')

        train_loss = np.array(self.model.history)[:, 0]
        test_loss = np.array(self.model.history)[:, 2]
        train_acc = np.array(self.model.history)[:, 1]
        test_acc = np.array(self.model.history)[:, 3]

        loss_max_x = max(np.max(train_loss), np.max(test_loss))+5
        loss_min_x = min(np.min(train_loss), np.min(test_loss))-5
        acc_max_x = max(np.max(train_acc), np.max(test_acc))+1
        acc_min_x = min(np.min(train_acc), np.min(test_acc))-1


        axs[0].set_ylim(loss_min_x, loss_max_x)
        axs[1].set_ylim(acc_min_x, acc_max_x)

        axs[0].plot(train_loss, label='train loss')
        axs[0].plot(test_loss, label='test loss')
        axs[1].plot(train_acc, label='train accuracy')
        axs[1].plot(test_acc, label='test accuracy')

        axs[0].legend()
        axs[1].legend()

        plt.show()


    def latent_space_grid(self, n_samples=20, img_shape=(28,28,1), figsize=(15,15)):
        ''' Draw samples for a gridlike distribution of z in the latent variable space

        Input
        n_samples:  number of samples per row in the plot
        digit_size: width/height of a single image
        figsize:    size of plot
        '''
        norm = tfp.distributions.Normal(0, 1)
        grid_x = norm.quantile(np.linspace(0.05, 0.95, n_samples))
        grid_y = norm.quantile(np.linspace(0.05, 0.95, n_samples))

        plot_height = n_samples*img_shape[0]
        plot_width = n_samples*img_shape[1]
        image = np.zeros((plot_height, plot_width))

        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                z = np.array([[xi, yi]])
                x_sample = self.model.decode(z, logits=False)[0]
                digit = tf.reshape(x_sample, (img_shape[0], img_shape[1]))
                image[i * img_shape[0]: (i + 1) * img_shape[0], j * img_shape[1]: (j + 1) * img_shape[1]] = digit.numpy()


        plt.figure(figsize=figsize)
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        plt.show()

    def generate_plot_write(self, epoch, test_sample, figsize=(6,6),save=False):
        ''' Generate samples from a fixed test_sample.

        Input
        model:        VAE, trained for some epochs
        epoch:        int, epoch used in the filename for the exported png file
        test_sample:  batch of images as starting point for the new samples
        '''  
        sample, _, _ = self.model(test_sample, logits=False)

        plt.figure(figsize=figsize)
        for i in range(sample.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(sample[i, :, :, 0], cmap='gray')
            plt.axis('off')

        if save==True:
            filename = 'img_epoch{:04d}.png'.format(epoch)
            plt.savefig(filename)
            print()
            print('Image saved: ' + filename)
        plt.show()

    def plot_embedding(self, x, y, figsize=(10,10)):
        latent_dim = self.model.latent_dim
        net = self.model.encoder.net

        if latent_dim != 2:
            from sklearn.decomposition import PCA
            print("latent dimension = " + str(latent_dim) + ": Apply PCA!")
            z, _ = self.model.encoder(x)
            pca = PCA(n_components=2)
            z = pca.fit_transform(z)
        else:
            z, _ = self.model.encoder(x)
        
        plt.figure(figsize=figsize)
        plt.scatter(z[:, 0], z[:, 1], c=y,
            alpha=.4, s=3**2, cmap='tab20')
        plt.colorbar()
        plt.title(net + '-Encoder')
        plt.xlabel('z1')
        plt.ylabel('z2')
        plt.show()

#######################################################################################
#    VARIATIONAL AUTOENCODER                                                          #
#######################################################################################
class VAE(Model):
    ''' Variational Autoencoder
    '''
    def __init__(self, encoder, decoder, loss_fn, conditional=False, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        
        if conditional:
            assert self.decoder.net == 'mlp' and self.decoder.net == 'mlp'
            self.inp_shape = encoder.inp_shape
            self.latent_dim = encoder.latent_dim
            self.outp_shape = decoder.outp_shape
        else:
            assert encoder.inp_shape == decoder.outp_shape, print('Encoder input shape and decoder output shape need to be equal!')
            assert encoder.latent_dim == decoder.latent_dim, print('Encoder and decoder latent space dimension need to be equal!')
            self.inp_shape = encoder.inp_shape
            self.latent_dim = encoder.latent_dim

        self.loss_fn = loss_fn
        self.optimizer = tf.keras.optimizers.Adam()

        self.print_info()

        self.init_metrics()

        # Initialize Output
        self.plotter = Plotter(self)
        self.train_summary_writer = None
        self.test_summary_writer = None

    def __call__(self, x, logits=True):
        z, mean, log_var = self.encode(x)
        x_out = self.decode(z, logits=logits)
        return x_out, mean, log_var

    def print_info(self):
        print()
        print('++++ VARIATIONAL AUTOENCODER ++++')
        self.encoder.print_info()
        self.decoder.print_info()

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
        self.metrics_['train_loss'] = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        self.metrics_['train_accuracy'] = tf.keras.metrics.CategoricalAccuracy('train_accuracy', dtype=tf.float32)
        self.metrics_['test_loss'] = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
        self.metrics_['test_accuracy'] = tf.keras.metrics.CategoricalAccuracy('test_accuracy', dtype=tf.float32)
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
        self.metrics_['train_accuracy'].update_state(x, x_out)

    def fit(self, train_ds, test_ds, epochs=20, lr=1e-4, save=True):
        print()
        print("Set Adam Optimizer")
        self.optimizer = tf.keras.optimizers.Adam(lr)

        num_samples = 16
        for x_test_batch in test_ds.take(1):
            test_sample = x_test_batch[0:num_samples, :, :, :]

        print('Start Training...')
        for epoch in range(1, epochs+1):
            start_time = time.time()
            for train_x in train_ds:
                self.train_step(train_x, self.optimizer)      
            end_time = time.time()
            if self.train_summary_writer!=None:
                with self.train_summary_writer.as_default():
                    tf.summary.scalar('loss', self.metrics_['train_loss'].result(), step=epoch)
                    tf.summary.scalar('accuracy', self.metrics_['train_accuracy'].result(), step=epoch)
            for test_x in test_ds:
                x_out, mean, log_var = self.__call__(test_x, logits=True)
                loss = self.loss_fn(test_x, x_out, mean, log_var)
                self.metrics_['test_loss'](loss)
                self.metrics_['test_accuracy'].update_state(test_x, x_out)
            if self.test_summary_writer!=None:
                with self.test_summary_writer.as_default():
                    tf.summary.scalar('loss', self.metrics_['test_loss'].result(), step=epoch)
                    tf.summary.scalar('accuracy', self.metrics_['test_accuracy'].result(), step=epoch)

            display.clear_output(wait=False)
            template = 'Progress:\nEpoch {}/{}, Loss: {:.2f}, Accuracy: {:.2f}, Test Loss: {:.2f}, Test Accuracy: {:.2f}, Time: {:.2f} s'
            print(template.format(epoch,
                        epochs,
                        self.metrics_['train_loss'].result(), 
                        self.metrics_['train_accuracy'].result()*100,
                        self.metrics_['test_loss'].result(), 
                        self.metrics_['test_accuracy'].result()*100,
                        end_time - start_time))
            self.history.append([self.metrics_['train_loss'].result(), self.metrics_['train_accuracy'].result()*100, self.metrics_['test_loss'].result(), self.metrics_['test_accuracy'].result()*100])

            self.plotter.plot_losses()
            self.plotter.generate_plot_write(epoch, test_sample, save=save)
            
            self.reset_metrics()