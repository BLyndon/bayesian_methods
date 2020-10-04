import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

#######################################################################################
#    PLOTTER                                                                          #
#######################################################################################


class Plotter:
    '''Visualize results from a trained vae.'''

    def plot_losses(self, model, figsize=(12, 6)):
        '''Plots the recorded losses and metrics.

        Inputs
        model (VAE): trained model with recorded history
        figsize (tuple): size of plot
        '''
        fig, axs = plt.subplots(1, 2, figsize=figsize, sharey=False)

        fig.suptitle(
            'Encoder: {} -- Decoder: {}'.format(model.encoder.net, model.decoder.net))
        fig.tight_layout(pad=3.0)

        axs[0].set_xlabel('Epochs')
        axs[1].set_xlabel('Epochs')

        axs[0].set_ylabel('NELBO')
        axs[1].set_ylabel('Accuracy')

        train_loss = np.array(model.history)[:, 0]
        test_loss = np.array(model.history)[:, 2]
        train_acc = np.array(model.history)[:, 1]
        test_acc = np.array(model.history)[:, 3]

        loss_max_x = max(np.max(train_loss), np.max(test_loss))+5
        loss_min_x = min(np.min(train_loss), np.min(test_loss))-5
        acc_max_x = max(np.max(train_acc), np.max(test_acc))+1
        acc_min_x = min(np.min(train_acc), np.min(test_acc))-1

        axs[0].set_ylim(loss_min_x, loss_max_x)
        axs[1].set_ylim(acc_min_x, acc_max_x)

        axs[0].plot(train_loss, label='train data')
        axs[0].plot(test_loss, label='test data')
        axs[1].plot(train_acc, label='train data')
        axs[1].plot(test_acc, label='test data')

        axs[0].legend()
        axs[1].legend()

        plt.show()

    def plot_processing(self, model, x_train, x_test, figsize=(10, 5)):
        fig = plt.figure(figsize=figsize)
        for fid_idx, (x_data, title) in enumerate(
                zip([x_train, x_test], ['Train', 'Validation'])):
            n = 10
            digit_height = x_test.shape[1]
            digit_width = x_test.shape[2]

            figure = np.zeros((digit_height * 2, digit_width * n))
            decoded, _, _ = model(x_data[:n, :], logits=False)
            decoded = decoded.numpy()
            for i in range(n):
                figure[:digit_height, i * digit_width: (
                    i + 1) * digit_width] = x_data[i, :].reshape(digit_height, digit_width)
                figure[digit_height:, i * digit_width: (
                    i + 1) * digit_width] = decoded[i, :].reshape(digit_height, digit_width)
            ax = fig.add_subplot(2, 1, fid_idx + 1)
            ax.imshow(figure, cmap='gray')
            ax.set_title(title)
            ax.axis('off')
        plt.show()

    def plot_processing_cond(self, model, x_train, x_test, y_train, y_test, figsize=(10, 5)):
        fig = plt.figure(figsize=figsize)
        for fid_idx, (x_data, y_data, title) in enumerate(
                zip([x_train, x_test], [y_train, y_test], ['Train', 'Validation'])):
            n = 10
            digit_height = x_test.shape[1]
            digit_width = x_test.shape[2]

            figure = np.zeros((digit_height * 2, digit_width * n))
            decoded, _, _ = model(x_data[:n, :], y_data[:n, :], logits=False)
            decoded = decoded.numpy()
            for i in range(n):
                figure[:digit_height, i * digit_width: (
                    i + 1) * digit_width] = x_data[i, :].reshape(digit_height, digit_width)
                figure[digit_height:, i * digit_width: (
                    i + 1) * digit_width] = decoded[i, :].reshape(digit_height, digit_width)
            ax = fig.add_subplot(2, 1, fid_idx + 1)
            ax.imshow(figure, cmap='gray')
            ax.set_title(title)
            ax.axis('off')
        plt.show()

    def latent_space_grid(self, model, n_samples=20, img_shape=(28, 28, 1), figsize=(15, 15)):
        ''' Draw samples from different latent space points. The points are set by a grid given by different quantiles.

        Input
        n_samples (int):  number of samples per row in the plot
        image_shape (tuple): fixes width & height of a single image
        figsize (tuple): size of plot
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
                x_sample = model.decode(z, logits=False)[0]
                digit = tf.reshape(x_sample, (img_shape[0], img_shape[1]))
                image[i * img_shape[0]: (i + 1) * img_shape[0], j *
                      img_shape[1]: (j + 1) * img_shape[1]] = digit.numpy()

        plt.figure(figsize=figsize)
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        plt.show()

    def generate_plot_write(self, model, epoch, x_sample, y_sample=None, figsize=(6, 6), save=False):
        ''' Generate samples from a give test_sample.

        Input
        model (VAE): VAE, trained for some epochs
        epoch (int): epoch used in the filename of the exported png file
        test_sample (tf.data):  batch of images passed through the vae to generate new samples
        '''
        if y_sample == None:
            sample, _, _ = model(x_sample, logits=False)
        else:
            sample, _, _ = model(x_sample, y_sample, logits=False)

        plt.figure(figsize=figsize)
        for i in range(sample.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(sample[i, :, :, 0], cmap='gray')
            plt.axis('off')

        if save == True:
            filename = 'img_epoch{:04d}.png'.format(epoch)
            plt.savefig(filename)
            print()
            print('Image saved: ' + filename)
        plt.show()

    def plot_embedding(self, model, x, y, figsize=(10, 10)):
        ''' Plot embedding of datapoints in the latent space.

        Input
        model (VAE): VAE, trained for some epochs
        epoch (int): epoch used in the filename of the exported png file
        test_sample (tf.data):  batch of images passed through the vae to generate new samples
        '''
        latent_dim = model.latent_dim
        net = model.encoder.net

        if latent_dim != 2:
            from sklearn.decomposition import PCA
            print("latent dimension = " + str(latent_dim) + ": Apply PCA!")
            z, _ = model.encoder(x)
            pca = PCA(n_components=2)
            z = pca.fit_transform(z)
        else:
            z, _ = model.encoder(x)

        y_categorical = tf.argmax(y, axis=1)

        plt.figure(figsize=figsize)
        plt.scatter(z[:, 0], z[:, 1], c=y_categorical,
                    alpha=.4, s=3**2, cmap='tab20')
        plt.colorbar()
        plt.title(net + '-Encoder')
        plt.xlabel('z1')
        plt.ylabel('z2')
        plt.show()
