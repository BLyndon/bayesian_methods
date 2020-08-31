import tensorflow as tf
import numpy as np

def get_datasets(img_shape=(28, 28, 1), batch_size=32, asarray=False, dataset='mnist'):
    ''' Choose between 'mnist' and 'fashion_mnist' 
    By default the MNIST dataset wil be rescaled and batched and transformed to tf.data.Dataset.
    In addition it can be returned as arrays to extract the images and labels.

    Input
    int:    Splitting and batch size of the dataset.
    bool:   Switch between both types of return
    string: Choose dataset

    Return
    rf.data.Dataset:  train_ds, test_ds   batched and rescaled Dataset
    arrays:           (x_train, y_train), (x_test, y_test) rescaled. Data points and labels are easily accesible in this form
    '''
    def preprocess(images, size, img_shape):
        images = images.reshape(size, img_shape[0], img_shape[1], img_shape[2]) / 255.
        return np.where(images > .5, 1.0, 0.0).astype('float32')
    def to_dataset(x, y, size, batch_size):
        return (tf.data.Dataset.from_tensor_slices((x, y)).shuffle(size).batch(batch_size))

    print()

    if dataset=='mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        print('Load MNIST')
    elif dataset=='fashion_mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        print('Load Fashion MNIST')

    train_size = x_train.shape[0]
    test_size = x_test.shape[0]
    print("Train Size: {}".format(train_size))
    print("Test Size: {}".format(test_size))
    
    if asarray:
        return (x_train, y_train), (x_test, y_test)
    else:
        (x_train, x_test) = (preprocess(x_train, train_size, img_shape), preprocess(x_test, test_size, img_shape))
        (train_ds, test_ds) = (to_dataset(x_train, y_train, train_size, batch_size), to_dataset(x_test, y_test, test_size, batch_size))
        return train_ds, test_ds

if __name__ == "__main__":
    pass