import numpy as np
import tensorflow as tf
import h5py

#######################################################################################
#    DATASET LOADER                                                                   #
#######################################################################################
def get_vortex_data(path, asarray=False, batch_size=32):
    def scaleC(tmp='U'):
        nt,nx,nz=np.shape(tmp)
        tmp=np.reshape(tmp,(nt,nx*nz))
        tmp=scaler.fit_transform(tmp.T)
        tmp=tmp.T
        tmp=np.reshape(tmp,(nt,nx,nz,1))
        tmp=tmp[:,2:-2,2:,:]
        return tmp
    hf = h5py.File(path, 'r')
    U = hf.get('U')
    U = np.array(U)
    hf.close()
    print('Data Loaded')
    
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0.01, 0.99))

    U=scaleC(U)
    x_train, x_test = train_test_split(U, test_size=0.33, random_state=42)
    train_size = x_train.shape[0]
    test_size = x_test.shape[0]
    print('Database size: {}'.format(np.shape(U)))
    print('Train Set size: {}'.format(np.shape(x_train)))
    print('Test Set size: {}'.format(np.shape(x_test)))
    if asarray:
        return x_train, x_test, x_train.shape, x_test.shape
    else:
        train_ds = tf.data.Dataset.from_tensor_slices(x_train.astype('float32')).batch(batch_size)
        test_ds = tf.data.Dataset.from_tensor_slices(x_test.astype('float32')).batch(batch_size)
        return train_ds, test_ds, x_train.shape, x_test.shape

def get_mnist(img_shape=(28, 28, 1), batch_size=32, asarray=False, dataset='mnist'):
    ''' Choose between 'mnist' and 'fashion_mnist' 
    By default the MNIST dataset wil be rescaled and batched and transformed to tf.data.Dataset.
    In addition it can be returned as arrays to extract the images and labels.
    '''
    def preprocess(images, size):
        images = images.reshape(size, img_shape[0], img_shape[1], img_shape[2]) / 255.
        return np.where(images > .5, 1.0, 0.0).astype('float32')
    def to_dataset(x, size):
        return tf.data.Dataset.from_tensor_slices(x).shuffle(size).batch(batch_size)

    if dataset=='mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        print()
        print('Load MNIST')
    elif dataset=='fashion_mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        print()
        print('Load Fashion MNIST')
    
    x_train = preprocess(x_train, x_train.shape[0])
    x_test = preprocess(x_test, x_test.shape[0])

    if asarray:
        print('Return as Arrays')
        return (x_train, y_train), (x_test, y_test)
    else:
        train_ds = to_dataset(x_train, x_train.shape[0])
        test_ds = to_dataset(x_test, x_test.shape[0])
        print('Return as Datasets')
        return train_ds, test_ds, x_train.shape, x_test.shape

    if __name__ == "__main__":
        pass