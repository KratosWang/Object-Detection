import time
from keras.datasets import cifar10
import numpy as np
import tensorflow as tf

end1 = time.clock()
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

#X_train = tf.image.resize_images(X_train,(128,128))
#X_test = tf.image.resize_images(X_test,(128,128))


print('X_train shape', X_train.shape)
print('X_test shape', X_test.shape)
print(X_train.shape[0],'train samples')
print(X_test.shape[0], 'test samples')

X_train_128 = np.ndarray(shape=(50000,128,128,3))
k = X_train.shape

for i in range(k[0]):

    for j in range(k[3]):
        a = np.hstack((X_train[i,:,:,j],np.zeros((32,96))))
        b = np.vstack((a,np.zeros((96,128))))
        X_train_128[i,:,:,j] = b



end2 = time.clock()
print(end2-end1)