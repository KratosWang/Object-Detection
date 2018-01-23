# 1
# 加载keras模块
from __future__ import print_function
import numpy as np
np.random.seed(1337)

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation
from keras.optimizers import SGD,Adam,RMSprop
from keras.utils import np_utils
from keras.objectives import categorical_crossentropy

# 加载Keras后端和tensorflow
import tensorflow as tf
from keras import backend as K

# 2
# 变量初始化
batch_size = 128
nb_classes = 10
nb_epoch = 20

# 3
# 准备数据
(X_train, y_train),(X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000,784)
X_test = X_test.reshape(10000, 784)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /=255
X_test /=255

print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# 转换类符号
''' y_train
    array([5, 0, 4, ..., 5, 6, 8], dtype=uint8)
    Y_train
    array([[ 0.,  0.,  0., ...,  0.,  0.,  0.],
           [ 1.,  0.,  0., ...,  0.,  0.,  0.],
           [ 0.,  0.,  0., ...,  0.,  0.,  0.],
          ..., 
           [ 0.,  0.,  0., ...,  0.,  0.,  0.],
           [ 0.,  0.,  0., ...,  0.,  0.,  0.],
           [ 0.,  0.,  0., ...,  0.,  1.,  0.]])
'''
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# 4
# 建立模型 使用Sequential
model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax'))

# 打印模型
model.summary()

# 训练与评估
model.compile(loss='categorical_crossentropy',
              # 优化器
              optimizer=RMSprop(),
              # 指标
              metrics=['accuracy']
              )

# 迭代训练
history = model.fit(X_train, Y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(X_test, Y_test))

# 模型评估
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score', score[0])
print('Test accuracy', score[1])