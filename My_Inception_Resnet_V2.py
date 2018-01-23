# 1-加载keras模块
from __future__ import print_function
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Input,Dense,Dropout,Activation,Flatten,Lambda,AveragePooling2D
from keras.layers import Concatenate
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
import tensorflow as tf
import six
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.layers.merge import add
import numpy as np
import time

# 2-变量初始化
batch_size = 32
nb_class= 10
nb_epoch = 20
data_augmentation = True

img_rows, img_cols = 128, 128

img_channels = 3

# 3-准备数据
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# X_train = tf.image.resize_images(X_train,(128,128))
# X_test = tf.image.resize_images(X_test,(128,128))


X_train_128 = np.ndarray(shape=(50000,128,128,3))
X_test_128 = np.ndarray(shape=(10000,128,128,3))
k_train = X_train.shape
k_test = X_test.shape


# TODO time
print(time.asctime(time.localtime(time.time())))


for i in range(k_train[0]):

    for j in range(k_train[3]):
        a = np.hstack((X_train[i,:,:,j],np.zeros((32,96))))
        b = np.vstack((a,np.zeros((96,128))))
        X_train_128[i,:,:,j] = b
print('X_train shape', X_train.shape)
print(X_train.shape[0],'train samples')
# TODO time
print(time.asctime(time.localtime(time.time())))


for i in range(k_test[0]):

    for j in range(k_test[3]):
        a = np.hstack((X_test[i,:,:,j],np.zeros((32,96))))
        b = np.vstack((a,np.zeros((96,128))))
        X_test_128[i,:,:,j] = b

print('X_test shape', X_test.shape)
print(X_test.shape[0], 'test samples')
# TODO time
print(time.asctime(time.localtime(time.time())))



Y_train = np_utils.to_categorical(y_train,nb_class)
Y_test = np_utils.to_categorical(y_test,nb_class)

# 4-建立模型
# TODO axis=-1
LLambda = Lambda(lambda x: K.concatenate([x[0],x[1]],axis=-1))

def _handle_dim_ordering():
    global ROW_AXIS
    global COL_AXIS
    global CHANNEL_AXIS

    if K.image_dim_ordering() == 'tf':
        ROW_AXIS = 1
        COL_AXIS = 2
        CHANNEL_AXIS = 3

    else:
        CHANNEL_AXIS = 1
        ROW_AXIS = 2
        COL_AXIS = 3

#   TODO 不太明白的def
def _get_block(identifier):
    if isinstance(identifier, six.string_types):
        res = globals().get(identifier)
        if not res:
            raise  ValueError('Invaild {}'.format(identifier))
        return res
    return identifier

def _bn_relu(input):
    norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
    return Activation("relu")(norm)


def _conv_bn_relu(**conv_params):
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides",1,1)
    kernel_initializer = conv_params.setdefault("kernel_initializer","he_normal")
    padding = conv_params.setdefault("padding","same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer",l2(1.e-4))

    def f(input):
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides,padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(input)
        return _bn_relu(conv)

    return f



def conv_my(**conv_params):
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params["strides"]
    padding = conv_params.setdefault("padding", "same")
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(input)
        return _bn_relu(conv)

    return f



def _residual_block(block_fuction, filters, repetition, is_first_layer=False):
    def f(input):
        init_strides = (1,1)
        input = block_fuction(filters=filters, init_strides=init_strides)(input)
        return input
    return f

def _shortcut(input, residual):
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[ROW_AXIS])/residual_shape[ROW_AXIS])
    stride_height = int(round(input_shape[COL_AXIS]/residual_shape[COL_AXIS]))
    equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

    shortcut = input

    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],
                          kernel_size=(1,1),
                          strides=(1,1),
                          padding='valid',
                          kernel_initializer='he_normal',
                          kernel_regularizer=l2(0.0001))(input)
    return add([shortcut,residual])


def build(input_shape, num_output):
    _handle_dim_ordering()
    if len(input_shape) !=3:
        raise Exception("Input shape should be a tuple(nb_channels, nb_rows, nb_clos)")

    if K.image_dim_ordering() == 'tf':
        input_shape = (input_shape[1], input_shape[2], input_shape[0])

    #block_fn = _get_block(block_fn)

    # TODO : Stem
    input = Input(shape=input_shape)

    conv1 = conv_my(filters=32, kernel_size=(3,3), strides=(2,2), padding='same')(input)
    conv2 = conv_my(filters=32, kernel_size=(3,3), strides=(1,1), padding='same')(conv1)
    conv3 = conv_my(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(conv2)

    pool4_1 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(conv3)
    conv4_2 =conv_my(filters=96, kernel_size=(3,3), strides=(2,2), padding='same')(conv3)

    f_concat_1 = LLambda([pool4_1,conv4_2])

    conv_left_1 = conv_my(filters=64, kernel_size=(1,1), strides=(1,1), padding='same')(f_concat_1)
    conv_left_2 = conv_my(filters=96, kernel_size=(3,3), strides=(1,1), padding='same')(conv_left_1)

    conv_right_1 = conv_my(filters=64, kernel_size=(1,1), strides=(1,1), padding='same')(f_concat_1)
    conv_right_2 = conv_my(filters=64, kernel_size=(7,1), strides=(1,1), padding='same')(conv_right_1)
    conv_right_3 = conv_my(filters=64, kernel_size=(1,7), strides=(1,1), padding='same')(conv_right_2)
    conv_right_4 = conv_my(filters=96, kernel_size=(3,3), strides=(1,1), padding='same')(conv_right_3)

    f_concat_2 = LLambda([conv_left_2,conv_right_4])

    conv_last_left = conv_my(filters=192, kernel_size=(3,3), strides=(2,2), padding='same')(f_concat_2)
    pool_last_right = MaxPooling2D(pool_size=(3,3), strides=(2,2),padding='same')(f_concat_2)

    stem_out = LLambda([conv_last_left,pool_last_right])

    # TODO Output size = 16 x 16 x 384

    # TODO : Residual block

    relu_top = _bn_relu(stem_out)

    patch_1 = conv_my(filters=32, kernel_size=(1,1), strides=(1,1), padding='same')(stem_out)

    patch_2_1 = conv_my(filters=32, kernel_size=(1,1), strides=(1,1), padding='same')(stem_out)
    patch_2_2 = conv_my(filters=32, kernel_size=(3,3), strides=(1,1), padding='same')(patch_2_1)

    patch_3_1 = conv_my(filters=32, kernel_size=(1,1), strides=(1,1), padding='same')(stem_out)
    patch_3_2 = conv_my(filters=48, kernel_size=(3,3), strides=(1,1), padding='same')(patch_3_1)
    patch_3_3 = conv_my(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(patch_3_2)

    patch_x = LLambda([patch_1,patch_2_2])
    patch = LLambda([patch_x,patch_3_3])

    patch_conv = conv_my(filters=384, kernel_size=(1,1), strides=(1,1), padding='same')(patch)

    res_out = _shortcut(relu_top,patch_conv)

    # TODO : 圆圈加号后

    block = Activation(activation='relu')(res_out)

    block_shape = K.int_shape(block)
    pool2 = AveragePooling2D(pool_size=(block_shape[ROW_AXIS],block_shape[COL_AXIS]),
                             strides=(1,1))(block)
    flatten1 = Flatten()(pool2)
    dense = Dense(units=num_output, kernel_initializer='he_normal',
                  activation='softmax')(flatten1)
    model = Model(inputs=input, outputs=dense)

    return model







# 打印模型
#model.summary()
model = build((img_channels,img_rows,img_cols),nb_class)
# 5-训练与评估
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

# 数据压缩为0~1之间
X_train_128 = X_train_128.astype('float32')
X_test_128 = X_test_128.astype('float32')
X_train_128 /= 255
X_test_128 /= 255

# 数据增强
if not data_augmentation:
    print('Not using data augmentation')
    model.fit(X_train_128, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_data=(X_test_128, Y_test),
              shuffle=True)

else:
    print('Using data augmentation')

    # 这将做预处理和实时数据增加
    datagen = ImageDataGenerator(
        featurewise_center=False, # 在数据集上将输入平均值设置为0
        samplewise_center=False ,# 将每个样本均值设置为0
        featurewise_std_normalization=False , #将输入除以数据集的std
        samplewise_std_normalization=False, #将每个输入除以其std
        zca_whitening=False, #应用ZCA白化
        rotation_range=0, #在一个范围下随机旋转图像（degrees， 0 to 180）
        width_shift_range=0.1, #水平随机移位图像（总宽度的分数）
        height_shift_range=0.2, # 随机的垂直移位图像（总高度的分数）
        horizontal_flip=True, #随机翻转图像
        vertical_flip=False #随机翻转图像
    )

    # 计算特征方向归一化所需要的参数
    # （std, mean, and_principal components if ZCA whitening is applied

    datagen.fit(X_train_128)

    #fit the model on the batchs generated by datagen.flow()
    model.fit_generator(datagen.flow(X_train_128,Y_train,
                        batch_size=batch_size),
                        samples_per_epoch=X_train_128.shape[0],
                        nb_epoch=nb_epoch,
                        validation_data=(X_test_128,Y_test))

score = model.evaluate(X_test_128, Y_test, verbose=0)
print('Test score : ', score[0])
print('Test accuracy : ', score[1])