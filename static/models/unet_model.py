import numpy as np
import os
# import skimage.io as io
# import skimage.transform as trans
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as keras


def unet(pretrained_weights = None,input_size = (None,None,1), net_scale = 1):
    net_scale = 2^net_scale
    inputs = Input(input_size)
    conv1 = Conv2D(2*net_scale, 3, activation = 'relu', padding = 'same')(inputs)
    conv1 = Conv2D(2*net_scale, 3, activation = 'relu', padding = 'same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(4*net_scale, 3, activation = 'relu', padding = 'same')(pool1)
    conv2 = Conv2D(4*net_scale, 3, activation = 'relu', padding = 'same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(8*net_scale, 3, activation = 'relu', padding = 'same')(pool2)
    conv3 = Conv2D(8*net_scale, 3, activation = 'relu', padding = 'same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(16*net_scale, 3, activation = 'relu', padding = 'same')(pool3)
    conv4 = Conv2D(16*net_scale, 3, activation = 'relu', padding = 'same')(conv4)
    drop4 = Dropout(0.7)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(32*net_scale, 3, activation = 'relu', padding = 'same')(pool4)
    conv5 = Conv2D(32*net_scale, 3, activation = 'relu', padding = 'same')(conv5)
    drop5 = Dropout(0.7)(conv5)

    up6 = Conv2D(16*net_scale, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6],axis=3)
    conv6 = Conv2D(16*net_scale, 3, activation = 'relu', padding = 'same')(merge6)
    conv6 = Conv2D(16*net_scale, 3, activation = 'relu', padding = 'same')(conv6)

    up7 = Conv2D(8*net_scale, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7],axis=3)
    conv7 = Conv2D(8*net_scale, 3, activation = 'relu', padding = 'same')(merge7)
    conv7 = Conv2D(8*net_scale, 3, activation = 'relu', padding = 'same')(conv7)

    up8 = Conv2D(4*net_scale, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8],axis=3)
    conv8 = Conv2D(4*net_scale, 3, activation = 'relu', padding = 'same')(merge8)
    conv8 = Conv2D(4*net_scale, 3, activation = 'relu', padding = 'same')(conv8)

    up9 = Conv2D(2*net_scale, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9],axis=3)
    conv9 = Conv2D(2*net_scale, 3, activation = 'relu', padding = 'same')(merge9)
    conv9 = Conv2D(2*net_scale, 3, activation = 'relu', padding = 'same')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    # model = Model(input = inputs, output = conv10)
    model = Model(inputs, conv10)

    #sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=2, nesterov=True)

    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    #model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model
