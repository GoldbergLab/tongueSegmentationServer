from tensorflow.keras.backend import clear_session
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, Callback as keras_callback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from static.models.unet_model import unet
from scipy.io import loadmat
import json
import random
import logging

# This script uses a set of training data assembled into a .mat file consisting of a stack of images
#    and a corresponding set of binary masks that label the pixels in the image stacks into two classes.
#    The script then initializes a randomized U-NET (using the topology defined in the file model.py).
#    It then initiates training using a given batch size and # of epochs, saving the best net at each
#    step to the given .hdf5 file path.
#
# Written by Teja Bollu, documented and modified by Brian Kardon

def createDataAugmentationParameters(rotation_range=None, width_shift_range=0.1,
    height_shift_range=0.3, zoom_range=0.4, horizontal_flip=True,
    vertical_flip=True):
    # Create dictionary of data augmentation parameter
    return {
        "rotation_range":rotation_range,
        "width_shift_range":width_shift_range,
        "height_shift_range":height_shift_range,
        "zoom_range":zoom_range,
        "horizontal_flip":horizontal_flip,
        "vertical_flip":vertical_flip
    }

class PrintLogger:
    def __init__(self):
        pass
    def log(lvl, msg):
        print(msg)

def trainNetwork(trained_network_path, training_data_path, start_network_path=None,
    augment=True, batch_size=10, epochs=512, image_field_name='imageStack',
    mask_field_name='maskStack', data_augmentation_parameters={},
    epoch_progress_callback=None, logger=None):
    # Actually train the network, saving the best network to a file after each epoch.
    # augment = boolean flag indicating whether to randomly augment training data
    # batch_size = Size of training batches (size of batches that dataset is divided into for each epoch):
    # epochs = Number of training epochs (training runs through whole dataset):
    # training_data_path = .mat file containing the training data (image data and corresponding manually created mask target output):
    # image_field_name = Field within .mat file that contains the relevant images:
    # mask_field_name = Field within .mat file that contains the relevant masks:
    # trained_network_path = File path to save trained network to:
    # data_augmentation_parameters = to use for data augmentation:
    # epoch_progress_callback = a function to call at the end of each epoch,
    #   which takes a progress argument which will be a dictionary of progress
    #   indicators

    if logger is None:
        logger = PrintLogger()

    # Reset whatever buffers or saved state exists...not sure exactly what that consists of.
    # This may not actually work? Word is you have to restart whole jupyter server to get this to work.
    clear_session()

    # Convert inter-epoch progress callback to a tf.keras.Callback object
    epoch_progress_callback = TrainingProgressCallback(epoch_progress_callback)

    # Load training data
    print('Loading images and masks...')
    data = loadmat(training_data_path)
    img = data[image_field_name]
    mask = data[mask_field_name]

    # Process image and mask data into the proper format
    img_shape = img.shape;
    num_samples = img_shape[0]
    img_size_x = img_shape[1]
    img_size_y = img_shape[2]
    img = img.reshape(num_samples, img_size_x, img_size_y, 1)
    mask = mask.reshape(num_samples, img_size_x, img_size_y, 1)

    print("...image and mask data loaded.")
    print("Image stack dimensions:", img.shape)
    print(" Mask stack dimensions:", mask.shape)
    print('start path:', start_network_path)
    print('train path:', trained_network_path)

    if augment:
        imgGen = ImageDataGenerator(**data_augmentation_parameters)
        maskGen = ImageDataGenerator(**data_augmentation_parameters)

    if start_network_path is None:
        # Randomize new network structure using architecture in model.py file
        lickbot_net = unet(net_scale = 1)
    else:
        # Load previously trained network from a file
        lickbot_net = load_model(start_network_path)

    # Instruct training algorithm to save best network to disk whenever an improved network is found.
    model_checkpoint = ModelCheckpoint(str(trained_network_path), monitor='loss', verbose=1, save_best_only=True)
    callback_list = [model_checkpoint] #, TestCallback()]
    # if epoch_progress_callback is not None:
    #     callback_list.append(epoch_progress_callback)

    if augment:
        print("Using automatically augmented training data.")
        # Train network using augmented dataset
        seed = random.randint(0, 1000000000)
        imgIterator = imgGen.flow(img, seed=seed, shuffle=False, batch_size=batch_size)
        maskIterator = maskGen.flow(mask, seed=seed, shuffle=False, batch_size=batch_size)

        steps_per_epoch = int(num_samples / batch_size)
        lickbot_net.fit(
            ((imgBatch, maskBatch) for imgBatch, maskBatch in zip(imgIterator, maskIterator)),
            steps_per_epoch=steps_per_epoch, # # of batches of generated data per epoch
            epochs=epochs,
            verbose=1,
            callbacks=callback_list
        )
    else:
        lickbot_net.fit(
            img,
            mask,
            epochs=epochs,
            verbose=1,
            callbacks=callback_list
        )

class TestCallback(keras_callback):
    def on_epoch_end(self, epoch, logs=None):
        print('Epoch done!!!!')
        print(epoch)
        print(logs)

class TrainingProgressCallback(keras_callback):
    def __init__(self, progressFunction):
        super(TrainingProgressCallback, self).__init__()
        self.logs = []
        self.progressFunction = progressFunction

    def on_epoch_end(self, epoch, logs=None):
        # self.logs.append(logs)
        # keys = list(logs.keys())
        self.progressFunction(epoch)
        # print("End epoch {} of training; got log keys: {}".format(epoch, keys))

def validateNetwork(trained_network_path, img=None, imgIterator=None, maskIterator=None):
    # Load trained network
    lickbot_net = load_model(trained_network_path)

    if augment:
        img_validate = imgIterator.next()
        mask_validate = maskIterator.next()
    else:
        print('Using original dataset for visualization')
        img_validate = img
        mask_validate = mask

    mask_pred = lickbot_net.predict(img_validate)
    mask_pred.shape

    # %matplotlib inline
    from matplotlib import pyplot as plt
    from matplotlib import gridspec

    numValidation = img_validate.shape[0]

    img_shape = img.shape;
    num_samples = img_shape[0]
    img_size_x = img_shape[1]
    img_size_y = img_shape[2]

    img_disp =                       img_validate.reshape(numValidation,img_size_x,img_size_y)
    mask_disp =                     mask_validate.reshape(numValidation,img_size_x,img_size_y)
    mask_pred = lickbot_net.predict(img_validate).reshape(numValidation,img_size_x,img_size_y)

    scaleFactor = 3

    plt.figure(figsize=(scaleFactor*3,scaleFactor*numValidation))
    plt.subplots_adjust(wspace=0, hspace=0)
    gs = gridspec.GridSpec(nrows=numValidation, ncols=3, width_ratios=[1, 1, 1],
             wspace=0.0, hspace=0.0, bottom=0, top=1, left=0, right=1)
    for k in range(numValidation):
        plt.subplot(gs[k, 0])
        plt.imshow(mask_disp[k])
        plt.subplot(gs[k, 1])
        plt.imshow(mask_pred[k])
        plt.subplot(gs[k, 2])
        plt.imshow(img_disp[k])
