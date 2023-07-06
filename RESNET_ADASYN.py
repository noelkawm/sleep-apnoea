from __future__ import division
import sys, getopt
import pandas as pd
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
#from scipy.signal import resample
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import ADASYN
import sklearn
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from pydrive2.auth import GoogleAuth
from oauth2client.service_account import ServiceAccountCredentials
from google.oauth2.credentials import Credentials
from pydrive2.drive import GoogleDrive
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.oauth2 import service_account
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.http import MediaFileUpload
import io
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import time
import argparse
import logging
import socket
from sklearn.model_selection import train_test_split

# For plotting
import matplotlib.pyplot as plt
import plotly
import plotly.offline as py
import plotly.tools as tls
#plotly.offline.init_notebook_mode()
# For building and training the neural network
from keras.models import Model
from keras.layers import (Input, Conv1D, MaxPooling1D, Dropout,
                          BatchNormalization, Dense, Activation, Add, Flatten)
from keras.utils.vis_utils import model_to_dot
from keras.optimizers import Adam
from keras.callbacks import (ModelCheckpoint, TensorBoard, ReduceLROnPlateau)
# IPython features
from IPython.display import SVG


timeout_in_sec = 60*60*10 # 10h timeout limit
socket.setdefaulttimeout(timeout_in_sec)

def ResNet1D(input_dims, residual_blocks_dims, nclasses,
             dropout_rate=0.8, kernel_size=16, kernel_initializer='he_normal'):
    """Create unidimensiona residual neural network model.
    
    Parameters
    ----------
    input_dims : tuple
        Input dimensions. Tuple containing dimensions for the neural network 
        input tensor. Should be like: ``(nsamples, nfilters)``.
    residual_blocks_dims : list of tuples
        Dimensions of residual blocks.  The i-th tuple should contain the dimensions
        of the output (i-1)-th residual block and the input to the i-th residual
        block. Each tuple shoud be like: ``(nsamples, nfilters)``.
    nclasses : int
        Number of classes in the output layer.
    dropout_rate: float
        Rate of dropout layers. Between 0 and 1. By default 0.8.
    kernel_size : int
        Kernel length used in convolutional layers. By default 16, 
        same as in (Rajpurkar et al., 2017) .
    kernel_initializer : str
        Method used for initializing the kernel By default 'he_normal'
        same as in (Rajpurkar et al., 2017) .
        
    Returns
    -------
    model: keras.models.Model
        Keras model describing the neural network archtecture
        
    Notes
    -----
    When specifing ``residual_blocks_dims`` the number of samples
    should always decrease and the ratio between two consecutive 
    ``nsamples`` should be an integer number. 
    Besides that it is usual in residual networks that the number 
    of filters should always increase.
    
    """
    # TODO: Try to include l2 regularization in the parameters from all layers.
    # This may help reducing variance error of our model.

    # Residual Block
    def residual_block(X, nsamples_in, nsamples_out,
                       nfilters_in, nfilters_out, first_block):
        # Skip Connection
        # Deal with downsampling
        downsample = nsamples_in // nsamples_out
        if downsample > 1:
            Y = MaxPooling1D(downsample, strides=downsample, padding='same')(X)
        elif downsample == 1:
            Y = X
        else:
            raise ValueError("Number of samples should always decrease.")
        # Deal with nfiters dimension increase
        if nfilters_in != nfilters_out:
            # This is one of the two alternatives presented in ResNet paper
            # Other option is to just fill the matrix with zeros.
            # TODO: try the alternative and see which one works better (low priority).
            Y = Conv1D(nfilters_out, 1, padding='same', 
                       use_bias=False, kernel_initializer=kernel_initializer)(Y)

        # End of 2nd layer from last residual block
        if not first_block:
            X = BatchNormalization()(X)
            X = Activation('relu')(X)
            X = Dropout(dropout_rate)(X)

        # 1st layer
        X = Conv1D(nfilters_out, kernel_size, padding='same', 
                   use_bias=False, kernel_initializer=kernel_initializer)(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)
        X = Dropout(dropout_rate)(X)

        # 2nd layer
        X = Conv1D(nfilters_out, kernel_size, strides=downsample, padding='same',
                   use_bias=False, kernel_initializer=kernel_initializer)(X)
        X = Add()([X, Y]) # Skip connection and main connection
        return X

    # Define input representing ecg_signals
    ecg_signals = Input(shape=input_dims, dtype=np.float32, name='ecg_signals')
    X = ecg_signals

    # First layer
    downsample = input_dims[0] // residual_blocks_dims[0][0]
    X = Conv1D(residual_blocks_dims[0][1], kernel_size, strides=downsample, 
               padding='same', kernel_initializer=kernel_initializer, 
               use_bias=False)(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    # Residual blocks
    first_block = True
    nresidual_blocks = len(residual_blocks_dims) - 1
    for i in range(nresidual_blocks):
        X = residual_block(X, residual_blocks_dims[i][0], residual_blocks_dims[i+1][0], 
                           residual_blocks_dims[i][1], residual_blocks_dims[i+1][1], first_block)
        first_block = False        
    # End of 2nd layer from last residual block
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Dropout(dropout_rate)(X)

    # Last layer
    X = Flatten()(X)
    diagnosis = Dense(nclasses, kernel_initializer=kernel_initializer, 
                      activation='softmax', name='diagnosis')(X)

    return Model(ecg_signals, diagnosis)


def driveUpload(service, filepath):
    body = {'name': filepath, 'mimeType': 'application/octet-stream', 'parents': ['1qLxroprD2jwb1ag6vUwYsdCM2imWH3Cx']}
    media = MediaFileUpload(filepath, mimetype = 'text/html')
    fiahl = service.files().create(body=body, media_body=media).execute()
    logging.info("Created file '%s' id '%s'." % (fiahl.get('name'), fiahl.get('id')))

def createModel(epochs, batch_size, noRows):

    logging.basicConfig(filename="logRESNET_ADASYN",
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)
    
    logging.info("Running RESNET_ADASYN.py")
    
    scope = ['https://www.googleapis.com/auth/drive']
    service_account_json_key = 'my_key.json'
    credentials = service_account.Credentials.from_service_account_file(
                                  filename=service_account_json_key, 
                                  scopes=scope)
    service = build('drive', 'v3', credentials=credentials)
    logging.info('Connected to Google Drive')
    
    logger = logging.getLogger('RESNET_ADASYN')
       
    # Define dimensions
    nclasses = 3
    input_dims = [2560, 1]  # Input dimensions
    residual_blocks_dims = [[2560, 64],   # Output of conv layer         / Input of 1st residual blk
                            [2560, 64],   # Output of 1st residual blk / Input of 2st residual blk
                            [2048, 64],   # Output of 2st residual blk / Input of 3st residual blk
                            [2048, 64],   # Output of 3rd residual blk / Input of 4st residual blk
                            [1024, 128],  # Output of 4th residual blk / Input of 5st residual blk
                            [1024, 128],  # Output of 5th residual blk / Input of 6st residual blk
                            [512,  128],  # Output of 6th residual blk / Input of 7st residual blk
                            [512,  128],  # Output of 7th residual blk / Input of 8st residual blk
                            [256,  192],  # Output of 8th residual blk / Input of 9st residual blk
                            [256,  192],  # Output of 9th residual blk / Input of 10st residual blk
                            [128,  192],  # Output of 10th residual blk / Input of 11st residual blk
                            [128,  192],  # Output of 11th residual blk / Input of 12st residual blk
                            [64,   256],  # Output of 12th residual blk / Input of 13st residual blk
                            [64,   256],  # Output of 13th residual blk / Input of 14st residual blk
                            [32,   256],  # Output of 14th residual blk / Input of 15st residual blk
                            [32,   256],  # Output of 15th residual blk / Input of 16st residual blk
                            [16,   256]]  # Output of 16th residual blk / Input of dense layer
    
    # Get model
    model = ResNet1D(input_dims, residual_blocks_dims, nclasses)
    
    # Present model summary
    model.summary()
    
    
    # Optimization parameters
    lr = 0.001  # Learning rate
    
    # Adam optimizer
    opt = Adam(lr, beta_1=0.9, beta_2=0.999)
    
    # Compile model
    model.compile(loss='categorical_crossentropy', 
                  optimizer=opt, metrics=['accuracy'])
    
    callbacks = []
    
    # Reduce learning rate on platteu. That is, we
    # reduce the learning rate when the validation
    # loss stops improving. This strategy is addopted
    # in (Rajpurkar et al., 2017).
    callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                       patience=10, min_lr=lr/10000))
    
    # Create file for tensorboard visualization.
    # Run tensor board from command line to
    # visualize dynamic graphs of your training 
    # and test metrics. Run:
    # $ tensorboard --logdir=/full_path_to_your_logs
    callbacks.append(TensorBoard(log_dir='./logs',  batch_size=batch_size,
                                 write_graph=False))
    
    # Save the model from time to time.
    # To load the model just run:
    # model = load_model('./backup_model.hdf5')
    callbacks.append(ModelCheckpoint(filepath= 'model_RESNET_ADASYN-' + str(epochs) + "-" + str(batch_size) + '.h5',
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True))
    
    df = pd.read_csv("combined_data.csv", index_col=0,nrows=noRows)
    logger.info('Loaded dataset')
    
    y = df['Type']
    X = df.loc[:, df.columns != "Type"]
    
    su = ADASYN(random_state=42)
    X_su, y_su = su.fit_resample(X, y)
    
    X_su = X_su.to_numpy()
    y_su = y_su.to_numpy()
    
    y_train_new = np.zeros((y_su.shape[0],3))
    for i in range(y_su.shape[0]):
        y_train_new[i][y_su[i]] = 1
    
    X_su = X_su.reshape(X_su.shape[0],X_su.shape[1],1)
    
    x_train, x_test, y_train, y_test = train_test_split(X_su,y_train_new, test_size=0.20, shuffle= True)
    
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test),epochs=epochs,batch_size = batch_size, callbacks = callbacks)
    logger.info('Model is trained')
    
    filepath = "modelRESNET_ADASYN-" + str(epochs) + "-" + str(batch_size) + "-val_accuracy-" + str(max(history.history['val_accuracy'])) + ".png"
    
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(filepath)
    driveUpload(service, filepath)
    
    filepath = "modelRESNET_ADASYN-" + str(epochs) + "-" + str(batch_size) + "-val_loss-" + str(max(history.history['val_loss'])) + ".png"
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(filepath)
    driveUpload(service, filepath)
    
    def inttostring(x):
        if(x < 10):
            return "0" + str(x)
        else:
            return str(x)
        
        
    def floattostring(x):
        s = str(x)
        if(len(s) > 6):
            s = s[:6]
        elif len(s) < 6:
            while(len(s) < 6):
                s += "0"
        return(s)
    
    filepath = 'model_RESNET_ADASYN-' + str(epochs) + "-" + str(batch_size) + '.h5'
    model.load_weights(filepath)
    driveUpload(service, filepath)
    
    y_pred = model.predict(X_su)
    y_pred_new = np.zeros(y_pred.shape[0])
    for i in range(y_pred.shape[0]):
        y_pred_new[i] = np.argmax(y_pred[i])
    
    
    filepath = "modelRESNET_ADASYN_confusion_matrix-" + str(epochs) + "-" + str(batch_size) + "-val_accuracy-" + str(max(history.history['val_accuracy'])) + ".png"
    
    matrix = confusion_matrix(y_su, y_pred_new)
    
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(x=j, y=i,s=matrix[i, j], va='center', ha='center', size='xx-large')
     
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.savefig(filepath)
    driveUpload(service, filepath)

def main(argv):
    # Default parameters
    epochs = 2
    batch_size = 64
    noRows = 200
    
    try:
       opts, args = getopt.getopt(argv,"e:, b:, n:",["ifile="])
    except getopt.GetoptError:
       print('RESNET_ADASYN.py -e <no epochs (default 3)> -b <batch size (default 64) > -n <number of rows to load from combined_data.csv> (default 500)')
       sys.exit(2)
    for opt, arg in opts:
       if opt == '-h':
          print('RESNET_ADASYN.py -e <no epochs (default 3)> -b <batch size (default 64) > -n <number of rows to load from combined_data.csv> (default 500, None for all rows)')
          sys.exit()
       elif opt in ("-e"):
          epochs = int(arg)       
       elif opt in ("-b"):
          batch_size = int(arg)       
       elif opt in ("-n"):
           if arg == 'None':
               noRows = None
           else:
               noRows = int(arg)                 
    createModel(epochs, batch_size, noRows)

if __name__ == '__main__':
    main(sys.argv[1:])
