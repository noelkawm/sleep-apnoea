import sys, getopt
import pandas as pd
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from scipy.signal import resample
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
timeout_in_sec = 60*60*10 # 10h timeout limit
socket.setdefaulttimeout(timeout_in_sec)

def driveUpload(service, filepath):
    body = {'name': filepath, 'mimeType': 'application/octet-stream', 'parents': ['1qLxroprD2jwb1ag6vUwYsdCM2imWH3Cx']}
    media = MediaFileUpload(filepath, mimetype = 'text/html')
    fiahl = service.files().create(body=body, media_body=media).execute()
    logging.info("Created file '%s' id '%s'." % (fiahl.get('name'), fiahl.get('id')))

def createModel(epochs, batch_size, noRows):

    logging.basicConfig(filename="logRNN_SMOTE",
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)
    
    logging.info("Running RNN_SMOTE.py")
    
    scope = ['https://www.googleapis.com/auth/drive']
    service_account_json_key = 'my_key.json'
    credentials = service_account.Credentials.from_service_account_file(
                                  filename=service_account_json_key, 
                                  scopes=scope)
    service = build('drive', 'v3', credentials=credentials)
    logging.info('Connected to Google Drive')
    
    logger = logging.getLogger('RNN_SMOTE')
    
    df = pd.read_csv("combined_data.csv", index_col=0,nrows=noRows)
    logger.info('Loaded dataset')
    
    y = df['Type']
    X = df.loc[:, df.columns != "Type"]
    
    su = BorderlineSMOTE(random_state=42)
    X_su, y_su = su.fit_resample(X, y)
    logger.info('Dataset is balanced')
    
    X_su = X_su.to_numpy()
    y_su = y_su.to_numpy()
    
    #X_train, X_test, y_train, y_test = train_test_split(X_su, y_su, test_size = 0.25, random_state=0)
    
    model = keras.Sequential()
    
    model.add(layers.SimpleRNN(512,input_shape=(2560,1)))
    
    model.add(layers.Dense(4096, activation='sigmoid'))
    model.add(layers.Dense(2048, activation='sigmoid'))
    model.add(layers.Dense(1024, activation='sigmoid'))
    model.add(layers.Dense(512, activation='sigmoid'))
    model.add(layers.Dense(256, activation='sigmoid'))
    model.add(layers.Dense(128, activation='sigmoid'))
    model.add(layers.Dense(32, activation='sigmoid'))
    model.add(layers.Dense(16, activation='sigmoid'))
    model.add(layers.Dense(8, activation='sigmoid'))
    model.add(layers.Dense(3, activation='softmax'))
    
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath= 'model_RNN_SMOTE-' + str(epochs) + "-" + str(batch_size) + '.h5',
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)
      
    model.compile(optimizer =tf.keras.optimizers.Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics="accuracy")
    logger.info('Model is compiled')
    
    y_train_new = np.zeros((y_su.shape[0],3))
    for i in range(y_su.shape[0]):
        y_train_new[i][y_su[i]] = 1
    
    X_su = X_su.reshape(X_su.shape[0],X_su.shape[1],1)
    
    
    history = model.fit(X_su,y_train_new,epochs=epochs,batch_size = batch_size, validation_split=0.05, callbacks = [model_checkpoint_callback])
    logger.info('Model is trained')
    
    filepath = "modelRNN_SMOTE-" + str(epochs) + "-" + str(batch_size) + "-val_accuracy-" + str(max(history.history['val_accuracy'])) + ".png"
    
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(filepath)
    driveUpload(service, filepath)
    
    filepath = "modelRNN_SMOTE-" + str(epochs) + "-" + str(batch_size) + "-val_loss-" + str(max(history.history['val_loss'])) + ".png"
    
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
    
    filepath = "modelRNN_SMOTE." + inttostring(history.history['val_accuracy'].index(max(history.history['val_accuracy'])) + 1) + "-"
    filepath = 'model_RNN_SMOTE-' + str(epochs) + "-" + str(batch_size) + '.h5'
    model.load_weights(filepath)
    driveUpload(service, filepath)
    
    y_pred = model.predict(X_su)
    y_pred_new = np.zeros(y_pred.shape[0])
    for i in range(y_pred.shape[0]):
        y_pred_new[i] = np.argmax(y_pred[i])
    
    
    filepath = "modelRNN_SMOTE_confusion_matrix-" + str(epochs) + "-" + str(batch_size) + "-val_accuracy-" + str(max(history.history['val_accuracy'])) + ".png"
    
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
    epochs = 3
    batch_size = 64
    noRows = 500
    
    try:
       opts, args = getopt.getopt(argv,"e:, b:, n:",["ifile="])
    except getopt.GetoptError:
       print('RNN_SMOTE.py -e <no epochs (default 3)> -b <batch size (default 64) > -n <number of rows to load from combined_data.csv> (default 500)')
       sys.exit(2)
    for opt, arg in opts:
       if opt == '-h':
          print('RNN_SMOTE.py -e <no epochs (default 3)> -b <batch size (default 64) > -n <number of rows to load from combined_data.csv> (default 500, None for all rows)')
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
