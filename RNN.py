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


logging.basicConfig(filename="logRNN",
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)

logging.info("Running RNN.py")

logger = logging.getLogger('RNN')

"""scope = ['https://www.googleapis.com/auth/drive']
service_account_json_key = 'my_key.json'
credentials = service_account.Credentials.from_service_account_file(
                              filename=service_account_json_key, 
                              scopes=scope)
service = build('drive', 'v3', credentials=credentials)

logger.info('Connected to Google Drive')

df = pd.DataFrame()

page_token = None
items=[]

while True:
    results = service.files().list(pageSize=1000, fields="nextPageToken, files(id, name)", q='name contains "csv"', pageToken=page_token).execute()
    # get the results
    items.extend(results.get('files', []))
    page_token = results.get("nextPageToken", None)
    if page_token is None:
        break
        
#print(len(items))
#print(items)
        
i = 0
while(i < len(items)):
    try: 
        request_file = service.files().get_media(fileId=items[i]['id'])
        file = io.BytesIO()
        downloader = MediaIoBaseDownload(file, request_file)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            #print(F'Download {int(status.progress() * 100)}.')
    except HttpError as error:
        print(F'An error occurred: {error}')

    file_retrieved: str = file.getvalue()
    with open(f"{items[i]['name']}", 'wb') as f:
        f.write(file_retrieved)
    df_temp = pd.read_csv(items[i]['name'], header=None)

    name_type = 0 if items[i]['name'].find("CNT") != -1 else 1 if items[i]['name'].find("oh") != -1 else 2
    df_temp.insert(0, "Type", [name_type for _ in range(df_temp.shape[0])])

    df = pd.concat([df, df_temp])

    df_temp = pd.DataFrame(None)

    os.remove(items[i]['name'])
    
    #print(i)

    i += 1

logger.info('Loaded dataset from Google Drive')
    
"""

df = pd.read_csv("combined_data.csv", index_col=0)
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

#model.summary()


model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath= 'modelRNN.{epoch:02d}-{accuracy:.4f}-{val_accuracy:.4f}.h5',
    save_weights_only=True,
    monitor='accuracy',
    mode='max',
    save_best_only=True)


model.compile(optimizer =tf.keras.optimizers.Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics="accuracy")
logger.info('Model is compiled')


y_train_new = np.zeros((y_su.shape[0],3))
for i in range(y_su.shape[0]):
    y_train_new[i][y_su[i]] = 1


X_su = X_su.reshape(X_su.shape[0],X_su.shape[1],1)


history = model.fit(X_su,y_train_new,epochs=5,validation_split=0.05,verbose=0, callbacks = [model_checkpoint_callback])
logger.info('Model is trained')


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


y_pred = model.predict(X_su)
y_pred_new = np.zeros(y_pred.shape[0])
for i in range(y_pred.shape[0]):
    y_pred_new[i] = np.argmax(y_pred[i])


matrix = confusion_matrix(y_su, y_pred_new)

fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(matrix, cmap=plt.cm.Blues, alpha=0.3)
for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        ax.text(x=j, y=i,s=matrix[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()
