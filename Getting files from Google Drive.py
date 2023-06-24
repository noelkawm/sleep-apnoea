#pip install -U threadpoolctl

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

logging.basicConfig(filename="logGFFGD",
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

logging.info("Running Getting files from Google Drive.py")

logger = logging.getLogger('GFFGD')


scope = ['https://www.googleapis.com/auth/drive']
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
        logger.error(F'An error occurred: {error}')

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


df.to_csv("combined_data.csv")
logger.info('Saved dataset as combined_data.csv')

