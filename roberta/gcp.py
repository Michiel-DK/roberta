import os

from google.cloud import storage
from termcolor import colored
import joblib


def storage_upload(rm=False):
    client = storage.Client().bucket(BUCKET_NAME)

    local_model_name = 'model.joblib'
    storage_location = f"models/{MODEL_NAME}/{MODEL_VERSION}/{local_model_name}"
    blob = client.blob(storage_location)
    blob.upload_from_filename('model.joblib')
    print(colored(f"=> model.joblib uploaded to bucket {BUCKET_NAME} inside {storage_location}",
                  "green"))
    if rm:
        os.remove('model.joblib')

def download_model(model_directory="v2", bucket=BUCKET_NAME, rm=True):
    client = storage.Client().bucket(bucket)

    storage_location = 'models/{}/{}/{}'.format(
        MODEL_NAME,
        model_directory,
        'model.joblib')
    blob = client.blob(storage_location)
    blob.download_to_filename('model.joblib')
    print("=> pipeline downloaded from storage")
    model = joblib.load('model.joblib')
    if rm:
        os.remove('model.joblib')
    return model