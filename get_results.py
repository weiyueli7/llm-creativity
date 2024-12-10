"""
Utility script to download the results from the Google Cloud Storage bucket.

Usage:
    python get_results.py
"""
import os
from google.cloud import storage
from tqdm import tqdm

def download_folder_from_bucket(bucket_name, source_folder, destination_folder, credentials_file):

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_file
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    blobs = bucket.list_blobs(prefix=source_folder)

    for blob in tqdm(blobs):
        if blob.name.endswith("/"):
            continue

        # construct the local path
        relative_path = os.path.relpath(blob.name, source_folder)
        local_path = os.path.join(destination_folder, relative_path)

        local_dir = os.path.dirname(local_path)
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)

        print(f"Downloading {blob.name} to {local_path}...")
        blob.download_to_filename(local_path)

    print("Download completed.")

if __name__ == "__main__":
    BUCKET_NAME = "nlp_result"
    SOURCE_FOLDER = "Results"
    DESTINATION_FOLDER = "./Results"
    CREDENTIALS_FILE = "secrets.json"

    download_folder_from_bucket(BUCKET_NAME, SOURCE_FOLDER, DESTINATION_FOLDER, CREDENTIALS_FILE)
