import os
import io
import time
import random
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.errors import HttpError

# Path to your service account key file
SERVICE_ACCOUNT_FILE = '/root/hcmus/alpine-keep-445414-q6-f0413ad6fb6b.json'

# Define the required scope
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

# Folder ID from the provided Google Drive link
FOLDER_ID = '1_qyXfbCDT5i8ad0hv66BvOjt3M8dYtV7'

# Directory to save downloaded videos
OUTPUT_DIR = '/root/hcmus/EnTube/2021'

# Maximum number of retries
MAX_RETRIES = 5

# Initial delay for backoff in seconds
INITIAL_DELAY = 1

# Maximum delay for backoff in seconds
MAX_DELAY = 64

# Number of threads for concurrent downloads
MAX_WORKERS = 16

# Lock for synchronizing file operations
file_lock = Lock()

def create_service():
    """Create and return a new Google Drive service instance."""
    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    return build('drive', 'v3', credentials=credentials)

def list_mp4_files():
    """List all MP4 files in the specified Google Drive folder, handling pagination."""
    service = create_service()
    query = f"'{FOLDER_ID}' in parents and mimeType='video/mp4'"
    files = []
    page_token = None

    while True:
        response = service.files().list(
            q=query,
            spaces='drive',
            fields='nextPageToken, files(id, name)',
            pageToken=page_token
        ).execute()

        files.extend(response.get('files', []))
        page_token = response.get('nextPageToken', None)
        if page_token is None:
            break

    return files

def download_file(file_id, file_name):
    """Download a file from Google Drive with exponential backoff and retry."""
    service = create_service()
    file_path = os.path.join(OUTPUT_DIR, file_name)

    # Check if file already exists
    with file_lock:
        if os.path.exists(file_path):
            print(f"File {file_name} already exists. Skipping download.")
            return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    request = service.files().get_media(fileId=file_id)

    retry_count = 0
    delay = INITIAL_DELAY

    while retry_count < MAX_RETRIES:
        try:
            with io.FileIO(file_path, 'wb') as fh:
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while not done:
                    status, done = downloader.next_chunk()
                    if status:
                        print(f"Downloading {file_name}: {int(status.progress() * 100)}% complete.")
            print(f"Successfully downloaded: {file_name}")
            return
        except HttpError as error:
            if error.resp.status in [500, 502, 503, 504]:
                print(f"Server error {error.resp.status} encountered. Retrying in {delay} seconds.")
                time.sleep(delay + random.uniform(0, 1))
                delay = min(delay * 2, MAX_DELAY)
                retry_count += 1
            else:
                print(f"Failed to download {file_name}: {error}")
                return
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return

    print(f"Exceeded maximum retries for {file_name}. Moving to next file.")

def main():
    """Main function to download all MP4 files from the specified Google Drive folder using multithreading."""
    mp4_files = list_mp4_files()
    if not mp4_files:
        print("No MP4 files found in the folder.")
        return

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for file in mp4_files:
            executor.submit(download_file, file['id'], file['name'])

if __name__ == '__main__':
    main()
