from googleapiclient.discovery import build
from google.oauth2 import service_account
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
import os

creds = service_account.Credentials.from_service_account_file(
    'datasciencestuff-504b93ed360f.json',
    scopes=['https://www.googleapis.com/auth/drive']
)

drive_service = build('drive', 'v3', credentials=creds)

def download_file(file_id, file_path):
    request = drive_service.files().get_media(fileId=file_id)
    with open(file_path, 'wb') as f:
        downloader = MediaIoBaseDownload(f, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            print(f"Download {int(status.progress() * 100)}%")

def process_folder(folder_id, local_path):
    results = drive_service.files().list(
        pageSize=100,
        fields="nextPageToken, files(id, name, mimeType)",
        q=f"'{folder_id}' in parents and trashed=false"
    ).execute()
    
    items = results.get('files', [])
    if not items:
        print(f'No files found in this folder.')
        return
        
    for item in items:
        print(f"Found: {item['name']}")
        if item['mimeType'] == 'application/vnd.google-apps.folder':
            # Create local subfolder
            new_folder_path = os.path.join(local_path, item['name'])
            os.makedirs(new_folder_path, exist_ok=True)
            # Recursively process subfolder
            process_folder(item['id'], new_folder_path)
        else:
            # Download file
            file_path = os.path.join(local_path, item['name'])
            print(f"Downloading: {item['name']}")
            try:
                download_file(item['id'], file_path)
            except Exception as e:
                print(f"Error downloading {item['name']}: {str(e)}")

def upload_file(file_path, folder_id):
    file_metadata = {
        'name': os.path.basename(file_path),
        'parents': [folder_id]
    }
    media = MediaFileUpload(file_path, resumable=True)
    file = drive_service.files().create(
        body=file_metadata,
        media_body=media,
        fields='id'
    ).execute()
    print(f"File {file_path} uploaded with ID: {file.get('id')}")

# Create local directory if it doesn't exist
local_folder = 'datasd download process
folder_id = '1K5jG2Q8MOkppU2DXJFM6FzHR5IvQVxXP'
#process_folder(folder_id, local_folder)

# Upload the specific file to the corresponding subfolder on Google Drive
file_to_upload = '/home/user/Desktop/RP/datasets/ourData/Data/Dataset/Annotations/landmarks_normalized_vladimir.json'
subfolder_id = '1K5jG2Q8MOkppU2DXJFM6FzHR5IvQVxXP'  # Replace with the actual subfolder ID
upload_file(file_to_upload, subfolder_id)
