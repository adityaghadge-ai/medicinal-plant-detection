# download_test_images.py
import os
import requests

# Folder for test images
os.makedirs("test_images", exist_ok=True)

# Sample images from neem leaf dataset (Mendeley Data - direct file links)
image_urls = [
    "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/sjtxmcv5d4-1.zip"
]

print("⬇️ Downloading Neem dataset ZIP file...")

for url in image_urls:
    filename = os.path.join("test_images", url.split("/")[-1])
    response = requests.get(url, stream=True)
    with open(filename, "wb") as f:
        f.write(response.content)
    print(f"✅ Saved {filename}")

print("Now unzip the file inside test_images/ and pick some images for prediction.")
