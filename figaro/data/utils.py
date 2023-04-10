import os
import hashlib
import tarfile
import tqdm
import urllib.request
import shutil

CACHE_DIR = os.getenv("CACHE_DIR", "data_cache")


def download_extract_tar(url, cache_dir=CACHE_DIR):
    """Written using ChatGPT"""

    # Create the cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)

    # Normalize the URL
    url_key = urllib.parse.urlparse(url)._replace(scheme="", query="", fragment="").geturl().lstrip("//")

    # Compute the hash of the URL
    hash_value = hashlib.sha256(url_key.encode()).hexdigest()

    # Use the hash as the cache key
    cached_file = os.path.join(cache_dir, hash_value)

    # Check if the file already exists in the cache directory
    if os.path.exists(cached_file):
        return cached_file

    # Extract the filename from the URL
    filename = os.path.basename(url)

    # Download the file
    with urllib.request.urlopen(url) as response:
        chunk_size = 1024 * 1024
        total = int(response.info().get("Content-Length")) // chunk_size
        with tqdm.tqdm(total=total, desc="Downloading", unit="MB") as pbar:
            with open(os.path.join(cache_dir, filename), "wb") as out_file:
                while True:
                    data = response.read(chunk_size)
                    if not data:
                        break
                    out_file.write(data)
                    pbar.update(1)

    # Extract the contents of the file
    with tarfile.open(os.path.join(cache_dir, filename), "r:gz") as tar, \
         tqdm.tqdm(total=len(tar.getmembers()), desc="Extracting") as pbar:
        # Keep track of extracted directories
        extracted_dirs = set()
        for member in tar:
            # Skip top-level directories
            if member.isdir() and member.name.count("/") == 1:
                continue

            # Extract the member
            tar.extract(member, path=cache_dir)
            pbar.update(1)

            # Keep track of extracted directories
            if member.isdir():
                extracted_dirs.add(member.name.split("/")[0])

    # Create a new folder to contain the extracted directories
    extracted_dir = os.path.join(cache_dir, hash_value)

    # Move extracted directories into the new folder
    if len(extracted_dirs) > 1:
        os.makedirs(extracted_dir, exist_ok=True)
        for extracted_dir_name in extracted_dirs:
            extracted_dir_path = os.path.join(cache_dir, extracted_dir_name)
            if os.path.exists(extracted_dir_path):
                shutil.move(extracted_dir_path, extracted_dir)
    else:
        extracted_dir_path = os.path.join(cache_dir, max(extracted_dirs))
        shutil.move(extracted_dir_path, extracted_dir)

    # Delete the compressed file
    os.remove(os.path.join(cache_dir, filename))

    # Return the path to the extracted folder
    return extracted_dir
