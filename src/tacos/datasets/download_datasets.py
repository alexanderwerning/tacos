import os
import requests
from tqdm import tqdm
import subprocess

import aac_datasets
from aac_datasets.datasets.functional.clotho import download_clotho_datasets
from aac_datasets.datasets.functional.wavcaps import download_wavcaps_datasets

def download_clotho(data_path: str):

    download_clotho_datasets(
        subsets=["dev", "val", "eval"],
        root=data_path,
        clean_archives=False,
        verbose=5
    )


def download_zip_from_cloud(url: str, zip_file: str):

    if os.path.exists(zip_file):
        print(f"{zip_file} already exists. Skipping download. {url}")
        return

    response = requests.get(
        url,
        stream=True
    )

    if response.status_code == 200:
        total_size = int(response.headers.get("content-length", 0))  # Get file size in bytes
        progress_bar = tqdm(total=total_size, unit="B", unit_scale=True, desc=f"Downloading {url}")

        with open(zip_file, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    progress_bar.update(len(chunk))
    else:
        raise Exception(f"Failed to download {url}.")


def extract_zip(zip_file: str, extract_to_dir: str):
    subprocess.run(["7z", "x", zip_file, f"-o{extract_to_dir}"], check=True)
