import os

from torchvision.datasets import Food101

TARGET_DIR = "./data"
DATASET_DIR = os.path.join(TARGET_DIR, "food-101")

# if automatic download fails, download manually at https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
def download_and_extract_food101():
    """
    Downloads the Food101 dataset and returns the local path.
    """
    os.makedirs(TARGET_DIR, exist_ok=True)

    if os.path.isdir(DATASET_DIR):
        print("Dataset already exists !")
        print("Path to dataset files:", DATASET_DIR)
        return DATASET_DIR

    print("Downloading...")
    Food101(root=TARGET_DIR, split="train", download=True)
    print("Download complete!")
    print("Path to dataset files:", DATASET_DIR)
    return DATASET_DIR


if __name__ == "__main__":
    download_and_extract_food101()