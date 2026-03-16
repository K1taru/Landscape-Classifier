import kagglehub

# Download Intel Image Classification dataset
# Dataset: https://www.kaggle.com/datasets/puneet6060/intel-image-classification
# Classes: buildings, forest, glacier, mountain, sea, street
path = kagglehub.dataset_download("puneet6060/intel-image-classification")

print("Dataset downloaded to:", path)
print("Move the 6 class folders into: src/data/raw/Datasets/")
