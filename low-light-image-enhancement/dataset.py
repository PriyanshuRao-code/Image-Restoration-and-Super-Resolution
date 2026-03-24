import kagglehub

# Download latest version
path = kagglehub.dataset_download("soumikrakshit/lol-dataset")

print("Path to dataset files:", path)