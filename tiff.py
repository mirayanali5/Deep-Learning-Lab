import pickle, os
from PIL import Image
import numpy as np
from tqdm import tqdm

# Path to your CIFAR batch file
batch_path = r"C:\Users\miray\OneDrive\Desktop\DL\Deep-Learning-Lab\data_batch_1"

# CIFAR-10 label names
labels = ['airplane','automobile','bird','cat','deer',
          'dog','frog','horse','ship','truck']

# 1️⃣ Load CIFAR binary batch
with open(batch_path, 'rb') as f:
    batch = pickle.load(f, encoding='bytes')

data = batch[b'data']
labels_list = batch[b'labels']

print(f"Loaded {len(data)} images from {batch_path}")

# 2️⃣ Create output directories
base_dir = os.path.join(os.path.dirname(batch_path), "CIFAR10_TIFF")
os.makedirs(base_dir, exist_ok=True)
for label in labels:
    os.makedirs(os.path.join(base_dir, label), exist_ok=True)

# 3️⃣ Convert images to TIFF
print("Converting and saving as .tiff files...")
for i in tqdm(range(len(data))):
    img_array = data[i].reshape(3, 32, 32).transpose(1, 2, 0)
    cls = labels[labels_list[i]]
    img = Image.fromarray(img_array)
    img.save(os.path.join(base_dir, cls, f"img_{i}.tiff"))

print("\n✅ Done! TIFF images saved to:", base_dir)

# 4️⃣ Optional: verify
for cls in labels:
    count = len(os.listdir(os.path.join(base_dir, cls)))
    print(f"{cls:12s}: {count} images")
