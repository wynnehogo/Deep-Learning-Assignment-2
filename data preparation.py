import random
import os
import shutil
from shutil import copy2
import glob
import json
from sklearn.model_selection import train_test_split

## 1. CLASSIFICATION DATASET
src = r"C:/Users/ACER/Downloads/archive (5)/imagenette2-160/train" # change it with your folder source

# Destination folders to save the train/val split
dst_train = r"data/imagenette_160/train"
dst_val = r"data/imagenette_160/val"

# Get only directories inside src (the classes)
classes = [cls for cls in os.listdir(src) if os.path.isdir(os.path.join(src, cls))]

print(f"Classes found: {classes}")

for cls in classes:
    cls_path = os.path.join(src, cls)
    imgs = [img for img in os.listdir(cls_path) if os.path.isfile(os.path.join(cls_path, img))]

    if len(imgs) < 30:
        raise ValueError(f"Class '{cls}' has less than 30 images ({len(imgs)})!")

    # Randomly sample 30 images
    sampled_imgs = random.sample(imgs, 30)

    # Create destination directories for each class
    os.makedirs(os.path.join(dst_train, cls), exist_ok=True)
    os.makedirs(os.path.join(dst_val, cls), exist_ok=True)

    # Copy 24 images to train folder
    for img in sampled_imgs[:24]:
        copy2(os.path.join(cls_path, img), os.path.join(dst_train, cls))

    # Copy remaining 6 images to val folder
    for img in sampled_imgs[24:]:
        copy2(os.path.join(cls_path, img), os.path.join(dst_val, cls))

print("Sampling and splitting complete.")


## 2. SEGMENTATION DATASET
mask_dir = r"C:\Users\ACER\Downloads\archive (4)\VOC2012_train_val\VOC2012_train_val\SegmentationClass" # change it to your mask path
image_dir = r"C:\Users\ACER\Downloads\archive (4)\VOC2012_train_val\VOC2012_train_val\JPEGImages" # change it to your image path

# Step 1: Find all .png masks and check their corresponding .jpg 
mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
print(f"Found {len(mask_paths)} total masks.")

# Filter only those with a matching .jpg file
valid_pairs = []
for mask_path in mask_paths:
    mask_filename = os.path.basename(mask_path)
    image_filename = mask_filename.replace(".png", ".jpg")
    image_path = os.path.join(image_dir, image_filename)
    if os.path.exists(image_path):
        valid_pairs.append((image_path, mask_path))
    else:
        print(f"Missing image for mask: {mask_filename}")

print(f"Valid image-mask pairs: {len(valid_pairs)}")

# Step 2: Split into train (240) and val (60)
train_pairs, temp_val = train_test_split(valid_pairs, train_size=240, random_state=42)
val_pairs = temp_val[:60]

# Step 3: Define function to clear + copy 
def copy_pairs(pairs, split_name):
    img_dest = f"data/mini_voc_seg/{split_name}/images"
    mask_dest = f"data/mini_voc_seg/{split_name}/masks"

    # Clean old content
    if os.path.exists(img_dest):
        shutil.rmtree(img_dest)
    if os.path.exists(mask_dest):
        shutil.rmtree(mask_dest)

    os.makedirs(img_dest, exist_ok=True)
    os.makedirs(mask_dest, exist_ok=True)

    # Copy new files
    for img_path, mask_path in pairs:
        img_filename = os.path.basename(img_path)
        mask_filename = os.path.basename(mask_path)
        shutil.copy(img_path, os.path.join(img_dest, img_filename))
        shutil.copy(mask_path, os.path.join(mask_dest, mask_filename))

    print(f"Copied {len(pairs)} {split_name} images and masks.")

# Step 4: Copy to output folders
copy_pairs(train_pairs, "train")
copy_pairs(val_pairs, "val")

print("✅ Segmentation dataset split and copied successfully!")
print("📂 Saved to:", os.path.abspath("data/mini_voc_seg"))


## 3. DETECTION DATASET
json_path = r"D:\C\wyns\0. MILS\hw2 .py\instances_minitrain2017.json" # change it with your .json file path
image_src_dir = r"C:\Users\ACER\Downloads\archive\images" # change it with your.jpg file path

# Target save folders
train_dir = "data/mini_coco_det/train"
val_dir = "data/mini_coco_det/val"
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

print(f"Loading JSON from: {json_path}")
with open(json_path, "r") as f:
    data = json.load(f)

images = data["images"]
annotations = data["annotations"]
categories = data["categories"]

# Step 1: split off val with 60 images
train_val_imgs, val_imgs = train_test_split(images, test_size=60, random_state=42)

# Step 2: from the train_val_imgs, select exactly 240 for training
train_imgs, _ = train_test_split(train_val_imgs, train_size=240, random_state=42)

# Step 3: Filter annotations by image_id
def filter_anns(imgs, anns):
    img_ids = {img["id"] for img in imgs}
    return [ann for ann in anns if ann["image_id"] in img_ids]

train_data = {
    "images": train_imgs,
    "annotations": filter_anns(train_imgs, annotations),
    "categories": categories
}

val_data = {
    "images": val_imgs,
    "annotations": filter_anns(val_imgs, annotations),
    "categories": categories
}

# Step 4: Save new JSON files 
with open(os.path.join(train_dir, "instances_train.json"), "w") as f:
    json.dump(train_data, f)

with open(os.path.join(val_dir, "instances_val.json"), "w") as f:
    json.dump(val_data, f)

# Step 5: Copy corresponding images 
def copy_images(img_list, dest_dir):
    count = 0
    for img in img_list:
        fname = img["file_name"]
        src = os.path.join(image_src_dir, fname)
        dst = os.path.join(dest_dir, fname)
        if os.path.exists(src):
            shutil.copy(src, dst)
            count += 1
        else:
            print(f"⚠️ Missing image: {src}")
    print(f"✅ Copied {count} images to {dest_dir}")

copy_images(train_imgs, train_dir)
copy_images(val_imgs, val_dir)

print("\n✅ COCO dataset split completed.")
print("📁 Train folder:", os.path.abspath(train_dir))
print("📁 Val folder:  ", os.path.abspath(val_dir))
