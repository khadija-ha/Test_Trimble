import os
import shutil
import random

# Define paths to the source dataset folder and destination folders
dataset_folder = '/media/khadija/data_ssd1/trimble/dataset/' 
train_folder = '/media/khadija/data_ssd1/trimble/dataset/train/'
val_folder = '/media/khadija/data_ssd1/trimble/dataset/val/'

# Create destination folders if they don't already exist
os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)

# Define the split ratio between the training and validation sets (80% - 20%)
train_split_ratio = 0.8

# Iterate through each class (fields and roads)
for class_name in ['fields', 'roads']:
    class_folder = os.path.join(dataset_folder, class_name)
    image_files = os.listdir(class_folder)
    random.shuffle(image_files)

    num_images = len(image_files)
    num_train_images = int(num_images * train_split_ratio)
    
    # Split images into training and validation sets    
    train_images = image_files[:num_train_images]
    val_images = image_files[num_train_images:]

    # Copy images to the training and validation folders
    for image_file in train_images:
        src_path = os.path.join(class_folder, image_file)
        dest_path = os.path.join(train_folder, class_name, image_file)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy(src_path, dest_path)

    for image_file in val_images:
        src_path = os.path.join(class_folder, image_file)
        dest_path = os.path.join(val_folder, class_name, image_file)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy(src_path, dest_path)

print("Data division into training and validation sets is complete.")
