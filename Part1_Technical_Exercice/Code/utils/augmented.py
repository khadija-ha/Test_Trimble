import cv2
import numpy as np
import os


# Path to the folder containing the original images
input_folder ='/media/khadija/data_ssd1/trimble/dataset/fields/'
# Path to the folder where augmented images will be saved
output_folder = '/media/khadija/data_ssd1/trimble/aug/field/'
os.makedirs(output_folder, exist_ok=True)
# List all the image files in the input folder
image_files = os.listdir(input_folder)
# Desired number of augmented images
target_count = 500

# Loop until the desired number of augmented images
while len(image_files) < target_count:
    # Choosing a random image from the input folder
    random_image = np.random.choice(image_files)    
    # Loading the image
    img = cv2.imread(os.path.join(input_folder, random_image))   
    # Apply random augmentation transformations
    # Example: rotate, flip, and change brightness
    angle = np.random.randint(-30, 30)
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    img = cv2.flip(img, np.random.choice([-1, 0, 1]))
    brightness = np.random.uniform(0.7, 1.3)
    img = cv2.convertScaleAbs(img, alpha=brightness, beta=0)   
    # Saving the augmented image to the output folder
    output_path = os.path.join(output_folder, f'augmented_{len(image_files)}.jpg')
    cv2.imwrite(output_path, img)    
    # Adding the augmented image to the list
    image_files.append(output_path)

print(f'{len(image_files)} augmented images created.')