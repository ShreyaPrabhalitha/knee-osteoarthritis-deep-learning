import os
import cv2
import numpy as np
import pydicom
from tensorflow.keras.utils import to_categorical

def load_images_from_folder(folder_path):
    images = []
    labels = []  # Placeholder for labels loading mechanism
    for filename in os.listdir(folder_path):
        if filename.endswith('.dcm') or filename.endswith('.jpg') or filename.endswith('.png'):
            filepath = os.path.join(folder_path, filename)
            if filename.endswith('.dcm'):  # Handle DICOM files
                ds = pydicom.dcmread(filepath)
                img = ds.pixel_array
            else:
                img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (224, 224))
            images.append(img)
            # labels.append(  # Implement label extraction if available )
    images = np.array(images)
    images = images / 255.0  # Normalize pixel values to [0,1]
    images = images.reshape(-1, 224, 224, 1)  # Add channel dimension
    return images, np.array(labels)
