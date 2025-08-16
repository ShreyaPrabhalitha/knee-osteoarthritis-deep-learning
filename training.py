import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from src.data_preprocessing import load_images_from_folder
from src.model_architecture import build_resnet50_model

# Load dataset
folder_path = './data/raw/'
X, y = load_images_from_folder(folder_path)

# For demonstration, creating dummy labels for 5 classes
if y.size == 0:
    y = np.random.randint(0, 5, X.shape[0])
y_cat = to_categorical(y, num_classes=5)

# Split dataset
X_train, X_val, y_train, y_val = train_test_split(X, y_cat, test_size=0.3, random_state=42)

# Build model
model = build_resnet50_model(num_classes=5)

# Train model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=16)

# Save the model
import os
os.makedirs('models', exist_ok=True)
model.save('models/knee_oa_resnet50.h5')
