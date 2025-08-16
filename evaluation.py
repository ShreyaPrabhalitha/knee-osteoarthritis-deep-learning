import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from src.data_preprocessing import load_images_from_folder
from tensorflow.keras.utils import to_categorical

# Load test data
folder_path = './data/test/'
X_test, y_test = load_images_from_folder(folder_path)

# Dummy labels if not available
if y_test.size == 0:
    y_test = np.random.randint(0, 5, X_test.shape[0])
y_test_cat = to_categorical(y_test, num_classes=5)

# Load saved model
model = load_model('models/knee_oa_resnet50.h5')

# Predictions
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

# Classification Report
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Confusion Matrix
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
