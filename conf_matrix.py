import tensorflow as tf
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Завантаження моделі
model = tf.keras.models.load_model('image_classification_model.keras')

# Класи
classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# Шлях до тестової директорії
test_dir = './seg_test'

test_datagen = ImageDataGenerator(rescale=1./255)

# Генератор зображень для тест сету
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)
# Предікт для тест сету
Y_pred = model.predict(test_generator)
y_pred = np.argmax(Y_pred, axis=1)

# Отримання target y
y_true = test_generator.classes

# Створення confusion matrix
cm = confusion_matrix(y_true, y_pred)
cm_df = pd.DataFrame(cm, index=classes, columns=classes)

# Плотимо
plt.figure(figsize=(10, 8))
sns.heatmap(cm_df, annot=True, fmt='g', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()