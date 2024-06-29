import tensorflow as tf
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

# Генератор зображень для тестової вибірки
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # Дуже важливо, щоб не перемішувати дані для аналізу
)

# Генерація передбачень для тестового набору
Y_pred = model.predict(test_generator, test_generator.samples // test_generator.batch_size + 1)
y_pred = np.argmax(Y_pred, axis=1)

# Отримання справжніх міток
y_true = test_generator.classes

# Побудова конфузійної матриці
cm = confusion_matrix(y_true, y_pred)
cm_df = pd.DataFrame(cm, index=classes, columns=classes)

plt.figure(figsize=(10, 8))
sns.heatmap(cm_df, annot=True, fmt='g', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()